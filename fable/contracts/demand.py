"""Demand data contracts for FABLE."""

from ._shared import *  # noqa: F401,F403
from .semantic import SemanticPredicate

class DataMovementConstraints(FableModel):
    """Application hard limits on where and how evidence may move."""
    raw_data_must_remain_local: bool = True
    allowed_node_ids: tuple[str, ...] = ()
    allowed_regions: tuple[str, ...] = ()
    maximum_transfer_bytes: int | None = Field(default=None, ge=0)
    required_access_modes: tuple[ArtifactAccessMode, ...] = ()

class ContinuationRequirement(FableModel):
    """Evidence that must remain usable after the current checkpoint resolves."""
    artifact_type: NonEmptyStr
    required_until: datetime
    compatible_consumer_families: tuple[str, ...] = ()
    required_bindings: tuple[str, ...] = ()

    @field_validator("required_until")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)

class SourcePreference(FableModel):
    """Soft source ordering derived from topology, motion, or other context.

    This is advisory metadata. Hard source eligibility remains in
    ``eligible_source_ids`` and policy constraints.
    """

    source_id: NonEmptyStr
    priority_rank: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: NonEmptyStr
    sensor_id: str | None = None
    observation_group: int | None = Field(default=None, ge=1)
    corridor_id: str | None = None
    prediction_id: str | None = None

class DemandBindingPolicy(FableModel):
    """How one specialized demand may consume, validate, or introduce bindings."""

    role_modes: dict[str, BindingCapability] = Field(default_factory=dict)
    forkable_roles: tuple[str, ...] = ()
    fork_unit: NonEmptyStr = "CANONICAL_BINDING_TUPLE"

    @model_validator(mode="after")
    def _validate_fork_roles(self) -> "DemandBindingPolicy":
        unknown = set(self.forkable_roles) - set(self.role_modes)
        if unknown:
            raise ValueError(f"forkable roles are missing role modes: {sorted(unknown)}")
        return self

class PredicateDemand(VersionedModel):
    """The provider-independent scheduling boundary.

    DemandCompiler creates it and PhysicalPlanner consumes it. There is
    intentionally no provider, model, image, container, or selected
    node field in this schema. Those choices belong to physical alternatives.
    """

    SCHEMA_VERSION: ClassVar[str] = "fable.predicate_demand.v1"
    schema_version: Literal["fable.predicate_demand.v1"] = SCHEMA_VERSION
    demand_id: UUID7 = Field(default_factory=uuid7)
    request_id: NonEmptyStr
    graph_hash: NonEmptyStr
    hypothesis_id: UUID7
    hypothesis_version: int = Field(ge=0)
    frontier_id: UUID7
    checkpoint_id: UUID7
    graph_node_id: NonEmptyStr
    semantic_predicate: SemanticPredicate
    bound_roles: dict[str, str] = Field(default_factory=dict)
    unbound_roles: tuple[str, ...] = ()
    event_time_interval: EventTimeInterval
    deadline: DeadlineSpec
    lateness_policy: LatenessPolicy = Field(default_factory=LatenessPolicy)
    eligible_source_ids: tuple[str, ...] = ()
    eligible_regions: tuple[str, ...] = ()
    source_preferences: tuple[SourcePreference, ...] = ()
    spatial_prediction_id: str | None = None
    required_capabilities: tuple[str, ...] = ()
    required_input_artifact_types: tuple[str, ...] = ()
    acceptable_output_types: tuple[str, ...]
    hard_constraints: DataMovementConstraints = Field(default_factory=DataMovementConstraints)
    continuation_requirements: tuple[ContinuationRequirement, ...] = ()
    desired_continuation_types: tuple[str, ...] = ()
    retrospective_context: dict[str, JSONValue] | None = None
    binding_policy: DemandBindingPolicy = Field(default_factory=DemandBindingPolicy)
    cancellation_scope: CancellationScope = CancellationScope.BRANCH
    sharing_key: str | None = None

    @model_validator(mode="after")
    def _validate_roles_and_derive_key(self) -> "PredicateDemand":
        predicate_roles = {role.role_name for role in self.semantic_predicate.roles}
        referenced_roles = set(self.bound_roles) | set(self.unbound_roles)
        if not referenced_roles.issubset(predicate_roles):
            raise ValueError("bound_roles/unbound_roles reference roles absent from semantic_predicate")
        if set(self.bound_roles) & set(self.unbound_roles):
            raise ValueError("a role cannot be both bound and unbound")
        unknown_policy_roles = set(self.binding_policy.role_modes) - predicate_roles
        if unknown_policy_roles:
            raise ValueError("binding_policy references roles absent from semantic_predicate")
        if set(self.binding_policy.forkable_roles) - set(self.unbound_roles):
            raise ValueError("only unbound roles may be forkable for this demand")
        if not self.acceptable_output_types:
            raise ValueError("at least one acceptable output type is required")
        preference_ids = [item.source_id for item in self.source_preferences]
        if len(preference_ids) != len(set(preference_ids)):
            raise ValueError("source_preferences must contain unique source IDs")
        if self.eligible_source_ids and not set(preference_ids).issubset(
            set(self.eligible_source_ids)
        ):
            raise ValueError("source_preferences must be eligible for the demand")
        if self.spatial_prediction_id is not None:
            mismatched = [
                item.source_id
                for item in self.source_preferences
                if item.prediction_id not in (None, self.spatial_prediction_id)
            ]
            if mismatched:
                raise ValueError(
                    "source preference prediction IDs do not match spatial_prediction_id"
                )
        expected = demand_sharing_key(
            semantic_predicate=self.semantic_predicate,
            event_time_interval=self.event_time_interval,
            acceptable_output_types=self.acceptable_output_types,
            hard_constraints=self.hard_constraints,
        )
        if self.sharing_key is None:
            object.__setattr__(self, "sharing_key", expected)
        elif self.sharing_key != expected:
            raise ValueError("sharing_key does not match semantic demand fields")
        return self
