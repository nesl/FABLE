"""Versioned cross-module contracts for FABLE Phase 0."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, StringConstraints, UUID7, field_validator, model_validator

from .base import FableModel, FrozenFableModel, FrozenVersionedModel, JSONValue, VersionedModel
from .enums import (
    ArtifactAccessMode,
    ArtifactLocationKind,
    BindingCapability,
    CancellationScope,
    CheckpointKind,
    CheckpointStatus,
    ExecutionMode,
    GraphEdgeKind,
    GraphNodeKind,
    HypothesisLifecycle,
    HypothesisNodeStatus,
    NodeAvailability,
    PlanStatus,
    ProviderLeaseStatus,
    ProviderPortKind,
    ResultKind,
    TemporalGuardKind,
    TruthValue,
)
from .ids import (
    canonical_hypothesis_key,
    demand_sharing_key,
    physical_plan_label_id,
    uuid7,
)
from .time import (
    DeadlineSpec,
    EventTimeInterval,
    LatenessPolicy,
    SourceWatermark,
    ensure_utc,
    utc_now,
)

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class RoleDefinition(FableModel):
    role_name: NonEmptyStr
    entity_type: NonEmptyStr
    cardinality_min: int = Field(default=1, ge=0)
    cardinality_max: int | None = Field(default=1, ge=1)
    distinct_from: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_cardinality(self) -> "RoleDefinition":
        if self.cardinality_max is not None and self.cardinality_max < self.cardinality_min:
            raise ValueError("cardinality_max must be >= cardinality_min")
        return self


class PredicateRole(FableModel):
    role_name: NonEmptyStr
    variable: NonEmptyStr
    entity_type: NonEmptyStr


class SemanticPredicate(FableModel):
    """Provider-independent semantic condition."""

    predicate_id: NonEmptyStr
    roles: tuple[PredicateRole, ...] = ()
    parameters: dict[str, JSONValue] = Field(default_factory=dict)
    result_kind: ResultKind

    @model_validator(mode="after")
    def _unique_role_names(self) -> "SemanticPredicate":
        names = [role.role_name for role in self.roles]
        if len(names) != len(set(names)):
            raise ValueError("semantic predicate role names must be unique")
        return self


class TemporalGuard(FableModel):
    guard_id: NonEmptyStr
    kind: TemporalGuardKind
    source_node_ids: tuple[str, ...]
    target_node_id: str | None = None
    minimum_ms: int | None = Field(default=None, ge=0)
    maximum_ms: int | None = Field(default=None, ge=0)
    count: int | None = Field(default=None, ge=1)
    required_source_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_bounds(self) -> "TemporalGuard":
        if self.minimum_ms is not None and self.maximum_ms is not None:
            if self.maximum_ms < self.minimum_ms:
                raise ValueError("maximum_ms must be >= minimum_ms")
        if not self.source_node_ids:
            raise ValueError("temporal guard must reference at least one source node")
        if self.kind == TemporalGuardKind.REPEAT_WITHIN and self.count is None:
            raise ValueError("REPEAT_WITHIN requires count")
        return self


class GraphNode(FableModel):
    node_id: NonEmptyStr
    authored_key: NonEmptyStr
    kind: GraphNodeKind
    name: NonEmptyStr
    predicate: SemanticPredicate | None = None
    k: int | None = Field(default=None, ge=1)
    checkpoint_boundary: bool = False
    annotations: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_kind_fields(self) -> "GraphNode":
        if self.kind == GraphNodeKind.PREDICATE and self.predicate is None:
            raise ValueError("PREDICATE node requires predicate")
        if self.kind != GraphNodeKind.PREDICATE and self.predicate is not None:
            raise ValueError("only PREDICATE nodes may carry a semantic predicate")
        if self.kind == GraphNodeKind.K_OF_N and self.k is None:
            raise ValueError("K_OF_N node requires k")
        if self.kind != GraphNodeKind.K_OF_N and self.k is not None:
            raise ValueError("k is valid only for K_OF_N nodes")
        return self


class GraphEdge(FableModel):
    edge_id: NonEmptyStr
    source_node_id: NonEmptyStr
    target_node_id: NonEmptyStr
    kind: GraphEdgeKind
    temporal_guard_ids: tuple[str, ...] = ()
    branch_label: str | None = None


class SemanticGraph(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.semantic_graph.v1"
    schema_version: Literal["fable.semantic_graph.v1"] = SCHEMA_VERSION
    graph_id: NonEmptyStr
    graph_hash: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    graph_version: int = Field(default=1, ge=1)
    name: NonEmptyStr
    description: str | None = None
    root_node_id: NonEmptyStr
    roles: tuple[RoleDefinition, ...] = ()
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...] = ()
    temporal_guards: tuple[TemporalGuard, ...] = ()
    authored_variant_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_references(self) -> "SemanticGraph":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph node identifiers must be unique")
        authored_keys = [node.authored_key for node in self.nodes]
        if len(authored_keys) != len(set(authored_keys)):
            raise ValueError("graph authored keys must be unique")
        if self.root_node_id not in set(node_ids):
            raise ValueError("root_node_id does not reference a graph node")
        guard_ids = [guard.guard_id for guard in self.temporal_guards]
        if len(guard_ids) != len(set(guard_ids)):
            raise ValueError("temporal guard identifiers must be unique")
        node_set = set(node_ids)
        guard_set = set(guard_ids)
        for edge in self.edges:
            if edge.source_node_id not in node_set or edge.target_node_id not in node_set:
                raise ValueError(f"edge {edge.edge_id} references an unknown node")
            unknown_guards = set(edge.temporal_guard_ids) - guard_set
            if unknown_guards:
                raise ValueError(f"edge {edge.edge_id} references unknown guards {unknown_guards}")
        for guard in self.temporal_guards:
            if set(guard.source_node_ids) - node_set:
                raise ValueError(f"guard {guard.guard_id} references an unknown source node")
            if guard.target_node_id and guard.target_node_id not in node_set:
                raise ValueError(f"guard {guard.guard_id} references an unknown target node")
        return self


class EntityBinding(FableModel):
    role_name: NonEmptyStr
    entity_type: NonEmptyStr
    canonical_entity_id: NonEmptyStr
    local_entity_ids: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    established_by_occurrence_id: str | None = None


class HypothesisNodeState(FableModel):
    node_id: NonEmptyStr
    status: HypothesisNodeStatus = HypothesisNodeStatus.UNRESOLVED
    truth: TruthValue = TruthValue.UNKNOWN
    occurrence_ids: tuple[str, ...] = ()
    event_time_intervals: tuple[EventTimeInterval, ...] = ()
    last_updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("last_updated_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class Hypothesis(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.hypothesis.v1"
    schema_version: Literal["fable.hypothesis.v1"] = SCHEMA_VERSION
    hypothesis_id: UUID7 = Field(default_factory=uuid7)
    request_id: NonEmptyStr
    graph_id: NonEmptyStr
    graph_hash: NonEmptyStr
    graph_version: int = Field(ge=1)
    version: int = Field(default=0, ge=0)
    lifecycle: HypothesisLifecycle = HypothesisLifecycle.ACTIVE
    anchor_occurrence_id: NonEmptyStr
    role_bindings: dict[str, EntityBinding] = Field(default_factory=dict)
    structural_branch_ids: tuple[str, ...] = ()
    node_states: dict[str, HypothesisNodeState] = Field(default_factory=dict)
    event_time_window: EventTimeInterval
    deadline: DeadlineSpec
    frontier_id: UUID7 | None = None
    evidence_artifact_ids: tuple[UUID7, ...] = ()
    provenance_result_ids: tuple[UUID7, ...] = ()
    canonical_key: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at", "updated_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _derive_and_validate_key(self) -> "Hypothesis":
        for key, binding in self.role_bindings.items():
            if key != binding.role_name:
                raise ValueError("role_bindings keys must match EntityBinding.role_name")
        for key, state in self.node_states.items():
            if key != state.node_id:
                raise ValueError("node_states keys must match HypothesisNodeState.node_id")
        expected = canonical_hypothesis_key(
            request_id=self.request_id,
            graph_hash=self.graph_hash,
            anchor_occurrence_id=self.anchor_occurrence_id,
            canonical_bindings={
                key: {
                    "entity_type": binding.entity_type,
                    "canonical_entity_id": binding.canonical_entity_id,
                }
                for key, binding in sorted(self.role_bindings.items())
            },
            structural_branch_ids=self.structural_branch_ids,
        )
        if self.canonical_key is None:
            object.__setattr__(self, "canonical_key", expected)
        elif self.canonical_key != expected:
            raise ValueError("canonical_key does not match hypothesis identity fields")
        return self


class SemanticCheckpoint(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.semantic_checkpoint.v1"
    schema_version: Literal["fable.semantic_checkpoint.v1"] = SCHEMA_VERSION
    checkpoint_id: UUID7 = Field(default_factory=uuid7)
    hypothesis_id: UUID7
    hypothesis_version: int = Field(ge=0)
    kind: CheckpointKind
    node_ids: tuple[str, ...]
    status: CheckpointStatus = CheckpointStatus.OPEN
    event_time_interval: EventTimeInterval
    success_activates_node_ids: tuple[str, ...] = ()
    failure_activates_node_ids: tuple[str, ...] = ()
    branch_ids: tuple[str, ...] = ()
    required_artifact_types_after_resolution: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _require_nodes(self) -> "SemanticCheckpoint":
        if not self.node_ids:
            raise ValueError("semantic checkpoint must contain at least one node")
        return self


class FrontierSnapshot(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.frontier_snapshot.v1"
    schema_version: Literal["fable.frontier_snapshot.v1"] = SCHEMA_VERSION
    frontier_id: UUID7 = Field(default_factory=uuid7)
    request_id: NonEmptyStr
    graph_hash: NonEmptyStr
    hypothesis_id: UUID7
    hypothesis_version: int = Field(ge=0)
    enabled_node_ids: tuple[str, ...]
    checkpoint_ids: tuple[UUID7, ...]
    derived_at: datetime = Field(default_factory=utc_now)
    source_watermarks: dict[str, SourceWatermark] = Field(default_factory=dict)

    @field_validator("derived_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _require_frontier(self) -> "FrontierSnapshot":
        if not self.enabled_node_ids and not self.checkpoint_ids:
            raise ValueError("frontier must contain enabled nodes or checkpoints")
        return self


class DataMovementConstraints(FableModel):
    raw_data_must_remain_local: bool = True
    allowed_node_ids: tuple[str, ...] = ()
    allowed_regions: tuple[str, ...] = ()
    maximum_transfer_bytes: int | None = Field(default=None, ge=0)
    required_access_modes: tuple[ArtifactAccessMode, ...] = ()


class ContinuationRequirement(FableModel):
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

    There is intentionally no provider, model, image, container, or selected
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


class ProviderRoleCapability(FableModel):
    role_name: NonEmptyStr
    capabilities: tuple[BindingCapability, ...]


class ProviderSemanticCapabilities(FableModel):
    predicate_ids: tuple[str, ...]
    role_capabilities: tuple[ProviderRoleCapability, ...] = ()
    result_kinds: tuple[ResultKind, ...] = ()

    @model_validator(mode="after")
    def _require_predicate(self) -> "ProviderSemanticCapabilities":
        if not self.predicate_ids:
            raise ValueError("provider must implement at least one semantic predicate")
        return self


class ProviderPort(FableModel):
    name: NonEmptyStr
    kind: ProviderPortKind
    data_type: NonEmptyStr
    required: bool = True
    purpose: str | None = None


class ParameterSpec(FableModel):
    type: NonEmptyStr
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    enum: tuple[JSONValue, ...] = ()
    default: JSONValue = None

    @model_validator(mode="after")
    def _validate_range(self) -> "ParameterSpec":
        if self.minimum is not None and self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("parameter maximum must be >= minimum")
        return self


class ProviderExecutionCapabilities(FableModel):
    modes: tuple[ExecutionMode, ...]
    supports_shared_execution: bool = False
    accepted_input_access: tuple[ArtifactAccessMode, ...] = ()
    state_operations: tuple[str, ...] = ()


class CompatibilityGroup(FableModel):
    ports: tuple[str, ...]
    require_same_runtime_keys: tuple[str, ...] = ()


class ProviderContract(VersionedModel):
    """One executable implementation contract, never a hypothesis controller."""

    SCHEMA_VERSION: ClassVar[str] = "fable.provider_contract.v1"
    schema_version: Literal["fable.provider_contract.v1"] = SCHEMA_VERSION
    provider_id: NonEmptyStr
    contract_version: int = Field(default=1, ge=1)
    description: NonEmptyStr
    semantic_capabilities: ProviderSemanticCapabilities
    ports: tuple[ProviderPort, ...]
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    execution_capabilities: ProviderExecutionCapabilities
    compatibility_groups: tuple[CompatibilityGroup, ...] = ()
    eligible_node_classes: tuple[str, ...] = ()
    immutable_image_digest: str | None = None

    @model_validator(mode="after")
    def _validate_ports(self) -> "ProviderContract":
        names = [port.name for port in self.ports]
        if len(names) != len(set(names)):
            raise ValueError("provider port names must be unique")
        input_names = {
            port.name
            for port in self.ports
            if port.kind in (ProviderPortKind.INPUT, ProviderPortKind.STATE_INPUT)
        }
        for group in self.compatibility_groups:
            unknown = set(group.ports) - input_names
            if unknown:
                raise ValueError(f"compatibility group references unknown input ports: {unknown}")
        return self


class ProviderFamily(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.provider_family.v1"
    schema_version: Literal["fable.provider_family.v1"] = SCHEMA_VERSION
    family_id: NonEmptyStr
    description: NonEmptyStr
    predicate_ids: tuple[str, ...]
    provider_contract_ids: tuple[str, ...]
    acceptable_input_types: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _require_members(self) -> "ProviderFamily":
        if not self.predicate_ids:
            raise ValueError("provider family requires at least one predicate")
        if not self.provider_contract_ids:
            raise ValueError("provider family requires at least one contract")
        return self


class ArtifactLocation(FableModel):
    kind: ArtifactLocationKind
    node_id: str | None = None
    uri: str | None = None
    inline_data_base64: str | None = None

    @model_validator(mode="after")
    def _validate_location(self) -> "ArtifactLocation":
        if self.kind == ArtifactLocationKind.INLINE:
            if not self.inline_data_base64:
                raise ValueError("INLINE artifact location requires inline_data_base64")
        elif not self.uri:
            raise ValueError("non-inline artifact location requires uri")
        return self


class ArtifactProducer(FableModel):
    provider_id: NonEmptyStr
    provider_contract_version: int = Field(ge=1)
    model_id: str | None = None
    model_version: str | None = None


class ArtifactRef(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.artifact_ref.v1"
    schema_version: Literal["fable.artifact_ref.v1"] = SCHEMA_VERSION
    artifact_id: UUID7 = Field(default_factory=uuid7)
    artifact_type: NonEmptyStr
    artifact_schema_version: NonEmptyStr
    producer: ArtifactProducer
    event_time_interval: EventTimeInterval
    bindings: dict[str, str] = Field(default_factory=dict)
    location: ArtifactLocation
    access_modes: tuple[ArtifactAccessMode, ...]
    compatibility_keys: dict[str, JSONValue] = Field(default_factory=dict)
    compatible_consumer_families: tuple[str, ...] = ()
    bytes: int | None = Field(default=None, ge=0)
    checksum_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    created_at: datetime = Field(default_factory=utc_now)
    valid_until: datetime | None = None
    expires_at: datetime | None = None
    policy_tags: tuple[str, ...] = ()

    @field_validator("created_at", "valid_until", "expires_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    @model_validator(mode="after")
    def _validate_lifetime(self) -> "ArtifactRef":
        if not self.access_modes:
            raise ValueError("artifact must expose at least one access mode")
        if self.valid_until and self.valid_until < self.event_time_interval.end:
            raise ValueError("valid_until cannot precede the artifact event interval")
        if self.expires_at and self.expires_at < self.created_at:
            raise ValueError("expires_at cannot precede created_at")
        return self


class PlanStep(FrozenFableModel):
    step_id: NonEmptyStr
    provider_id: NonEmptyStr
    node_id: NonEmptyStr
    input_artifact_ids: tuple[UUID7, ...] = ()
    input_data_types: tuple[str, ...] = ()
    output_data_types: tuple[str, ...] = ()
    parameters: tuple[tuple[str, JSONValue], ...] = ()
    depends_on_step_ids: tuple[str, ...] = ()
    estimated_startup_ms: int = Field(default=0, ge=0)
    estimated_execution_ms: int = Field(default=0, ge=0)
    estimated_transfer_ms: int = Field(default=0, ge=0)
    estimated_transfer_bytes: int = Field(default=0, ge=0)


class PlanCost(FrozenFableModel):
    predicted_completion_ms: int = Field(ge=0)
    deadline_slack_ms: int
    startup_cost_ms: int = Field(ge=0)
    resource_cost_units: float = Field(ge=0)
    transfer_bytes: int = Field(ge=0)


class PhysicalPlanLabel(FrozenVersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.physical_plan_label.v1"
    schema_version: Literal["fable.physical_plan_label.v1"] = SCHEMA_VERSION
    label_id: str | None = None
    checkpoint_id: UUID7
    covered_demand_ids: tuple[UUID7, ...]
    steps: tuple[PlanStep, ...]
    input_artifact_ids: tuple[UUID7, ...] = ()
    continuation_output_types: tuple[str, ...] = ()
    cost: PlanCost
    hard_constraints_satisfied: bool
    quality_floor_satisfied: bool
    feasibility_reasons: tuple[str, ...] = ()
    parent_label_id: str | None = None

    @model_validator(mode="after")
    def _derive_label_id(self) -> "PhysicalPlanLabel":
        if not self.covered_demand_ids:
            raise ValueError("physical plan label must cover at least one demand")
        payload = self.model_dump(mode="python", exclude={"label_id", "schema_version"}, exclude_none=True)
        expected = physical_plan_label_id(payload)
        if self.label_id is None:
            object.__setattr__(self, "label_id", expected)
        elif self.label_id != expected:
            raise ValueError("label_id does not match label content")
        return self


class ResourceReservation(FableModel):
    node_id: NonEmptyStr
    cpu_cores: float = Field(default=0, ge=0)
    memory_mb: int = Field(default=0, ge=0)
    gpu_memory_mb: int = Field(default=0, ge=0)
    network_bytes: int = Field(default=0, ge=0)


class ExecutionPlan(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.execution_plan.v1"
    schema_version: Literal["fable.execution_plan.v1"] = SCHEMA_VERSION
    plan_id: UUID7 = Field(default_factory=uuid7)
    label_id: NonEmptyStr
    checkpoint_id: UUID7
    demand_ids: tuple[UUID7, ...]
    steps: tuple[PlanStep, ...]
    reservations: tuple[ResourceReservation, ...] = ()
    status: PlanStatus = PlanStatus.CANDIDATE
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None

    @field_validator("created_at", "expires_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)


class BindingDelta(FableModel):
    introduced: dict[str, str] = Field(default_factory=dict)
    validated: dict[str, str] = Field(default_factory=dict)
    rejected_roles: tuple[str, ...] = ()


class ResultProvenance(FableModel):
    provider_id: NonEmptyStr
    provider_contract_version: int = Field(ge=1)
    node_id: NonEmptyStr
    source_ids: tuple[str, ...]
    source_sequence_ranges: dict[str, tuple[int, int]] = Field(default_factory=dict)
    model_id: str | None = None
    model_version: str | None = None


class PredicateResult(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.predicate_result.v1"
    schema_version: Literal["fable.predicate_result.v1"] = SCHEMA_VERSION
    result_id: UUID7 = Field(default_factory=uuid7)
    occurrence_id: NonEmptyStr
    demand_id: UUID7
    request_id: NonEmptyStr
    graph_hash: NonEmptyStr
    hypothesis_id: UUID7
    expected_hypothesis_version: int = Field(ge=0)
    frontier_id: UUID7
    checkpoint_id: UUID7
    graph_node_id: NonEmptyStr
    semantic_predicate: SemanticPredicate
    truth: TruthValue
    confidence: float | None = Field(default=None, ge=0, le=1)
    event_time_interval: EventTimeInterval
    binding_delta: BindingDelta = Field(default_factory=BindingDelta)
    artifact_ids: tuple[UUID7, ...] = ()
    provenance: ResultProvenance
    processing_started_at: datetime
    processing_completed_at: datetime

    @field_validator("processing_started_at", "processing_completed_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _validate_processing_order(self) -> "PredicateResult":
        if self.processing_completed_at < self.processing_started_at:
            raise ValueError("processing_completed_at cannot precede processing_started_at")
        return self


class ProviderLease(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.provider_lease.v1"
    schema_version: Literal["fable.provider_lease.v1"] = SCHEMA_VERSION
    lease_id: UUID7 = Field(default_factory=uuid7)
    provider_instance_id: NonEmptyStr
    provider_id: NonEmptyStr
    provider_contract_version: int = Field(ge=1)
    demand_id: UUID7
    plan_id: UUID7
    node_id: NonEmptyStr
    configuration_hash: NonEmptyStr
    status: ProviderLeaseStatus = ProviderLeaseStatus.REQUESTED
    starts_at: datetime
    expires_at: datetime
    attempt_id: UUID7 = Field(default_factory=uuid7)

    @field_validator("starts_at", "expires_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _validate_lease_time(self) -> "ProviderLease":
        if self.expires_at <= self.starts_at:
            raise ValueError("provider lease expires_at must be after starts_at")
        return self


class SourceHeartbeat(FableModel):
    source_id: NonEmptyStr
    latest_sequence: int = Field(ge=0)
    latest_event_time: datetime
    raw_buffer_interval: EventTimeInterval | None = None
    operational_coverage: bool = True

    @field_validator("latest_event_time")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class NodeCapacity(FableModel):
    cpu_free_cores: float = Field(ge=0)
    memory_free_mb: int = Field(ge=0)
    gpu_free_mb: int = Field(default=0, ge=0)
    network_tx_available_mbps: float | None = Field(default=None, ge=0)
    network_rx_available_mbps: float | None = Field(default=None, ge=0)


class NodeHeartbeat(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.node_heartbeat.v1"
    schema_version: Literal["fable.node_heartbeat.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    node_id: NonEmptyStr
    session_id: NonEmptyStr
    sequence: int = Field(ge=0)
    sent_at: datetime = Field(default_factory=utc_now)
    availability: NodeAvailability = NodeAvailability.AVAILABLE
    sources: dict[str, SourceHeartbeat] = Field(default_factory=dict)
    active_provider_instance_ids: tuple[str, ...] = ()
    active_demand_ids: tuple[UUID7, ...] = ()
    capacity: NodeCapacity

    @field_validator("sent_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _keys_match_sources(self) -> "NodeHeartbeat":
        mismatches = [key for key, value in self.sources.items() if key != value.source_id]
        if mismatches:
            raise ValueError(f"heartbeat source keys must match source_id: {mismatches}")
        return self
