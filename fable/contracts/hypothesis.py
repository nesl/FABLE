"""Hypothesis data contracts for FABLE."""

from ._shared import *  # noqa: F401,F403

class EntityBinding(FableModel):
    """Canonical CE role identity plus its known sensor-local aliases."""
    role_name: NonEmptyStr
    entity_type: NonEmptyStr
    canonical_entity_id: NonEmptyStr
    local_entity_ids: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    established_by_occurrence_id: str | None = None

class HypothesisNodeState(FableModel):
    """Truth/progress and occurrence evidence for one static graph node."""
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
    """Runtime progress for one candidate occurrence of a static event graph.

    ``SemanticRuntime`` creates and updates it; demand compilation and the
    controller read it. ``version`` is optimistic runtime-state concurrency,
    while ``graph_hash`` identifies the immutable authored definition.
    """
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
    """Planning, temporal, and cancellation context for active predicates."""
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
    """One hypothesis's currently useful graph-node IDs at a specific version.

    This is not one ``GraphNode``: a concurrent AND can expose several active
    predicates in the same snapshot. ``enabled_node_ids`` is retained as the
    v1 serialized name; use ``active_predicate_node_ids`` in readable code.
    """
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

    @property
    def active_predicate_node_ids(self) -> tuple[str, ...]:
        return self.enabled_node_ids
