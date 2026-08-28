"""Result data contracts for FABLE."""

from ._shared import *  # noqa: F401,F403
from .semantic import SemanticPredicate

class BindingDelta(FableModel):
    """CE role identities introduced, validated, or rejected by one result."""
    introduced: dict[str, str] = Field(default_factory=dict)
    validated: dict[str, str] = Field(default_factory=dict)
    rejected_roles: tuple[str, ...] = ()

class ResultProvenance(FableModel):
    """Execution provenance attached by a provider/result adapter.

    Provider contract version describes interface compatibility; model version
    identifies the actual ML artifact. They intentionally evolve separately.
    """
    provider_id: NonEmptyStr
    provider_contract_version: int = Field(ge=1)
    node_id: NonEmptyStr
    source_ids: tuple[str, ...]
    source_sequence_ranges: dict[str, tuple[int, int]] = Field(default_factory=dict)
    model_id: str | None = None
    model_version: str | None = None

class PredicateResult(VersionedModel):
    """Provider evidence normalized for consumption by ``SemanticRuntime``."""
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

class TerminalComplexEvent(VersionedModel):
    """Canonical terminal output of the semantic runtime.

    Evaluation and applications consume this record instead of reconstructing a
    completed event from control logs or persistence internals.
    """

    SCHEMA_VERSION: ClassVar[str] = "fable.terminal_complex_event.v1"
    schema_version: Literal["fable.terminal_complex_event.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    request_id: NonEmptyStr
    family_id: NonEmptyStr
    hypothesis_id: UUID7
    graph_hash: NonEmptyStr
    bindings: dict[str, str] = Field(default_factory=dict)
    event_time_window: EventTimeInterval
    provenance_result_ids: tuple[UUID7, ...] = ()
    emitted_at: datetime = Field(default_factory=utc_now)

    @field_validator("emitted_at")
    @classmethod
    def _normalize_emitted_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)
