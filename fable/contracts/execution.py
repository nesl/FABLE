"""Execution data contracts for FABLE."""

from ._shared import *  # noqa: F401,F403

class ExecutionInput(FrozenFableModel):
    """Fully resolved external input consumed by an execution step."""

    name: NonEmptyStr
    data_type: NonEmptyStr
    kind: ExecutionInputKind
    node_id: str | None = None
    source_id: str | None = None
    artifact_id: UUID7 | None = None
    bytes: int = Field(default=0, ge=0)
    expires_at: datetime | None = None

    @field_validator("expires_at")
    @classmethod
    def _normalize_expires_at(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

class PlanStep(FrozenFableModel):
    """Concrete execution step emitted by planning.

    The record intentionally contains the information required downstream by
    scheduling/execution so those layers do not need planner-internal
    ``PhysicalAlternative`` or ``StepPlacement`` objects.
    """

    step_id: NonEmptyStr
    provider_id: NonEmptyStr
    node_id: NonEmptyStr
    demand_id: UUID7 | None = None
    alternative_id: str | None = None
    chain_id: str | None = None
    execution_mode: ExecutionMode = ExecutionMode.LIVE
    inputs: tuple[ExecutionInput, ...] = ()
    input_artifact_ids: tuple[UUID7, ...] = ()
    input_data_types: tuple[str, ...] = ()
    output_data_types: tuple[str, ...] = ()
    parameters: tuple[tuple[str, JSONValue], ...] = ()
    depends_on_step_ids: tuple[str, ...] = ()
    cpu_cores: float = Field(default=0.0, ge=0)
    memory_mb: int = Field(default=0, ge=0)
    gpu_memory_mb: int = Field(default=0, ge=0)
    quality_score: float = Field(default=1.0, ge=0, le=1)
    reused_provider_instance_id: str | None = None
    estimated_startup_ms: int = Field(default=0, ge=0)
    estimated_execution_ms: int = Field(default=0, ge=0)
    estimated_transfer_ms: int = Field(default=0, ge=0)
    estimated_transfer_bytes: int = Field(default=0, ge=0)

    @property
    def source_signature(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    f"source:{item.source_id}"
                    for item in self.inputs
                    if item.source_id is not None
                }
                | {
                    f"node:{item.node_id}"
                    for item in self.inputs
                    if item.node_id is not None
                }
            )
        )

class PlanCost(FrozenFableModel):
    """Comparable completion, slack, startup, resource, and transfer estimates."""
    predicted_completion_ms: int = Field(ge=0)
    deadline_slack_ms: int
    startup_cost_ms: int = Field(ge=0)
    resource_cost_units: float = Field(ge=0)
    transfer_bytes: int = Field(ge=0)

class PhysicalPlanLabel(FrozenVersionedModel):
    """Immutable complete/partial plan state considered by beam search."""
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
        payload = self.model_dump(
            mode="python", exclude={"label_id", "schema_version"}, exclude_none=True
        )
        expected = physical_plan_label_id(payload)
        if self.label_id is None:
            object.__setattr__(self, "label_id", expected)
            return self
        if self.label_id == expected:
            return self

        # ``PlanStep`` was enriched without changing the v1 wire schema. Accept
        # labels written by the historical v1 projection so persisted fixtures
        # and checkpoints remain readable. Newly created labels hash the full
        # self-contained execution step above.
        legacy_payload = dict(payload)
        legacy_steps = []
        enriched_fields = {
            "demand_id",
            "alternative_id",
            "chain_id",
            "execution_mode",
            "inputs",
            "cpu_cores",
            "memory_mb",
            "gpu_memory_mb",
            "quality_score",
            "reused_provider_instance_id",
        }
        for step in legacy_payload.get("steps", ()):
            legacy_steps.append(
                {key: value for key, value in step.items() if key not in enriched_fields}
            )
        legacy_payload["steps"] = tuple(legacy_steps)
        if self.label_id != physical_plan_label_id(legacy_payload):
            raise ValueError("label_id does not match label content")
        return self

class ResourceReservation(FableModel):
    """CPU, RAM, GPU-memory, and network capacity reserved on one node."""
    node_id: NonEmptyStr
    cpu_cores: float = Field(default=0, ge=0)
    memory_mb: int = Field(default=0, ge=0)
    gpu_memory_mb: int = Field(default=0, ge=0)
    network_bytes: int = Field(default=0, ge=0)

class ExecutionPlan(VersionedModel):
    """Concrete provider steps and reservations selected by physical planning.

    The scheduler admits this plan and node execution starts its workers.
    """
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
