"""Typed records emitted by the Phase-4 bounded label-driven planner.

These records make planner behavior inspectable.  A search label is not a
semantic hypothesis: it is an immutable partial physical realization through
one semantic checkpoint.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from fable.common.base import FableModel, FrozenFableModel
from fable.common.schemas import ExecutionPlan, PhysicalPlanLabel
from fable.common.time import ensure_utc, utc_now


class PruneCode(StrEnum):
    PHASE3_PRUNED = "PHASE3_PRUNED"
    CHECKPOINT_MISMATCH = "CHECKPOINT_MISMATCH"
    DEMAND_MISMATCH = "DEMAND_MISMATCH"
    RESULT_SCHEMA_INCOMPATIBLE = "RESULT_SCHEMA_INCOMPATIBLE"
    INPUT_SCHEMA_INCOMPATIBLE = "INPUT_SCHEMA_INCOMPATIBLE"
    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    ARTIFACT_EXPIRED = "ARTIFACT_EXPIRED"
    EVENT_TIME_UNAVAILABLE = "EVENT_TIME_UNAVAILABLE"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    BINDING_CAPABILITY_MISSING = "BINDING_CAPABILITY_MISSING"
    REQUIRED_CAPABILITY_MISSING = "REQUIRED_CAPABILITY_MISSING"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    QUALITY_FLOOR = "QUALITY_FLOOR"
    CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"
    DEADLINE_INFEASIBLE = "DEADLINE_INFEASIBLE"
    CONTINUATION_INCOMPATIBLE = "CONTINUATION_INCOMPATIBLE"
    CHECKPOINT_CONTINUATION_INCOMPATIBLE = "CHECKPOINT_CONTINUATION_INCOMPATIBLE"
    DUPLICATE_LABEL = "DUPLICATE_LABEL"
    DOMINATED = "DOMINATED"
    BEAM_LIMIT = "BEAM_LIMIT"
    ORACLE_LIMIT = "ORACLE_LIMIT"


class NodeResourceFootprint(FrozenFableModel):
    node_id: str = Field(min_length=1)
    cpu_cores: float = Field(default=0, ge=0)
    memory_mb: int = Field(default=0, ge=0)
    gpu_memory_mb: int = Field(default=0, ge=0)


class LabelSearchState(FrozenFableModel):
    """Immutable planner state wrapping the cross-module PhysicalPlanLabel."""

    label: PhysicalPlanLabel
    selected_alternative_ids: tuple[str, ...]
    selected_chain_ids: tuple[str, ...]
    node_resources: tuple[NodeResourceFootprint, ...]
    completion_by_demand_ms: tuple[tuple[UUID, int], ...]
    minimum_quality_score: float = Field(ge=0, le=1)
    perishability_rank: int = Field(ge=0)
    spatial_preference_penalty: int = Field(default=0, ge=0)
    continuation_consumer_set: tuple[str, ...] = ()
    missing_desired_continuation_types: tuple[str, ...] = ()
    total_cpu_cores: float = Field(default=0, ge=0)
    total_memory_mb: int = Field(default=0, ge=0)
    total_gpu_memory_mb: int = Field(default=0, ge=0)
    expires_at: datetime | None = None

    @field_validator("expires_at")
    @classmethod
    def _normalize_expires_at(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    @model_validator(mode="after")
    def _validate_alignment(self) -> "LabelSearchState":
        if len(self.selected_alternative_ids) != len(self.label.covered_demand_ids):
            raise ValueError("one selected alternative is required for each covered demand")
        if len(self.selected_chain_ids) != len(self.selected_alternative_ids):
            raise ValueError("selected_chain_ids must align with selected alternatives")
        completion_ids = {item[0] for item in self.completion_by_demand_ms}
        if completion_ids != set(self.label.covered_demand_ids):
            raise ValueError("completion_by_demand_ms must cover the label demands exactly")
        return self

    @property
    def label_id(self) -> str:
        assert self.label.label_id is not None
        return self.label.label_id

    def resource_map(self) -> dict[str, NodeResourceFootprint]:
        return {item.node_id: item for item in self.node_resources}


class FeasibilityFailure(FrozenFableModel):
    code: PruneCode
    reason: str = Field(min_length=1)


class PruningRecord(FrozenFableModel):
    boundary_index: int = Field(ge=0)
    code: PruneCode
    reason: str = Field(min_length=1)
    demand_id: UUID | None = None
    alternative_id: str | None = None
    label_id: str | None = None
    parent_label_id: str | None = None
    dominated_by_label_id: str | None = None


class BeamBoundaryTrace(FableModel):
    boundary_index: int = Field(ge=0)
    demand_id: UUID | None = None
    generated_label_ids: tuple[str, ...] = ()
    feasible_label_ids: tuple[str, ...] = ()
    retained_label_ids: tuple[str, ...] = ()
    pruning_records: tuple[PruningRecord, ...] = ()


class OracleStatus(StrEnum):
    NOT_RUN = "NOT_RUN"
    MATCHED = "MATCHED"
    GAP = "GAP"
    NO_FEASIBLE_PLAN = "NO_FEASIBLE_PLAN"
    SKIPPED_LIMIT = "SKIPPED_LIMIT"


class OracleComparison(FableModel):
    status: OracleStatus = OracleStatus.NOT_RUN
    combinations_considered: int = Field(default=0, ge=0)
    oracle_label_id: str | None = None
    selected_label_id: str | None = None
    completion_gap_ms: int | None = None
    startup_gap_ms: int | None = None
    transfer_gap_bytes: int | None = None
    reason: str = ""


class PlanSearchTrace(FableModel):
    search_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    checkpoint_id: UUID
    beam_width: int = Field(ge=1)
    demand_order: tuple[UUID, ...]
    boundaries: tuple[BeamBoundaryTrace, ...] = ()
    phase3_pruning: tuple[PruningRecord, ...] = ()
    selected_label_id: str | None = None
    fallback_label_ids: tuple[str, ...] = ()
    required_checkpoint_consumers: tuple[str, ...] = ()
    selection_rank: tuple[Any, ...] = ()
    oracle: OracleComparison = Field(default_factory=OracleComparison)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def _normalize_created_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class PlanSearchResult(FableModel):
    selected: LabelSearchState | None = None
    fallbacks: tuple[LabelSearchState, ...] = ()
    execution_plan: ExecutionPlan | None = None
    trace: PlanSearchTrace

    @model_validator(mode="after")
    def _validate_selection(self) -> "PlanSearchResult":
        if self.selected is None and self.execution_plan is not None:
            raise ValueError("execution_plan requires a selected label")
        if self.selected is not None and self.execution_plan is None:
            raise ValueError("selected label requires an execution_plan")
        return self
