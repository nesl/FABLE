"""Typed, bounded LLM contracts for request interpretation and checkpoint advice."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from fable.common.base import FableModel
from fable.common.time import ensure_utc, utc_now
from fable.spatial.models import SpatialPrediction


class CheckpointAdvisorRequest(FableModel):
    request_id: str = Field(min_length=1)
    graph_hash: str = Field(min_length=1)
    hypothesis_id: UUID
    hypothesis_version: int = Field(ge=0)
    frontier_id: UUID
    checkpoint_ids: tuple[UUID, ...]
    active_graph_node_ids: tuple[str, ...]
    eligible_branch_ids: tuple[str, ...] = ()
    replayable_branch_ids: tuple[str, ...] = ()
    bound_roles: dict[str, str] = Field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    spatial_prediction: SpatialPrediction | None = None
    requested_at: datetime = Field(default_factory=utc_now)

    @field_validator("requested_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class BranchPriorityAdjustment(FableModel):
    branch_id: str = Field(min_length=1)
    adjustment: int = Field(ge=-1, le=1)
    reason: str = Field(min_length=1)


class CheckpointAdvisorHint(FableModel):
    ordered_branch_ids: tuple[str, ...] = ()
    priority_adjustments: tuple[BranchPriorityAdjustment, ...] = ()
    explanation: str = ""
    evidence_refs: tuple[str, ...] = ()
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _unique_branches(self) -> "CheckpointAdvisorHint":
        if len(self.ordered_branch_ids) != len(set(self.ordered_branch_ids)):
            raise ValueError("ordered_branch_ids must be unique")
        adjusted = [item.branch_id for item in self.priority_adjustments]
        if len(adjusted) != len(set(adjusted)):
            raise ValueError("priority_adjustments must be unique by branch")
        return self


class ValidatedCheckpointHint(FableModel):
    hint: CheckpointAdvisorHint
    accepted_branch_order: tuple[str, ...]
    accepted_adjustments: tuple[BranchPriorityAdjustment, ...]
    ignored_reasons: tuple[str, ...] = ()
