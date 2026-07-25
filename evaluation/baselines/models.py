"""Controlled-baseline planning inputs and inspectable decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from pydantic import Field

from fable.common.base import FableModel
from fable.common.schemas import PredicateDemand
from fable.planning.models import PhysicalAlternativeGraph
from fable.common.time import ensure_utc

from evaluation.schemas import BaselineId


@dataclass(frozen=True)
class BaselinePlanningCase:
    run_id: str
    trace_id: str
    request_id: str
    event_family: str
    frontier_demands: tuple[PredicateDemand, ...]
    all_task_demands: tuple[PredicateDemand, ...]
    frontier_graph: PhysicalAlternativeGraph
    whole_event_graph: PhysicalAlternativeGraph
    now: datetime
    replay_supported_sensor_ids: tuple[str, ...] = ()
    resource_epoch: int = 0
    semantic_epoch: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "now", ensure_utc(self.now))


class BaselineDecision(FableModel):
    baseline_id: BaselineId
    request_id: str = Field(min_length=1)
    checkpoint_id: UUID
    planning_scope: str = Field(min_length=1)
    selected_alternative_ids: tuple[str, ...] = ()
    selected_chain_ids: tuple[str, ...] = ()
    selected_node_ids: tuple[str, ...] = ()
    selected_source_ids: tuple[str, ...] = ()
    activated_provider_keys: tuple[str, ...] = ()
    continuation_types: tuple[str, ...] = ()
    predicted_completion_ms: int | None = Field(default=None, ge=0)
    predicted_transfer_bytes: int | None = Field(default=None, ge=0)
    planning_latency_ms: float = Field(default=0, ge=0)
    labels_generated: int = Field(default=0, ge=0)
    labels_pruned: int = Field(default=0, ge=0)
    labels_retained: int = Field(default=0, ge=0)
    oracle_gap_ms: int | None = None
    frozen: bool = False
    resource_epoch: int = Field(default=0, ge=0)
    semantic_epoch: int = Field(default=0, ge=0)
    reason: str = ""
    excluded_mobile_or_unavailable_sources: tuple[str, ...] = ()
