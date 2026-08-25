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
    placement_id: str = ""
    replay_supported_sensor_ids: tuple[str, ...] = ()
    resource_epoch: int = 0
    semantic_epoch: int = 0
    graph_version: int = 1
    replan_trigger: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "now", ensure_utc(self.now))


@dataclass(frozen=True)
class TaskResourcePlanningCase(BaselinePlanningCase):
    """B3's task-level view of the common planning case.

    The separate type is an architectural guard: B3 may replan the complete
    task on resource epochs, but it must not be handed additional semantic
    state beyond the common controlled case.
    """

    @classmethod
    def from_case(cls, case: BaselinePlanningCase) -> "TaskResourcePlanningCase":
        return cls(
            run_id=case.run_id,
            trace_id=case.trace_id,
            request_id=case.request_id,
            event_family=case.event_family,
            frontier_demands=case.frontier_demands,
            all_task_demands=case.all_task_demands,
            frontier_graph=case.frontier_graph,
            whole_event_graph=case.whole_event_graph,
            now=case.now,
            placement_id=case.placement_id,
            replay_supported_sensor_ids=case.replay_supported_sensor_ids,
            resource_epoch=case.resource_epoch,
            semantic_epoch=case.semantic_epoch,
            graph_version=case.graph_version,
            replan_trigger=case.replan_trigger,
        )


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
    predicted_compute_ms: int | None = Field(default=None, ge=0)
    predicted_slack_ms: int | None = None
    planning_latency_ms: float = Field(default=0, ge=0)
    labels_generated: int = Field(default=0, ge=0)
    labels_pruned: int = Field(default=0, ge=0)
    labels_retained: int = Field(default=0, ge=0)
    pruning_counts: dict[str, int] = Field(default_factory=dict)
    pruning_samples: tuple[str, ...] = ()
    oracle_gap_ms: int | None = None
    frozen: bool = False
    resource_epoch: int = Field(default=0, ge=0)
    semantic_epoch: int = Field(default=0, ge=0)
    replan_trigger: str = ""
    reason: str = ""
    excluded_mobile_or_unavailable_sources: tuple[str, ...] = ()
