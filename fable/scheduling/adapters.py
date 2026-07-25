"""Adapters between Phase-4 search results and Phase-5 admission candidates."""

from __future__ import annotations

from collections.abc import Iterable

from fable.common.schemas import PredicateDemand
from fable.planning.models import PhysicalAlternativeGraph
from fable.planning.search_models import LabelSearchState, PlanSearchResult

from .models import PlanCandidate, TaskSchedulingPolicy


class CandidateAdapterError(ValueError):
    pass


def candidate_from_search_result(
    result: PlanSearchResult,
    graph: PhysicalAlternativeGraph,
    demands: Iterable[PredicateDemand],
    *,
    task_policy: TaskSchedulingPolicy,
    fallback_index: int | None = None,
) -> PlanCandidate:
    demand_tuple = tuple(demands)
    if fallback_index is None:
        state = result.selected
        plan = result.execution_plan
        fallback_rank = 0
    else:
        try:
            state = result.fallbacks[fallback_index]
        except IndexError as exc:
            raise CandidateAdapterError(f"unknown fallback index {fallback_index}") from exc
        # Recreate the immutable execution-plan projection for the fallback.
        from fable.common.enums import PlanStatus
        from fable.common.schemas import ExecutionPlan, ResourceReservation

        network_bytes_by_node: dict[str, int] = {}
        for step in state.label.steps:
            network_bytes_by_node[step.node_id] = (
                network_bytes_by_node.get(step.node_id, 0)
                + step.estimated_transfer_bytes
            )
        plan = ExecutionPlan(
            label_id=state.label_id,
            checkpoint_id=state.label.checkpoint_id,
            demand_ids=state.label.covered_demand_ids,
            steps=state.label.steps,
            reservations=tuple(
                ResourceReservation(
                    node_id=item.node_id,
                    cpu_cores=item.cpu_cores,
                    memory_mb=item.memory_mb,
                    gpu_memory_mb=item.gpu_memory_mb,
                    network_bytes=network_bytes_by_node.get(item.node_id, 0),
                )
                for item in state.node_resources
            ),
            status=PlanStatus.CANDIDATE,
            expires_at=state.expires_at,
        )
        fallback_rank = fallback_index + 1
    if state is None or plan is None:
        raise CandidateAdapterError("search result has no feasible selected plan")
    alternatives_by_id = {item.alternative_id: item for item in graph.alternatives}
    try:
        alternatives = tuple(
            alternatives_by_id[item_id] for item_id in state.selected_alternative_ids
        )
    except KeyError as exc:
        raise CandidateAdapterError(
            f"search label references an alternative missing from the physical graph: {exc}"
        ) from exc
    return PlanCandidate(
        plan=plan,
        demands=demand_tuple,
        alternatives=alternatives,
        task_policy=task_policy,
        predicted_completion_ms=state.label.cost.predicted_completion_ms,
        startup_cost_ms=state.label.cost.startup_cost_ms,
        incremental_resource_cost_units=state.label.cost.resource_cost_units,
        transfer_bytes=state.label.cost.transfer_bytes,
        fallback_rank=fallback_rank,
    )
