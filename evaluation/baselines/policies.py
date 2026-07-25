"""B0-B4, FABLE, and exhaustive-oracle policies over one common planning case."""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import product
from math import prod
from time import perf_counter_ns
from typing import Protocol

from fable.common.ids import uuid7
from fable.common.schemas import PredicateDemand
from fable.planning.beam_search import BoundedLabelPlanner
from fable.planning.models import PhysicalAlternative, PhysicalAlternativeGraph
from fable.planning.search_models import LabelSearchState, PlanSearchResult

from evaluation.schemas import BaselineId

from .models import BaselineDecision, BaselinePlanningCase
from .static_registry import StaticPipelineRegistry


class BaselinePolicy(Protocol):
    baseline_id: BaselineId

    def plan(self, case: BaselinePlanningCase) -> BaselineDecision: ...


class AlwaysOnPolicy:
    baseline_id = BaselineId.B0_ALWAYS_ON

    def plan(self, case: BaselinePlanningCase) -> BaselineDecision:
        alternatives, excluded = _supported_alternatives(
            case.whole_event_graph.alternatives,
            case.replay_supported_sensor_ids,
        )
        return _decision_from_alternatives(
            baseline_id=self.baseline_id,
            case=case,
            alternatives=alternatives,
            planning_scope="WHOLE_EVENT_PROVIDER_UNION",
            reason=(
                "Activate the union of every feasible provider/source realization for the "
                "complete task and keep it active for the trace."
            ),
            excluded=excluded,
        )


class HandwrittenStaticPolicy:
    baseline_id = BaselineId.B1_HANDWRITTEN_STATIC

    def __init__(self, registry: StaticPipelineRegistry) -> None:
        self.registry = registry

    def plan(self, case: BaselinePlanningCase) -> BaselineDecision:
        spec = self.registry.get(case.event_family)
        alternatives, excluded = _supported_alternatives(
            case.whole_event_graph.alternatives,
            case.replay_supported_sensor_ids,
        )
        if spec is None:
            selected = _one_per_demand(alternatives)
            reason = "No handwritten pipeline was registered; used a deterministic first feasible plan per demand."
        else:
            preferred = set(spec.preferred_chain_ids)
            matching = tuple(item for item in alternatives if item.chain_id in preferred)
            selected = _one_per_demand(matching or alternatives)
            reason = (
                f"Fixed developer pipeline with preferred chains {sorted(preferred)}; "
                "placement and representation remain unchanged for the request."
            )
        return _decision_from_alternatives(
            baseline_id=self.baseline_id,
            case=case,
            alternatives=selected,
            planning_scope="HANDWRITTEN_STATIC_WHOLE_EVENT",
            reason=reason,
            frozen=True,
            excluded=excluded,
        )


class StaticWholeEventPolicy:
    baseline_id = BaselineId.B2_STATIC_WHOLE_EVENT

    def __init__(self, planner: BoundedLabelPlanner) -> None:
        self.planner = planner
        self._cache: dict[str, BaselineDecision] = {}

    def plan(self, case: BaselinePlanningCase) -> BaselineDecision:
        cached = self._cache.get(case.request_id)
        if cached is not None:
            return cached.model_copy(
                update={
                    "resource_epoch": case.resource_epoch,
                    "semantic_epoch": case.semantic_epoch,
                    "reason": cached.reason + " Reused the admission-time frozen plan.",
                }
            )
        graph, demands = _whole_event_checkpoint(case)
        started = perf_counter_ns()
        result = self.planner.search(graph, demands, now=case.now)
        elapsed_ms = (perf_counter_ns() - started) / 1_000_000
        decision = _decision_from_search(
            baseline_id=self.baseline_id,
            case=case,
            graph=graph,
            result=result,
            planning_scope="STATIC_WHOLE_EVENT",
            reason="Optimized the complete task once at admission and froze provider, placement, source, representation, and continuation choices.",
            frozen=True,
        )
        decision = decision.model_copy(update={"planning_latency_ms": elapsed_ms})
        self._cache[case.request_id] = decision
        return decision


class TaskResourceAdaptivePolicy:
    baseline_id = BaselineId.B3_TASK_RESOURCE_ADAPTIVE

    def __init__(self, planner: BoundedLabelPlanner) -> None:
        self.planner = planner
        self._cache: dict[tuple[str, int], BaselineDecision] = {}
        self._latest: dict[str, BaselineDecision] = {}

    def plan(self, case: BaselinePlanningCase) -> BaselineDecision:
        key = (case.request_id, case.resource_epoch)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.model_copy(update={"semantic_epoch": case.semantic_epoch})
        graph, demands = _whole_event_checkpoint(case, neutralize_spatial=True)
        result = self.planner.search(graph, demands, now=case.now)
        decision = _decision_from_search(
            baseline_id=self.baseline_id,
            case=case,
            graph=graph,
            result=result,
            planning_scope="TASK_LEVEL_RESOURCE_ADAPTIVE",
            reason=(
                "Replanned the complete task-level work set for a new resource epoch, "
                "without using current hypothesis progress, branch resolution, or likely next sensors."
            ),
        )
        self._cache[key] = decision
        self._latest[case.request_id] = decision
        return decision


class GreedyFrontierPolicy:
    baseline_id = BaselineId.B4_GREEDY_FRONTIER

    def plan(self, case: BaselinePlanningCase) -> BaselineDecision:
        supported, excluded = _supported_alternatives(
            case.frontier_graph.alternatives,
            case.replay_supported_sensor_ids,
        )
        grouped: dict[object, list[PhysicalAlternative]] = defaultdict(list)
        for item in supported:
            grouped[item.demand_id].append(item)
        chosen = []
        for demand in case.frontier_demands:
            candidates = grouped.get(demand.demand_id, [])
            if not candidates:
                continue
            chosen.append(
                min(
                    candidates,
                    key=lambda item: (
                        item.estimated_completion_ms,
                        item.estimated_transfer_bytes,
                        item.minimum_quality_score * -1,
                        item.alternative_id,
                    ),
                )
            )
        return _decision_from_alternatives(
            baseline_id=self.baseline_id,
            case=case,
            alternatives=tuple(chosen),
            planning_scope="GREEDY_FRONTIER_INDEPENDENT",
            reason=(
                "Used the grounded frontier but selected each demand independently by "
                "immediate completion and transfer cost; joint capacity and continuation value were ignored."
            ),
            excluded=excluded,
        )


class FablePolicy:
    baseline_id = BaselineId.FABLE

    def __init__(self, planner: BoundedLabelPlanner) -> None:
        self.planner = planner

    def plan(self, case: BaselinePlanningCase) -> BaselineDecision:
        result = self.planner.search(
            case.frontier_graph,
            case.frontier_demands,
            now=case.now,
        )
        return _decision_from_search(
            baseline_id=self.baseline_id,
            case=case,
            graph=case.frontier_graph,
            result=result,
            planning_scope="GROUNDED_FRONTIER_TO_NEXT_CHECKPOINT",
            reason=(
                "Constructed work from the current grounded hypothesis frontier and jointly "
                "selected provider, representation, placement, source, transfer, and continuation choices."
            ),
        )


class ExhaustiveOraclePolicy:
    baseline_id = BaselineId.O1_EXHAUSTIVE_ORACLE

    def __init__(self, planner: BoundedLabelPlanner, *, maximum_combinations: int = 100_000) -> None:
        self.planner = planner
        self.maximum_combinations = maximum_combinations

    def plan(self, case: BaselinePlanningCase) -> BaselineDecision:
        graph = case.frontier_graph
        demands = case.frontier_demands
        demand_map = {item.demand_id: item for item in demands}
        order = tuple(sorted(demand_map, key=str))
        grouped: dict[object, tuple[PhysicalAlternative, ...]] = {
            demand_id: tuple(
                sorted(
                    (item for item in graph.alternatives if item.demand_id == demand_id),
                    key=lambda item: item.alternative_id,
                )
            )
            for demand_id in order
        }
        combinations = prod(len(grouped[item]) for item in order) if order else 0
        if combinations > self.maximum_combinations:
            return BaselineDecision(
                baseline_id=self.baseline_id,
                request_id=case.request_id,
                checkpoint_id=demands[0].checkpoint_id,
                planning_scope="EXHAUSTIVE_ORACLE",
                resource_epoch=case.resource_epoch,
                semantic_epoch=case.semantic_epoch,
                reason=f"Skipped {combinations} combinations above the oracle limit {self.maximum_combinations}.",
            )
        feasible: list[LabelSearchState] = []
        if order and all(grouped[item] for item in order):
            for alternatives in product(*(grouped[item] for item in order)):
                state: LabelSearchState | None = None
                valid = True
                for demand_id, alternative in zip(order, alternatives, strict=True):
                    demand = demand_map[demand_id]
                    if self.planner.check_extension(
                        state,
                        alternative,
                        demand,
                        demand_map=demand_map,
                        now=case.now,
                    ):
                        valid = False
                        break
                    state = self.planner.extend_label(
                        state,
                        alternative,
                        demand,
                        demand_map=demand_map,
                        now=case.now,
                    )
                if valid and state is not None:
                    feasible.append(state)
        selected = min(feasible, key=self.planner.rank_key) if feasible else None
        if selected is None:
            return BaselineDecision(
                baseline_id=self.baseline_id,
                request_id=case.request_id,
                checkpoint_id=demands[0].checkpoint_id,
                planning_scope="EXHAUSTIVE_ORACLE",
                resource_epoch=case.resource_epoch,
                semantic_epoch=case.semantic_epoch,
                reason=f"Exhaustive enumeration considered {combinations} combinations and found no feasible plan.",
            )
        return _decision_from_state(
            baseline_id=self.baseline_id,
            case=case,
            graph=graph,
            state=selected,
            planning_scope="EXHAUSTIVE_ORACLE",
            reason=f"Selected the best feasible label after exhaustive enumeration of {combinations} combinations.",
        )


def _whole_event_checkpoint(
    case: BaselinePlanningCase,
    *,
    neutralize_spatial: bool = False,
) -> tuple[PhysicalAlternativeGraph, tuple[PredicateDemand, ...]]:
    checkpoint_id = uuid7()
    demands = tuple(
        item.model_copy(
            update={
                "checkpoint_id": checkpoint_id,
                **(
                    {"source_preferences": (), "spatial_prediction_id": None}
                    if neutralize_spatial
                    else {}
                ),
            }
        )
        for item in case.all_task_demands
    )
    alternatives = tuple(
        item.model_copy(
            update={
                "checkpoint_id": checkpoint_id,
                **(
                    {"spatial_preference_penalty": 0, "spatial_preference_reason": ""}
                    if neutralize_spatial
                    else {}
                ),
            }
        )
        for item in case.whole_event_graph.alternatives
    )
    graph = case.whole_event_graph.model_copy(
        update={"checkpoint_ids": (checkpoint_id,), "alternatives": alternatives}
    )
    return graph, demands


def _decision_from_search(
    *,
    baseline_id: BaselineId,
    case: BaselinePlanningCase,
    graph: PhysicalAlternativeGraph,
    result: PlanSearchResult,
    planning_scope: str,
    reason: str,
    frozen: bool = False,
) -> BaselineDecision:
    if result.selected is None:
        return BaselineDecision(
            baseline_id=baseline_id,
            request_id=case.request_id,
            checkpoint_id=result.trace.checkpoint_id,
            planning_scope=planning_scope,
            labels_generated=sum(len(item.generated_label_ids) for item in result.trace.boundaries),
            labels_pruned=sum(len(item.pruning_records) for item in result.trace.boundaries),
            labels_retained=sum(len(item.retained_label_ids) for item in result.trace.boundaries),
            oracle_gap_ms=result.trace.oracle.completion_gap_ms,
            frozen=frozen,
            resource_epoch=case.resource_epoch,
            semantic_epoch=case.semantic_epoch,
            reason=reason + " No feasible complete plan was found.",
        )
    return _decision_from_state(
        baseline_id=baseline_id,
        case=case,
        graph=graph,
        state=result.selected,
        planning_scope=planning_scope,
        reason=reason,
        frozen=frozen,
        labels_generated=sum(len(item.generated_label_ids) for item in result.trace.boundaries),
        labels_pruned=sum(len(item.pruning_records) for item in result.trace.boundaries),
        labels_retained=sum(len(item.retained_label_ids) for item in result.trace.boundaries),
        oracle_gap_ms=result.trace.oracle.completion_gap_ms,
    )


def _decision_from_state(
    *,
    baseline_id: BaselineId,
    case: BaselinePlanningCase,
    graph: PhysicalAlternativeGraph,
    state: LabelSearchState,
    planning_scope: str,
    reason: str,
    frozen: bool = False,
    labels_generated: int = 0,
    labels_pruned: int = 0,
    labels_retained: int = 0,
    oracle_gap_ms: int | None = None,
    planning_latency_ms: float = 0.0,
) -> BaselineDecision:
    alternatives = _lookup_alternatives(graph, state.selected_alternative_ids)
    return _decision_from_alternatives(
        baseline_id=baseline_id,
        case=case,
        alternatives=alternatives,
        planning_scope=planning_scope,
        reason=reason,
        frozen=frozen,
        planning_latency_ms=planning_latency_ms,
        labels_generated=labels_generated,
        labels_pruned=labels_pruned,
        labels_retained=labels_retained,
        oracle_gap_ms=oracle_gap_ms,
        predicted_completion_ms=state.label.cost.predicted_completion_ms,
        predicted_transfer_bytes=state.label.cost.transfer_bytes,
    )


def _decision_from_alternatives(
    *,
    baseline_id: BaselineId,
    case: BaselinePlanningCase,
    alternatives: tuple[PhysicalAlternative, ...],
    planning_scope: str,
    reason: str,
    frozen: bool = False,
    excluded: tuple[str, ...] = (),
    labels_generated: int = 0,
    labels_pruned: int = 0,
    labels_retained: int = 0,
    oracle_gap_ms: int | None = None,
    predicted_completion_ms: int | None = None,
    predicted_transfer_bytes: int | None = None,
    planning_latency_ms: float = 0.0,
) -> BaselineDecision:
    nodes = sorted({step.node_id for item in alternatives for step in item.step_placements})
    sources = sorted(
        {
            value.source_id
            for item in alternatives
            for value in item.external_inputs
            if value.source_id is not None
        }
    )
    providers = sorted(
        {
            f"{step.provider_id}@{step.node_id}"
            for item in alternatives
            for step in item.step_placements
        }
    )
    continuations = sorted(
        {value for item in alternatives for value in item.continuation_output_types}
    )
    checkpoint_id = (
        alternatives[0].checkpoint_id
        if alternatives
        else case.frontier_demands[0].checkpoint_id
    )
    return BaselineDecision(
        baseline_id=baseline_id,
        request_id=case.request_id,
        checkpoint_id=checkpoint_id,
        planning_scope=planning_scope,
        selected_alternative_ids=tuple(item.alternative_id for item in alternatives),
        selected_chain_ids=tuple(sorted({item.chain_id for item in alternatives})),
        selected_node_ids=tuple(nodes),
        selected_source_ids=tuple(sources),
        activated_provider_keys=tuple(providers),
        continuation_types=tuple(continuations),
        predicted_completion_ms=(
            predicted_completion_ms
            if predicted_completion_ms is not None
            else (max((item.estimated_completion_ms for item in alternatives), default=None))
        ),
        predicted_transfer_bytes=(
            predicted_transfer_bytes
            if predicted_transfer_bytes is not None
            else sum(item.estimated_transfer_bytes for item in alternatives)
        ),
        labels_generated=labels_generated,
        labels_pruned=labels_pruned,
        labels_retained=labels_retained,
        oracle_gap_ms=oracle_gap_ms,
        frozen=frozen,
        resource_epoch=case.resource_epoch,
        semantic_epoch=case.semantic_epoch,
        reason=reason,
        excluded_mobile_or_unavailable_sources=excluded,
    )


def _lookup_alternatives(
    graph: PhysicalAlternativeGraph,
    alternative_ids: tuple[str, ...],
) -> tuple[PhysicalAlternative, ...]:
    by_id = {item.alternative_id: item for item in graph.alternatives}
    return tuple(by_id[item] for item in alternative_ids if item in by_id)


def _one_per_demand(
    alternatives: tuple[PhysicalAlternative, ...],
) -> tuple[PhysicalAlternative, ...]:
    selected: dict[object, PhysicalAlternative] = {}
    for item in sorted(
        alternatives,
        key=lambda value: (
            value.demand_id,
            value.estimated_completion_ms,
            value.estimated_transfer_bytes,
            value.alternative_id,
        ),
    ):
        selected.setdefault(item.demand_id, item)
    return tuple(selected[key] for key in sorted(selected, key=str))


def _supported_alternatives(
    alternatives: tuple[PhysicalAlternative, ...],
    replay_supported_sensor_ids: tuple[str, ...],
) -> tuple[tuple[PhysicalAlternative, ...], tuple[str, ...]]:
    if not replay_supported_sensor_ids:
        return alternatives, ()
    allowed = {_sensor_token(item) for item in replay_supported_sensor_ids}
    kept: list[PhysicalAlternative] = []
    excluded: set[str] = set()
    for item in alternatives:
        source_ids = tuple(
            value.source_id
            for value in item.external_inputs
            if value.source_id is not None
        )
        unsupported = [
            source_id
            for source_id in source_ids
            if _sensor_token(source_id) is not None
            and _sensor_token(source_id) not in allowed
        ]
        if unsupported:
            excluded.update(unsupported)
            continue
        kept.append(item)
    return tuple(kept), tuple(sorted(excluded))


def _sensor_token(value: str) -> str | None:
    match = re.search(r"orin[_-]?(\d+)", value.lower())
    if match:
        return f"orin_{int(match.group(1))}"
    match = re.fullmatch(r"[nd](\d+)", value.lower())
    if match:
        return value.lower()
    return None
