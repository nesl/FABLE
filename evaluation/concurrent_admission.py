"""Bounded cross-hypothesis admission and resource reservation for E2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from fable.common.ids import deterministic_id, uuid7
from fable.planning.deployment import DeploymentGraph, DeploymentGraphError
from fable.planning.models import ComputeCapacity, PhysicalAlternativeGraph

from evaluation.baselines.models import BaselineDecision, BaselinePlanningCase


ReservationMap = dict[str, ComputeCapacity]
PolicyFactory = Callable[[DeploymentGraph], object]


@dataclass(frozen=True)
class AdmissionResult:
    request_id: str
    decision: BaselineDecision
    admitted: bool
    rejection_reason: str = ""
    committed_reservations: tuple[tuple[str, ComputeCapacity], ...] = ()


def merge_reservations(*parts: ReservationMap) -> ReservationMap:
    merged: dict[str, tuple[float, int, int]] = {}
    for part in parts:
        for pool_id, value in part.items():
            cpu, memory, gpu = merged.get(pool_id, (0.0, 0, 0))
            merged[pool_id] = (
                cpu + value.cpu_cores,
                memory + value.memory_mb,
                gpu + value.gpu_memory_mb,
            )
    return {
        pool_id: ComputeCapacity(
            cpu_cores=value[0], memory_mb=value[1], gpu_memory_mb=value[2]
        )
        for pool_id, value in merged.items()
    }


def fractional_reservation(
    deployment: DeploymentGraph, *, node_id: str, fraction: float
) -> ReservationMap:
    if not 0 <= fraction < 1:
        raise ValueError("reservation fraction must be in [0, 1)")
    pool_id, capacity = deployment.resource_pool(node_id)
    return {
        pool_id: ComputeCapacity(
            cpu_cores=capacity.cpu_cores * fraction,
            memory_mb=round(capacity.memory_mb * fraction),
            gpu_memory_mb=round(capacity.gpu_memory_mb * fraction),
        )
    }


def decision_reservations(
    deployment: DeploymentGraph,
    graph: PhysicalAlternativeGraph,
    decision: BaselineDecision,
) -> ReservationMap:
    alternatives = {item.alternative_id: item for item in graph.alternatives}
    usage: dict[str, tuple[float, int, int]] = {}
    share_instances = decision.baseline_id.value in {
        "FABLE",
        "O1_EXHAUSTIVE_ORACLE",
    }
    charged_provider_keys: set[tuple[str, str]] = set()
    for alternative_id in decision.selected_alternative_ids:
        alternative = alternatives.get(alternative_id)
        if alternative is None:
            continue
        for step in alternative.step_placements:
            if step.reused_provider_instance_id is not None:
                continue
            provider_key = (step.provider_id, step.node_id)
            if share_instances and provider_key in charged_provider_keys:
                continue
            charged_provider_keys.add(provider_key)
            pool_id, _ = deployment.resource_pool(step.node_id)
            cpu, memory, gpu = usage.get(pool_id, (0.0, 0, 0))
            usage[pool_id] = (
                cpu + step.cpu_cores,
                memory + step.memory_mb,
                gpu + step.gpu_memory_mb,
            )
    return {
        pool_id: ComputeCapacity(
            cpu_cores=value[0], memory_mb=value[1], gpu_memory_mb=value[2]
        )
        for pool_id, value in usage.items()
    }


def sequential_committed_admission(
    cases: Iterable[BaselinePlanningCase],
    *,
    deployment: DeploymentGraph,
    policy_factory: PolicyFactory,
    initial_reservations: ReservationMap | None = None,
) -> tuple[AdmissionResult, ...]:
    """Plan in declared order and commit each accepted resource footprint."""

    committed = dict(initial_reservations or {})
    results = []
    for case in cases:
        snapshot = deployment.with_resource_reservations(committed)
        policy = policy_factory(snapshot)
        decision = policy.plan(case)
        proposed = decision_reservations(snapshot, case.frontier_graph, decision)
        admitted = bool(decision.selected_alternative_ids)
        reason = ""
        if admitted:
            try:
                deployment.with_resource_reservations(
                    merge_reservations(committed, proposed)
                )
            except DeploymentGraphError as exc:
                admitted = False
                reason = str(exc)
        else:
            reason = "policy returned no feasible realization"
        if admitted:
            committed = merge_reservations(committed, proposed)
        results.append(
            AdmissionResult(
                request_id=case.request_id,
                decision=decision,
                admitted=admitted,
                rejection_reason=reason,
                committed_reservations=tuple(sorted(committed.items())),
            )
        )
    return tuple(results)


def joint_batch_case(
    cases: Iterable[BaselinePlanningCase],
    *,
    run_id: str,
    request_id: str,
) -> BaselinePlanningCase:
    """Merge immutable frontier snapshots into one synthetic admission batch."""

    members = tuple(cases)
    if not members:
        raise ValueError("joint batch requires at least one planning case")
    checkpoint_id = uuid7()
    demands = tuple(
        demand.model_copy(update={"checkpoint_id": checkpoint_id})
        for case in members
        for demand in case.frontier_demands
    )
    alternatives = tuple(
        alternative.model_copy(update={"checkpoint_id": checkpoint_id})
        for case in members
        for alternative in case.frontier_graph.alternatives
    )
    nodes = {
        item.node_id: item
        for case in members
        for item in case.frontier_graph.nodes
    }
    edges = {
        item.edge_id: item
        for case in members
        for item in case.frontier_graph.edges
    }
    pruned = tuple(
        item for case in members for item in case.frontier_graph.pruned
    )
    graph = PhysicalAlternativeGraph(
        graph_id=deterministic_id(
            "e2_batch_graph",
            {
                "request_id": request_id,
                "demand_ids": sorted(str(item.demand_id) for item in demands),
                "alternatives": sorted(item.alternative_id for item in alternatives),
            },
            length=32,
        ),
        checkpoint_ids=(checkpoint_id,),
        demand_ids=tuple(item.demand_id for item in demands),
        nodes=tuple(sorted(nodes.values(), key=lambda item: item.node_id)),
        edges=tuple(sorted(edges.values(), key=lambda item: item.edge_id)),
        alternatives=alternatives,
        pruned=pruned,
        built_at=max(case.frontier_graph.built_at for case in members),
    )
    first = members[0]
    return replace(
        first,
        run_id=run_id,
        request_id=request_id,
        event_family="concurrent_e2_batch",
        frontier_demands=demands,
        all_task_demands=demands,
        frontier_graph=graph,
        whole_event_graph=graph,
        replay_supported_sensor_ids=tuple(
            sorted(
                {
                    source
                    for case in members
                    for source in case.replay_supported_sensor_ids
                }
            )
        ),
    )
