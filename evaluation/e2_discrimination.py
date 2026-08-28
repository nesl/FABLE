"""Fail-closed structural gates for the redesigned E2 experiment."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from fable.planning.deployment import DeploymentGraph, DeploymentGraphError

from evaluation.baselines.models import BaselineDecision, BaselinePlanningCase
from evaluation.concurrent_admission import decision_reservations


def _decision_signature(cell: dict[str, object], policy_id: str = "FABLE") -> dict[str, object] | None:
    """Return the structural part of a serialized decision from an E2 cell."""

    decisions = cell.get("joint_decisions", ())
    if not isinstance(decisions, (list, tuple)):
        return None
    for decision in decisions:
        if not isinstance(decision, dict) or decision.get("baseline_id") != policy_id:
            continue
        return {
            "feasible": bool(decision.get("selected_alternative_ids")),
            "selected_chain_ids": tuple(decision.get("selected_chain_ids") or ()),
            "selected_node_ids": tuple(decision.get("selected_node_ids") or ()),
            "selected_source_ids": tuple(decision.get("selected_source_ids") or ()),
        }
    return None


def evaluate_e2_network_discrimination(
    cells: Iterable[dict[str, object]],
    *,
    nominal_network_id: str = "N0",
) -> dict[str, object]:
    """Require a network condition to cause a structural FABLE decision change.

    A change in the estimated completion time alone is not evidence of network
    adaptation.  Cells are paired by contention condition and hypothesis count.
    The gate passes only if a constrained network changes feasibility, selected
    chains, placement nodes, or bound sources. Alternative IDs are deliberately
    excluded because the ID includes the network-priced realization.
    """

    materialized = tuple(cells)
    nominal: dict[tuple[object, object], dict[str, object]] = {}
    for cell in materialized:
        if cell.get("network_id") == nominal_network_id:
            nominal[(cell.get("condition_id"), cell.get("hypotheses"))] = cell

    comparisons: list[dict[str, object]] = []
    for cell in materialized:
        network_id = cell.get("network_id")
        if network_id == nominal_network_id:
            continue
        key = (cell.get("condition_id"), cell.get("hypotheses"))
        base = nominal.get(key)
        if base is None:
            continue
        before = _decision_signature(base)
        after = _decision_signature(cell)
        structural_change = before is not None and after is not None and before != after
        comparisons.append({
            "condition_id": key[0],
            "hypotheses": key[1],
            "constrained_network_id": network_id,
            "structural_change": structural_change,
            "nominal_signature": before,
            "constrained_signature": after,
        })

    discriminating = [item for item in comparisons if item["structural_change"]]
    return {
        "schema_version": "fable.e2_network_discrimination.v1",
        "valid": bool(discriminating),
        "gate": "network_changes_feasibility_chain_placement_or_source",
        "cost_only_changes_count_as_adaptation": False,
        "paired_comparisons": len(comparisons),
        "discriminating_comparisons": len(discriminating),
        "comparisons": comparisons,
    }


def evaluate_e2_discrimination(
    case: BaselinePlanningCase,
    *,
    deployment: DeploymentGraph,
    decisions: Iterable[BaselineDecision],
) -> dict[str, object]:
    """Return an inspectable gate report; never infer value from placement alone."""

    alternatives_by_demand = Counter(
        item.demand_id for item in case.frontier_graph.alternatives
    )
    alternatives = case.frontier_graph.alternatives
    pool_users: dict[str, set[object]] = {}
    for alternative in alternatives:
        for step in alternative.step_placements:
            pool_id, _ = deployment.resource_pool(step.node_id)
            pool_users.setdefault(pool_id, set()).add(alternative.demand_id)
    competing_pools = sorted(
        pool_id for pool_id, demands in pool_users.items() if len(demands) >= 2
    )

    by_policy = {item.baseline_id.value: item for item in decisions}
    b4 = by_policy.get("B4_GREEDY_FRONTIER")
    fable = by_policy.get("FABLE")
    oracle = by_policy.get("O1_EXHAUSTIVE_ORACLE")

    def _fits(decision: BaselineDecision | None) -> bool:
        if decision is None or not decision.selected_alternative_ids:
            return False
        try:
            deployment.with_resource_reservations(
                decision_reservations(deployment, case.frontier_graph, decision)
            )
        except DeploymentGraphError:
            return False
        return True

    def _cost(decision: BaselineDecision | None) -> tuple[int, int] | None:
        if decision is None or not decision.selected_alternative_ids:
            return None
        return (
            int(decision.predicted_completion_ms or 0),
            int(decision.predicted_transfer_bytes or 0),
        )

    b4_cost = _cost(b4)
    fable_cost = _cost(fable)
    oracle_cost = _cost(oracle)
    b4_strictly_worse = bool(
        b4_cost is not None
        and oracle_cost is not None
        and b4_cost[0] >= oracle_cost[0]
        and b4_cost[1] >= oracle_cost[1]
        and b4_cost != oracle_cost
    )
    exact_oracle_match = fable_cost == oracle_cost and oracle_cost is not None
    within_oracle_tolerance = bool(
        fable_cost is not None
        and oracle_cost is not None
        and fable_cost[0] <= max(oracle_cost[0] + 1, round(oracle_cost[0] * 1.01))
        and fable_cost[1] <= max(oracle_cost[1] + 1, round(oracle_cost[1] * 1.01))
    )
    gates = {
        "multiple_demands": len(case.frontier_demands) >= 2,
        "multiple_realizations_per_demand": bool(alternatives_by_demand)
        and all(count >= 2 for count in alternatives_by_demand.values()),
        "shared_bounded_resource_pool": bool(competing_pools),
        "b4_discriminated": bool(b4)
        and ((not _fits(b4)) or b4_strictly_worse),
        "fable_feasible": _fits(fable),
        "oracle_feasible": _fits(oracle),
        "fable_within_oracle_tolerance": within_oracle_tolerance,
    }
    return {
        "schema_version": "fable.e2_discrimination.v1",
        "valid": all(gates.values()),
        "gates": gates,
        "diagnostics": {
            "fable_matches_oracle_cost_exactly": exact_oracle_match,
            "oracle_tolerance_fraction": 0.01,
        },
        "competing_resource_pools": competing_pools,
        "alternatives_per_demand": {
            str(key): value for key, value in alternatives_by_demand.items()
        },
        "policy_costs": {
            policy: {
                "completion_ms": decision.predicted_completion_ms,
                "transfer_bytes": decision.predicted_transfer_bytes,
                "capacity_feasible": _fits(decision),
            }
            for policy, decision in by_policy.items()
        },
    }
