"""Common evaluation records emitted at the live planning boundary."""

from __future__ import annotations

from time import perf_counter_ns
from typing import Iterable

from evaluation.baselines.models import BaselinePlanningCase
from evaluation.live_orchestration import LivePlanningResult
from evaluation.schemas import (
    BaselineId,
    PlanDecision,
    PredicateDemandRecord,
    ProviderCommand,
    ProviderLeaseEvent,
)
from evaluation.schemas.records import EvaluationRecord
from fable.common.ids import deterministic_id


def planning_records(
    *,
    case: BaselinePlanningCase,
    baseline_id: BaselineId,
    planning: LivePlanningResult,
) -> tuple[EvaluationRecord, ...]:
    """Normalize one live planning/dispatch boundary without private state."""

    decision = planning.decision
    common = {
        "run_id": case.run_id,
        "baseline_id": baseline_id,
        "trace_id": case.trace_id,
        "request_id": case.request_id,
        "event_time": case.now,
    }
    identity = {
        key: value for key, value in common.items() if key != "event_time"
    }
    records: list[EvaluationRecord] = [
        PredicateDemandRecord(
            **common,
            hypothesis_id=str(demand.hypothesis_id),
            monotonic_timestamp_ns=perf_counter_ns(),
            demand_id=str(demand.demand_id),
            checkpoint_id=str(demand.checkpoint_id),
            graph_version=case.graph_version,
            predicate_id=demand.semantic_predicate.predicate_id,
            semantic_epoch=case.semantic_epoch,
            resource_epoch=case.resource_epoch,
            bindings=demand.bound_roles,
            eligible_source_ids=demand.eligible_source_ids,
            deadline=demand.deadline.latest_useful_completion,
        )
        for demand in _decision_demands(case, decision.baseline_id)
    ]
    records.append(
        PlanDecision(
            **common,
            metadata={
                # Keep the bounded candidate set with the decision.  This is
                # small (the physical graph is already capped) and makes an
                # unexpected placement diagnosable without rerunning media or
                # relying on ephemeral orchestrator logs.
                "candidate_alternatives": [
                    {
                        "alternative_id": alternative.alternative_id,
                        "chain_id": alternative.chain_id,
                        "completion_ms": alternative.estimated_completion_ms,
                        "transfer_bytes": alternative.estimated_transfer_bytes,
                        "placements": [
                            {
                                "provider_id": step.provider_id,
                                "node_id": step.node_id,
                                "startup_ms": step.startup_ms,
                                "execution_ms": step.execution_ms,
                            }
                            for step in alternative.step_placements
                        ],
                    }
                    for alternative in case.frontier_graph.alternatives[:32]
                ],
                "graph_pruned": [
                    item.model_dump(mode="json")
                    for item in case.frontier_graph.pruned[:32]
                ],
            },
            monotonic_timestamp_ns=perf_counter_ns(),
            decision_id=deterministic_id(
                "live_plan_decision",
                {
                    "request": case.request_id,
                    "baseline": baseline_id,
                    "checkpoint": decision.checkpoint_id,
                    "resource_epoch": case.resource_epoch,
                    "semantic_epoch": case.semantic_epoch,
                },
                length=32,
            ),
            graph_version=case.graph_version,
            checkpoint_id=str(decision.checkpoint_id),
            planning_scope=decision.planning_scope,
            selected_alternative_ids=decision.selected_alternative_ids,
            selected_chain_ids=decision.selected_chain_ids,
            selected_node_ids=decision.selected_node_ids,
            selected_source_ids=decision.selected_source_ids,
            activated_provider_keys=decision.activated_provider_keys,
            continuation_types=decision.continuation_types,
            predicted_completion_ms=decision.predicted_completion_ms,
            predicted_transfer_bytes=decision.predicted_transfer_bytes,
            predicted_compute_ms=decision.predicted_compute_ms,
            predicted_slack_ms=decision.predicted_slack_ms,
            planning_latency_ms=decision.planning_latency_ms,
            labels_generated=decision.labels_generated,
            labels_pruned=decision.labels_pruned,
            labels_retained=decision.labels_retained,
            pruning_counts=decision.pruning_counts,
            pruning_samples=decision.pruning_samples,
            oracle_gap_ms=decision.oracle_gap_ms,
            frozen=decision.frozen,
            resource_epoch=decision.resource_epoch,
            semantic_epoch=decision.semantic_epoch,
            replan_trigger=decision.replan_trigger or case.replan_trigger,
            reason=decision.reason,
        )
    )
    for command in planning.commands:
        # Lightweight policy/bridge tests may use opaque command sentinels.
        # Only typed distributed commands cross the runtime logging boundary.
        if not hasattr(command, "message_id"):
            continue
        lease = getattr(command, "lease", None)
        demand = getattr(command, "demand", None)
        records.append(
            ProviderCommand(
                **identity,
                hypothesis_id=(
                    str(demand.hypothesis_id) if demand is not None else None
                ),
                provider_id=getattr(
                    getattr(command, "runtime", None), "provider_id", None
                ),
                event_time=getattr(command, "issued_at", case.now),
                monotonic_timestamp_ns=perf_counter_ns(),
                command_id=str(command.message_id),
                command="ACTIVATE",
                provider_instance_id=command.provider_instance_id,
                demand_ids=(
                    (str(demand.demand_id),) if demand is not None else ()
                ),
                node_id=command.node_id,
                emitted_at=getattr(command, "issued_at", case.now),
            )
        )
        if lease is not None:
            records.append(
            ProviderLeaseEvent(
                    **identity,
                    hypothesis_id=(
                        str(demand.hypothesis_id)
                        if demand is not None
                        else None
                    ),
                    provider_id=getattr(
                        getattr(command, "runtime", None),
                        "provider_id",
                        None,
                    ),
                    event_time=lease.starts_at,
                    monotonic_timestamp_ns=perf_counter_ns(),
                    lease_id=str(lease.lease_id),
                    provider_instance_id=lease.provider_instance_id,
                    demand_id=str(lease.demand_id),
                    lease_event=lease.status.value,
                    attached_at=lease.starts_at,
                    detached_at=None,
                )
            )
    return tuple(records)


def _decision_demands(
    case: BaselinePlanningCase,
    baseline_id: BaselineId,
) -> Iterable:
    if baseline_id in {
        BaselineId.B0_PRODUCE_ALL,
        BaselineId.B1_STATIC_WHOLE_EVENT,
        BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
    }:
        return case.all_task_demands
    return case.frontier_demands
