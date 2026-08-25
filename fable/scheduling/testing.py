"""Deterministic fake Phase-5 demands and plan candidates."""

from __future__ import annotations

from datetime import timedelta
from uuid import UUID

from fable.common.enums import (
    ExecutionInputKind,
    ExecutionMode,
    PlanStatus,
    ResultKind,
)
from fable.common.examples import BASE_TIME
from fable.common.ids import deterministic_id, uuid7
from fable.common.schemas import (
    DataMovementConstraints,
    ExecutionInput,
    ExecutionPlan,
    PhysicalPlanLabel,
    PlanCost,
    PlanStep,
    PredicateDemand,
    PredicateRole,
    ResourceReservation,
    SemanticPredicate,
)
from fable.common.time import DeadlineSpec, EventTimeInterval
from fable.planning.provider_registry import ProviderRegistry

from .models import PlanCandidate, TaskSchedulingPolicy


def fake_audio_demand(
    *,
    request_id: str = "audio_task",
    hypothesis_id: UUID | None = None,
    graph_node_id: str = "audio_branch",
    label: str = "gunshot",
    interval: EventTimeInterval | None = None,
    deadline_offset_ms: int = 10_000,
) -> PredicateDemand:
    interval = interval or EventTimeInterval(
        start=BASE_TIME,
        end=BASE_TIME + timedelta(seconds=1),
    )
    return PredicateDemand(
        request_id=request_id,
        graph_hash="sha256:" + "a" * 64,
        hypothesis_id=hypothesis_id or uuid7(),
        hypothesis_version=1,
        frontier_id=uuid7(),
        checkpoint_id=uuid7(),
        graph_node_id=graph_node_id,
        semantic_predicate=SemanticPredicate(
            predicate_id="AUDIO_EVENT",
            roles=(
                PredicateRole(
                    role_name="zone",
                    variable="store",
                    entity_type="zone",
                ),
            ),
            parameters={"label": label},
            result_kind=ResultKind.INSTANT_MATCH,
        ),
        bound_roles={"zone": "store"},
        event_time_interval=interval,
        deadline=DeadlineSpec(
            latest_useful_completion=BASE_TIME
            + timedelta(milliseconds=deadline_offset_ms)
        ),
        eligible_source_ids=("microphone_store",),
        acceptable_output_types=("predicate_match.v1",),
        hard_constraints=DataMovementConstraints(
            raw_data_must_remain_local=True,
            allowed_node_ids=("sensor_a",),
        ),
    )


def fake_audio_candidate(
    demand: PredicateDemand,
    *,
    provider_registry: ProviderRegistry,
    task_policy: TaskSchedulingPolicy | None = None,
    execution_mode: ExecutionMode = ExecutionMode.LIVE,
    input_kind: ExecutionInputKind = ExecutionInputKind.LIVE_SOURCE,
    artifact_id: UUID | None = None,
    expires_at=None,
    node_id: str = "sensor_a",
    node_class: str = "sensor",
    label: str | None = None,
    fallback_rank: int = 0,
) -> PlanCandidate:
    profile = provider_registry.profile("audio_event_classifier", node_class)
    label = label or str(demand.semantic_predicate.parameters.get("label", "gunshot"))
    alternative_id = deterministic_id(
        "alt",
        {
            "demand_id": demand.demand_id,
            "mode": execution_mode,
            "input_kind": input_kind,
            "artifact_id": artifact_id,
            "node": node_id,
            "label": label,
        },
        length=32,
    )
    external = ExecutionInput(
        name="audio",
        data_type="audio_segment.v1",
        kind=input_kind,
        node_id=node_id,
        source_id=("microphone_store" if input_kind == ExecutionInputKind.LIVE_SOURCE else None),
        artifact_id=artifact_id,
        bytes=0 if input_kind == ExecutionInputKind.LIVE_SOURCE else 512_000,
        expires_at=expires_at,
    )
    completion_ms = profile.startup_ms + profile.execution_ms
    step = PlanStep(
        step_id=f"{alternative_id}:classify",
        provider_id="audio_event_classifier",
        node_id=node_id,
        demand_id=demand.demand_id,
        alternative_id=alternative_id,
        chain_id="detect_audio_event",
        execution_mode=execution_mode,
        inputs=(external,),
        input_artifact_ids=(() if artifact_id is None else (artifact_id,)),
        input_data_types=("audio_segment.v1",),
        output_data_types=("audio_event_set.v1", "predicate_match.v1"),
        parameters=(("label", label),),
        cpu_cores=profile.cpu_cores,
        memory_mb=profile.memory_mb,
        gpu_memory_mb=profile.gpu_memory_mb,
        quality_score=profile.quality_score,
        estimated_startup_ms=profile.startup_ms,
        estimated_execution_ms=profile.execution_ms,
        estimated_transfer_bytes=external.bytes,
    )
    cost = PlanCost(
        predicted_completion_ms=completion_ms,
        deadline_slack_ms=int(
            (
                demand.deadline.latest_useful_completion - BASE_TIME
            ).total_seconds()
            * 1000
        )
        - completion_ms,
        startup_cost_ms=profile.startup_ms,
        resource_cost_units=(
            profile.cpu_cores
            + profile.memory_mb / 1024.0
            + 2.0 * profile.gpu_memory_mb / 1024.0
        ),
        transfer_bytes=external.bytes,
    )
    label_contract = PhysicalPlanLabel(
        checkpoint_id=demand.checkpoint_id,
        covered_demand_ids=(demand.demand_id,),
        steps=(step,),
        continuation_output_types=("audio_event_set.v1",),
        cost=cost,
        hard_constraints_satisfied=True,
        quality_floor_satisfied=True,
        feasibility_reasons=("synthetic Phase-5 fixture",),
    )
    plan = ExecutionPlan(
        label_id=label_contract.label_id or "",
        checkpoint_id=demand.checkpoint_id,
        demand_ids=(demand.demand_id,),
        steps=(step,),
        reservations=(
            ResourceReservation(
                node_id=node_id,
                cpu_cores=profile.cpu_cores,
                memory_mb=profile.memory_mb,
                gpu_memory_mb=profile.gpu_memory_mb,
                network_bytes=external.bytes,
            ),
        ),
        status=PlanStatus.CANDIDATE,
        expires_at=demand.deadline.latest_useful_completion,
    )
    policy = task_policy or TaskSchedulingPolicy(request_id=demand.request_id)
    return PlanCandidate(
        plan=plan,
        demands=(demand,),
        task_policy=policy,
        predicted_completion_ms=cost.predicted_completion_ms,
        startup_cost_ms=cost.startup_cost_ms,
        incremental_resource_cost_units=cost.resource_cost_units,
        transfer_bytes=cost.transfer_bytes,
        fallback_rank=fallback_rank,
        created_at=BASE_TIME,
    )
