from __future__ import annotations

from datetime import timedelta

from fable.common.enums import ExecutionMode
from fable.common.examples import BASE_TIME
from fable.common.ids import uuid7
from fable.common.time import EventTimeInterval
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import ComputeCapacity, DeploymentNode, ExternalInputKind
from fable.planning.testing import fake_provider_registry
from fable.scheduling import (
    CapacityLedger,
    MultiTenantOrderer,
    MultiTenantScheduler,
    ProviderLifecycleManager,
    TaskPriorityClass,
    TaskSchedulingPolicy,
)
from fable.scheduling.models import AdmissionDecision
from fable.scheduling.testing import fake_audio_candidate, fake_audio_demand


def _tiny_deployment() -> DeploymentGraph:
    return DeploymentGraph(
        nodes=(
            DeploymentNode(
                node_id="sensor_a",
                node_class="sensor",
                region="store",
                capacity=ComputeCapacity(
                    cpu_cores=0.6,
                    memory_mb=384,
                    gpu_memory_mb=512,
                ),
                capabilities=("audio", "gpu"),
            ),
        )
    )


def test_round_robin_fairness_between_tasks() -> None:
    registry = fake_provider_registry()
    policy_a = TaskSchedulingPolicy(
        request_id="task_a",
        priority_class=TaskPriorityClass.NORMAL,
    )
    policy_b = TaskSchedulingPolicy(
        request_id="task_b",
        priority_class=TaskPriorityClass.NORMAL,
    )
    candidates = []
    for request_id, policy in (("task_a", policy_a), ("task_b", policy_b)):
        for index in range(2):
            demand = fake_audio_demand(
                request_id=request_id,
                hypothesis_id=uuid7(),
                graph_node_id=f"branch_{index}",
                interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=index),
                    end=BASE_TIME + timedelta(seconds=index + 1),
                ),
            )
            candidates.append(
                fake_audio_candidate(
                    demand,
                    provider_registry=registry,
                    task_policy=policy,
                )
            )

    ordered = MultiTenantOrderer().order(candidates, now=BASE_TIME)
    assert [item.request_id for item in ordered] == [
        "task_a",
        "task_b",
        "task_a",
        "task_b",
    ]


def test_live_only_work_is_admitted_before_retained_work_under_pressure() -> None:
    registry = fake_provider_registry()
    deployment = _tiny_deployment()
    lifecycle = ProviderLifecycleManager(
        provider_registry=registry,
        capacity=CapacityLedger(deployment),
    )
    scheduler = MultiTenantScheduler(lifecycle=lifecycle)
    policy = TaskSchedulingPolicy(request_id="one_task")

    live_demand = fake_audio_demand(
        request_id="one_task",
        hypothesis_id=uuid7(),
        graph_node_id="live_branch",
        interval=EventTimeInterval(
            start=BASE_TIME,
            end=BASE_TIME + timedelta(seconds=1),
        ),
    )
    retained_demand = fake_audio_demand(
        request_id="one_task",
        hypothesis_id=uuid7(),
        graph_node_id="retained_branch",
        interval=EventTimeInterval(
            start=BASE_TIME - timedelta(seconds=5),
            end=BASE_TIME - timedelta(seconds=4),
        ),
    )
    live = fake_audio_candidate(
        live_demand,
        provider_registry=registry,
        task_policy=policy,
        execution_mode=ExecutionMode.LIVE,
        input_kind=ExternalInputKind.LIVE_SOURCE,
    )
    retained = fake_audio_candidate(
        retained_demand,
        provider_registry=registry,
        task_policy=policy,
        execution_mode=ExecutionMode.RETROSPECTIVE,
        input_kind=ExternalInputKind.RETAINED_ARTIFACT,
        artifact_id=uuid7(),
        expires_at=BASE_TIME + timedelta(minutes=1),
    )

    result = scheduler.admit((retained, live), now=BASE_TIME)
    assert result.ordered_candidate_ids[0] == live.candidate_id
    assert result.record_for(live.candidate_id or "").decision == AdmissionDecision.ADMITTED
    assert result.record_for(retained.candidate_id or "").decision == AdmissionDecision.DEFERRED
    assert result.resource_pressure is True
