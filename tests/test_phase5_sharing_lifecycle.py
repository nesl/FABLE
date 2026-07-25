from __future__ import annotations

from datetime import timedelta

from fable.common.enums import CancellationScope
from fable.common.examples import BASE_TIME
from fable.common.ids import uuid7
from fable.planning.testing import fake_deployment, fake_provider_registry
from fable.scheduling import (
    CancellationManager,
    CancellationRequest,
    CapacityLedger,
    MultiTenantScheduler,
    ProviderInstanceLifecycle,
    ProviderLifecycleManager,
    TaskSchedulingPolicy,
)
from fable.scheduling.testing import fake_audio_candidate, fake_audio_demand


def test_two_hypotheses_share_one_provider_and_keep_separate_leases() -> None:
    registry = fake_provider_registry()
    lifecycle = ProviderLifecycleManager(
        provider_registry=registry,
        capacity=CapacityLedger(fake_deployment()),
        idle_grace_ms=100,
    )
    scheduler = MultiTenantScheduler(lifecycle=lifecycle)

    hypothesis_a = uuid7()
    hypothesis_b = uuid7()
    demand_a = fake_audio_demand(
        request_id="task_a",
        hypothesis_id=hypothesis_a,
        graph_node_id="gunshot_a",
    )
    demand_b = fake_audio_demand(
        request_id="task_b",
        hypothesis_id=hypothesis_b,
        graph_node_id="gunshot_b",
    )
    candidate_a = fake_audio_candidate(
        demand_a,
        provider_registry=registry,
        task_policy=TaskSchedulingPolicy(request_id="task_a"),
    )
    candidate_b = fake_audio_candidate(
        demand_b,
        provider_registry=registry,
        task_policy=TaskSchedulingPolicy(request_id="task_b"),
    )

    batch = scheduler.admit((candidate_a, candidate_b), now=BASE_TIME)
    assert len(batch.admitted_plan_ids) == 2
    instances = [
        item
        for item in lifecycle.active_instances
        if item.share_key.provider_id == "audio_event_classifier"
    ]
    assert len(instances) == 1
    instance = instances[0]
    assert len(instance.active_lease_ids) == 2
    assert {
        lease.hypothesis_id for lease in lifecycle.leases_for_instance(instance.provider_instance_id)
    } == {hypothesis_a, hypothesis_b}
    assert instance.lifecycle_history == (
        ProviderInstanceLifecycle.COLD,
        ProviderInstanceLifecycle.WARMING,
    )

    lifecycle.mark_ready(instance.provider_instance_id, now=BASE_TIME + timedelta(milliseconds=10))
    cancellation = CancellationManager(lifecycle)
    first = cancellation.cancel(
        CancellationRequest(
            scope=CancellationScope.HYPOTHESIS,
            request_id="task_a",
            hypothesis_id=hypothesis_a,
            reason="candidate invalidated",
        ),
        now=BASE_TIME + timedelta(milliseconds=20),
    )
    assert instance.provider_instance_id in first.preserved_provider_instance_ids
    assert len(instance.active_lease_ids) == 1
    assert instance.lifecycle == ProviderInstanceLifecycle.ACTIVE

    second = cancellation.cancel(
        CancellationRequest(
            scope=CancellationScope.HYPOTHESIS,
            request_id="task_b",
            hypothesis_id=hypothesis_b,
            reason="candidate completed",
        ),
        now=BASE_TIME + timedelta(milliseconds=30),
    )
    assert instance.provider_instance_id in second.idle_provider_instance_ids
    assert instance.lifecycle == ProviderInstanceLifecycle.IDLE_LEASE

    draining = lifecycle.tick(now=BASE_TIME + timedelta(milliseconds=200))
    assert draining == (instance.provider_instance_id,)
    assert instance.lifecycle == ProviderInstanceLifecycle.DRAINING
    assert lifecycle.capacity.used("sensor_a").cpu_cores == 0


def test_phase4_follow_plans_share_the_yolo_detector_but_not_hypothesis_specific_follow_state() -> None:
    from fable.common.schemas import PredicateDemand
    from fable.planning.alternative_graph import PhysicalAlternativeGraphBuilder
    from fable.planning.beam_search import BeamSearchConfig, BoundedLabelPlanner
    from fable.planning.testing import fake_artifact_catalog, fake_follow_demand
    from fable.scheduling import candidate_from_search_result

    registry = fake_provider_registry()
    deployment = fake_deployment()
    from fable.planning.artifact_catalog import ArtifactCatalog
    artifacts = ArtifactCatalog(
        item for item in fake_artifact_catalog().artifacts
        if item.artifact_type != "detection_set.v1"
    )
    base = fake_follow_demand()

    def clone(request_id: str) -> PredicateDemand:
        payload = base.model_dump(mode="python")
        payload.update(
            {
                "demand_id": uuid7(),
                "request_id": request_id,
                "hypothesis_id": uuid7(),
                "frontier_id": uuid7(),
                "checkpoint_id": uuid7(),
                "sharing_key": None,
            }
        )
        return PredicateDemand.model_validate(payload)

    def plan(demand: PredicateDemand):
        graph = PhysicalAlternativeGraphBuilder(
            provider_registry=registry,
            artifact_catalog=artifacts,
            deployment=deployment,
        ).build((demand,), now=BASE_TIME)
        search = BoundedLabelPlanner(
            provider_registry=registry,
            artifact_catalog=artifacts,
            deployment=deployment,
            config=BeamSearchConfig(beam_width=8, run_oracle=False),
        ).search(graph, (demand,), now=BASE_TIME)
        return candidate_from_search_result(
            search,
            graph,
            (demand,),
            task_policy=TaskSchedulingPolicy(request_id=demand.request_id),
        )

    demand_a = clone("convoy_task_a")
    demand_b_payload = clone("convoy_task_b").model_dump(mode="python")
    demand_b_payload.update({"bound_roles": {"leader": "vehicle_99"}, "sharing_key": None})
    demand_b = PredicateDemand.model_validate(demand_b_payload)
    lifecycle = ProviderLifecycleManager(
        provider_registry=registry,
        capacity=CapacityLedger(deployment),
    )
    batch = MultiTenantScheduler(lifecycle=lifecycle).admit(
        (plan(demand_a), plan(demand_b)),
        now=BASE_TIME,
    )
    assert len(batch.admitted_plan_ids) == 2

    yolo_instances = [
        item
        for item in lifecycle.active_instances
        if item.share_key.provider_id == "yolo_vehicle_fast_640"
    ]
    follow_instances = [
        item
        for item in lifecycle.active_instances
        if item.share_key.provider_id == "follows_local_geometry"
    ]
    assert len(yolo_instances) == 1
    assert len(yolo_instances[0].active_lease_ids) == 2
    assert len(follow_instances) == 2
    assert all(len(item.active_lease_ids) == 1 for item in follow_instances)
