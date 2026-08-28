from evaluation.capacity_profiles import (
    ComputeCapacityProfile,
    ProfiledCapacityActionApplier,
    apply_compute_capacity_profile,
)
from evaluation.disturbance_schedule import DisturbanceKind, ScheduledDisturbanceAction
from evaluation.schemas import BaselineId, ResourceSample
from fable.common.examples import BASE_TIME
from fable.planning.testing import fake_deployment


def _profile(profile_id: str, fraction: float) -> ComputeCapacityProfile:
    return ComputeCapacityProfile(
        profile_id=profile_id,
        cpu_capacity_fraction=fraction,
        memory_capacity_fraction=fraction,
        gpu_capacity_fraction=fraction,
        execution_time_multiplier=2.25 if fraction < 1 else 1,
        queue_delay_ms=150 if fraction < 1 else 0,
    )


def test_compute_profile_changes_only_target_planner_capacity() -> None:
    deployment = fake_deployment()
    nominal = deployment.node("edge_1").capacity
    applied = apply_compute_capacity_profile(
        deployment,
        target_node_id="edge_1",
        profile=_profile("E1", 0.5),
        resource_epoch=4,
    )

    changed = applied.deployment.node("edge_1").capacity
    assert changed.cpu_cores == nominal.cpu_cores * 0.5
    assert changed.memory_mb == round(nominal.memory_mb * 0.5)
    assert applied.resource_epoch == 4
    assert deployment.node("edge_1").capacity == nominal


def test_profiled_capacity_action_emits_resource_epoch_sample() -> None:
    records = []
    applier = ProfiledCapacityActionApplier(
        deployment=fake_deployment(),
        profiles={"E1": _profile("E1", 0.5)},
        run_id="run",
        baseline_id=BaselineId.FABLE,
        trace_id="trace",
        request_id="request",
        record_sink=records.append,
    )
    action = ScheduledDisturbanceAction(
        step_id="compute",
        action="APPLY",
        kind=DisturbanceKind.CAPACITY_PROFILE,
        target_id="edge_1",
        condition_id="E1",
        due_at=BASE_TIME,
    )

    measurements = applier(action, 2)

    assert measurements["profile_id"] == "E1"
    assert len(records) == 1
    assert isinstance(records[0], ResourceSample)
    assert records[0].metadata["condition_epoch"] == 2


def test_physical_link_profile_changes_only_target_links() -> None:
    deployment = fake_deployment()
    target = "edge_1"
    adjusted = deployment.with_degraded_node_links(
        target,
        bandwidth_mbps=20,
        added_latency_ms=25,
        policy_tag="physical:test",
    )

    target_links = [
        link for link in adjusted.links
        if target in {link.source_node_id, link.target_node_id}
    ]
    assert target_links
    assert all(link.bandwidth_mbps <= 20 for link in target_links)
    assert all("physical:test" in link.policy_tags for link in target_links)
    untouched = [
        link for link in deployment.links
        if target not in {link.source_node_id, link.target_node_id}
    ]
    assert [
        link for link in adjusted.links
        if target not in {link.source_node_id, link.target_node_id}
    ] == untouched
    assert deployment.links != adjusted.links
