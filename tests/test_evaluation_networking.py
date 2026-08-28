from pathlib import Path

from evaluation.networking import (
    ProfiledNetworkActionApplier,
    NetworkExperimentState,
    apply_netwaggle_profile,
    load_netwaggle_profile,
    network_condition_records,
    retarget_sensor_uplink_profile,
    validated_link_target_node_id,
)
from evaluation.disturbance_schedule import DisturbanceKind, ScheduledDisturbanceAction
from evaluation.schemas import BaselineId
from evaluation.runner import EvaluationRunner
from evaluation.schemas import EvaluationMode
from fable.common.examples import BASE_TIME
from fable.distributed.config import load_deployment_graph
from fable.planning.testing import fake_deployment


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "netwaggle/configs/profiles"


def test_validated_link_targets_include_orin_and_mobile_sensors() -> None:
    assert validated_link_target_node_id("link:s_orin14:s_edge") == "dvpg_gq_orin_14"
    assert validated_link_target_node_id("link:s_mob6:s_edge") == "mobile_archive_6"


def test_netwaggle_profile_replaces_planner_network_costs() -> None:
    deployment = load_deployment_graph(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    profile = load_netwaggle_profile(PROFILES / "high_latency_cloud.json")
    applied = apply_netwaggle_profile(deployment, profile, resource_epoch=3)
    path = applied.deployment.shortest_path("dvpg_gq_orin_11", "cloud1")
    assert path is not None
    assert path.latency_ms == 185
    assert path.bottleneck_bandwidth_mbps == 100
    assert applied.resource_epoch == 3


def test_profiled_network_action_emits_planner_visible_condition_epoch() -> None:
    deployment = fake_deployment()
    profile = load_netwaggle_profile(PROFILES / "high_latency_cloud.json")
    records = []
    applier = ProfiledNetworkActionApplier(
        deployment=deployment,
        profiles={"W1": profile},
        run_id="run",
        baseline_id=BaselineId.FABLE,
        trace_id="trace",
        request_id="request",
        record_sink=records.append,
        node_switches={
            "sensor_a": "s_orin11",
            "sensor_b": "s_orin12",
            "edge_1": "s_edge",
            "server_1": "s_cloud",
        },
    )
    action = ScheduledDisturbanceAction(
        step_id="wan",
        action="APPLY",
        kind=DisturbanceKind.NETWORK_PROFILE,
        target_id="site_to_cloud",
        condition_id="W1",
        due_at=BASE_TIME,
    )

    measurements = applier(action, 3)

    assert measurements["profile_id"] == profile.profile_id
    assert applier.latest is not None
    assert applier.latest.resource_epoch == 3
    assert records
    assert all(item.condition_epoch == 3 for item in records)


def test_loss_reduces_effective_planner_bandwidth() -> None:
    deployment = load_deployment_graph(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    profile = load_netwaggle_profile(PROFILES / "lossy_edge.json")
    applied = apply_netwaggle_profile(deployment, profile, resource_epoch=0)
    path = applied.deployment.shortest_path("dvpg_gq_orin_11", "x86server")
    assert path is not None
    assert path.latency_ms == 25
    assert path.bottleneck_bandwidth_mbps == 29.4


def test_site_local_condition_can_target_selected_replay_sensor() -> None:
    profile = load_netwaggle_profile(
        PROFILES / "site_local_20node" / "L1.json"
    )
    assert profile.profile_id == "l1"
    targeted = retarget_sensor_uplink_profile(
        profile, target_switch="s_orin14"
    )
    edge_link = next(
        item
        for item in targeted.links
        if {item.source_switch, item.target_switch} == {"s_orin14", "s_edge"}
    )
    assert edge_link.bandwidth_mbps == 5
    assert edge_link.latency_ms == 40
    assert edge_link.packet_loss_fraction == 0.02


def test_profile_change_advances_resource_epoch(tmp_path) -> None:
    deployment = load_deployment_graph(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    state = NetworkExperimentState(deployment)
    good = load_netwaggle_profile(PROFILES / "good_network.json")
    degraded = load_netwaggle_profile(PROFILES / "cloud_degraded.json")
    assert state.activate(good).resource_epoch == 0
    assert state.activate(good).resource_epoch == 0
    changed = state.activate(degraded)
    assert changed.resource_epoch == 1
    assert (
        changed.deployment.shortest_path("dvpg_gq_orin_11", "cloud1").latency_ms
        == 128
    )
    records = network_condition_records(
        changed,
        run_id="run",
        baseline_id=BaselineId.FABLE,
        trace_id="trace",
        request_id="request",
        event_time=BASE_TIME,
    )
    assert records
    assert all(item.condition_epoch == 1 for item in records)
    assert all(item.metadata["netwaggle_profile"] == "cloud_degraded" for item in records)
    runner = EvaluationRunner(tmp_path, mode=EvaluationMode.FULL_STACK)
    assert runner.record_network_conditions(records) == records
    assert len(runner.store.read("network_condition")) == len(records)
