from datetime import UTC, datetime
from pathlib import Path

from evaluation.spatial_lookout_v2 import (
    SensorOperatingState,
    SpatialLookoutCoordinator,
    SpatialLookoutPolicyId,
)
from fable.spatial import (
    SiteSensorTransitionModel,
    SpatialObservation,
    load_sensor_bindings,
)


ROOT = Path(__file__).resolve().parents[1]


def _coordinator() -> SpatialLookoutCoordinator:
    return SpatialLookoutCoordinator(
        model=SiteSensorTransitionModel.from_json(
            ROOT / "evaluation/labels/site_sensor_transition_model_2024_2025.json"
        ),
        bindings=load_sensor_bindings(
            ROOT / "iobt-minimal-ce-replay/config/fable_spatial_bindings.yaml"
        ),
    )


def _states() -> dict[str, SensorOperatingState]:
    return {
        "orin_5": SensorOperatingState(
            sensor_id="orin_5", normalized_load=0.7, provider_ready_ms=500
        ),
        "d3": SensorOperatingState(
            sensor_id="d3", normalized_load=0.1, provider_ready_ms=100
        ),
        "orin_1": SensorOperatingState(sensor_id="orin_1"),
    }


def _update(policy: SpatialLookoutPolicyId, *, bound: bool = True):
    return _coordinator().update(
        policy_id=policy,
        observation=SpatialObservation(
            current_sensor_id="orin_6",
            observed_heading="SW",
            active_deployment_id="2025_package_exchange",
            object_binding_id="vehicle:canonical-7" if bound else None,
        ),
        semantic_frontier_id="frontier-after-arrival",
        prediction_time=datetime(2026, 8, 2, tzinfo=UTC),
        operating_state_by_sensor=_states(),
        maximum_active_sensors=1,
        actual_downstream_sensor_id="orin_5",
    )


def test_fable_ranks_topology_before_load_and_emits_bound_activation():
    update = _update(SpatialLookoutPolicyId.FABLE_SPATIAL)

    assert update.decision.activated_sensor_ids == ("d3",)
    assert update.intents[0].action == "ACTIVATE"
    assert update.intents[0].object_binding_id == "vehicle:canonical-7"
    assert update.intents[0].semantic_frontier_id == "frontier-after-arrival"


def test_fable_refuses_identity_free_predictive_handoff():
    update = _update(SpatialLookoutPolicyId.FABLE_SPATIAL, bound=False)

    assert update.decision.activated_sensor_ids == ()
    assert update.intents == ()


def test_new_baselines_have_distinct_activation_semantics():
    no_handoff = _update(SpatialLookoutPolicyId.S0_NO_HANDOFF).decision
    broadcast = _update(SpatialLookoutPolicyId.S1_BROADCAST).decision
    topology = _update(SpatialLookoutPolicyId.S2_TOPOLOGY_ONLY).decision
    resource = _update(SpatialLookoutPolicyId.S3_RESOURCE_AWARE_TOPOLOGY).decision
    oracle = _update(SpatialLookoutPolicyId.O_SPACE).decision

    assert no_handoff.activated_sensor_ids == ()
    assert broadcast.activated_sensor_ids == ("d3", "orin_1", "orin_5")
    assert topology.activated_sensor_ids == ("d3", "orin_5")
    assert resource.activated_sensor_ids == ("d3",)
    assert oracle.activated_sensor_ids == ("orin_5",)


def test_frontier_update_releases_stale_lookouts():
    update = _coordinator().update(
        policy_id=SpatialLookoutPolicyId.FABLE_SPATIAL,
        observation=SpatialObservation(
            current_sensor_id="orin_6",
            observed_heading="SW",
            active_deployment_id="2025_package_exchange",
            object_binding_id="vehicle:canonical-7",
        ),
        semantic_frontier_id="frontier-after-arrival",
        prediction_time=datetime(2026, 8, 2, tzinfo=UTC),
        operating_state_by_sensor=_states(),
        previously_active_sensor_ids=("orin_1",),
        maximum_active_sensors=1,
    )

    assert [(item.action, item.sensor_id) for item in update.intents] == [
        ("RELEASE", "orin_1"),
        ("ACTIVATE", "d3"),
    ]
