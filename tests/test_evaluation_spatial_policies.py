from pathlib import Path

from evaluation.spatial_policies import (
    BroadcastSpatialPolicy,
    FableSpatialPolicy,
    OracleSpatialPolicy,
    ResourceOnlySpatialPolicy,
    SpatialSelectionContext,
    TopologyShortlistPolicy,
)
from evaluation.spatial_scope import ReplaySpatialScope
from fable.spatial import SpatialObservation
from fable.spatial.transition_model import SiteSensorTransitionModel, load_sensor_bindings

ROOT = Path(__file__).resolve().parents[1]


def _context():
    model = SiteSensorTransitionModel.from_json(
        ROOT / "evaluation/labels/site_sensor_transition_model_2024_2025.json"
    )
    bindings = load_sensor_bindings(
        ROOT / "iobt-minimal-ce-replay/config/fable_spatial_bindings.yaml"
    )
    prediction = model.predict(
        SpatialObservation(
            current_sensor_id="orin_6",
            observed_heading="SW",
            active_deployment_id="2025_package_exchange",
        ),
        bindings=bindings,
    )
    scope = ReplaySpatialScope(model).apply(
        prediction,
        campaign_year=2025,
        available_replay_sensor_ids=("orin_1", "orin_5", "orin_6"),
    )
    return SpatialSelectionContext(
        replay_scope=scope,
        resource_load_by_sensor={"orin_1": 0.8, "orin_5": 0.2, "orin_6": 0.4},
        maximum_active_sensors=1,
        actual_downstream_sensor_id="orin_5",
    )


def test_spatial_policies_preserve_distinct_comparison_semantics() -> None:
    context = _context()
    broadcast = BroadcastSpatialPolicy().select(context)
    topology = TopologyShortlistPolicy().select(context)
    resource = ResourceOnlySpatialPolicy().select(context)
    fable = FableSpatialPolicy().select(context)
    oracle = OracleSpatialPolicy().select(context)

    assert set(broadcast.activated_sensor_ids) == {"orin_1", "orin_5", "orin_6"}
    assert topology.activated_sensor_ids == ("orin_5",)
    assert resource.activated_sensor_ids == ("orin_5",)
    assert fable.activated_sensor_ids == ("orin_5",)
    assert oracle.activated_sensor_ids == ("orin_5",)
    assert "d3" in fable.unavailable_mobile_sensor_ids
