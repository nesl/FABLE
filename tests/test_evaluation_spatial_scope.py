from pathlib import Path

from evaluation.spatial_scope import ReplaySpatialScope
from fable.spatial import SpatialObservation
from fable.spatial.transition_model import SiteSensorTransitionModel, load_sensor_bindings


ROOT = Path(__file__).resolve().parents[1]


def test_spatial_scope_keeps_orin_and_defers_mobile_replay() -> None:
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
            maximum_observation_groups=1,
        ),
        bindings=bindings,
    )
    selection = ReplaySpatialScope(model).apply(prediction, campaign_year=2025)
    assert selection.evaluation_eligible
    assert "orin_5" in selection.supported_sensor_ids
    assert "d3" in selection.unavailable_mobile_sensor_ids
    assert all("orin" in item for item in selection.supported_source_ids)


def test_spatial_scope_disables_2026_topology_metrics() -> None:
    model = SiteSensorTransitionModel.from_json(
        ROOT / "evaluation/labels/site_sensor_transition_model_2024_2025.json"
    )
    prediction = model.predict(
        SpatialObservation(current_sensor_id="orin_6", observed_heading="SW")
    )
    selection = ReplaySpatialScope(model).apply(prediction, campaign_year=2026)
    assert not selection.topology_available
    assert not selection.evaluation_eligible
    assert selection.supported_sensor_ids == ()
