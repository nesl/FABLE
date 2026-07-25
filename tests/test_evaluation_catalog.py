from pathlib import Path

from evaluation.catalog import ExperimentCatalog


ROOT = Path(__file__).resolve().parents[1]


def test_ground_truth_catalog_applies_campaign_spatial_scope() -> None:
    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv",
        transition_model_path=ROOT / "evaluation/labels/site_sensor_transition_model_2024_2025.json",
    )
    assert len(catalog.experiments) == 89
    assert catalog.summary().by_year == {2024: 49, 2025: 27, 2026: 13}

    convoy_2024 = next(
        item for item in catalog.experiments
        if item.campaign_year == 2024 and item.ce_variant == "Route convoy"
    )
    assert convoy_2024.spatial_topology_known
    assert convoy_2024.spatial_coordination_eligible
    assert convoy_2024.topology_deployment_ids == ("2024_temporal_ce1",)
    assert "orin_1" in convoy_2024.replay_supported_sensor_ids
    assert "n1" in convoy_2024.unavailable_mobile_sensor_ids

    chase_2024 = next(
        item for item in catalog.experiments
        if item.campaign_year == 2024 and item.ce_variant == "Two-vehicle chase"
    )
    assert chase_2024.topology_layout_ambiguous
    assert len(chase_2024.topology_deployment_ids) == 2

    experiment_2026 = next(item for item in catalog.experiments if item.campaign_year == 2026)
    assert not experiment_2026.spatial_topology_known
    assert not experiment_2026.spatial_coordination_eligible
    assert experiment_2026.replay_supported_sensor_ids == ()
    assert any("not available" in note for note in experiment_2026.spatial_notes)
