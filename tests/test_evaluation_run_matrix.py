from pathlib import Path

from evaluation.catalog import ExperimentCatalog
from evaluation.experiments.matrix import build_run_matrix
from evaluation.experiments.specs import ExperimentQuestion

ROOT = Path(__file__).resolve().parents[1]


def _catalog():
    return ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv",
        transition_model_path=ROOT / "evaluation/labels/site_sensor_transition_model_2024_2025.json",
    )


def test_spatial_run_matrix_uses_only_2024_2025_known_topology() -> None:
    catalog = _catalog()
    runs = build_run_matrix(catalog, ExperimentQuestion.RQ3_SPATIAL_COORDINATION)
    assert runs
    ids = {item.experiment_id for item in runs}
    assert all(catalog.by_id[item].campaign_year in (2024, 2025) for item in ids)
    assert all(item.spatial_metrics_enabled for item in runs)


def test_end_to_end_matrix_matches_current_recommended_catalog() -> None:
    catalog = _catalog()
    runs = build_run_matrix(catalog, ExperimentQuestion.RQ1_END_TO_END)
    ids = {item.experiment_id for item in runs}
    recommended = {item.experiment_id for item in catalog.recommended()}
    assert ids <= recommended
    assert any("three-visit-stalking" in item for item in ids)
