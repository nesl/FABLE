from pathlib import Path

from evaluation import CellOutcome, load_manifest, summarize_outcomes
from fable.planning import LinkState, NodeState, RuntimeState
from netwaggle import NetwaggleLinkObservation, apply_link_observations
from evaluation.runner import run_planning_cell
from evaluation.catalog import ExperimentRecord, group_counts, load_experiment_catalog


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_manifest_resolves_repository_inputs() -> None:
    manifest = load_manifest(ROOT / "evaluation/manifests/smoke.yaml")
    assert manifest.name == "refactored_core_smoke"
    assert manifest.cells[0].event.name == "convoy.yaml"
    assert manifest.cells[0].deployment.name == "deployment.example.yaml"


def test_netwaggle_bridge_replaces_only_observed_link() -> None:
    state = RuntimeState(
        nodes={"a": NodeState("a", "sensor"), "b": NodeState("b", "edge")},
        sources={},
        links=(LinkState("a", "b", 1.0, 1000.0, True),),
        profiles={},
    )
    changed = apply_link_observations(
        state, (NetwaggleLinkObservation("a", "b", 50.0, 5.0, False),)
    )
    assert len(changed.links) == 1
    assert changed.links[0].source_node == "a"
    assert changed.links[0].destination_node == "b"
    assert changed.links[0].latency_ms == 50.0
    assert changed.links[0].bandwidth_mbps == 5.0
    assert changed.links[0].available is False
    assert state.links[0].available is True


def test_evaluation_summary_is_grouped_by_policy() -> None:
    summary = summarize_outcomes(
        (
            CellOutcome("a", "convoy", "FABLE", "SUCCESS", 1.0, 2),
            CellOutcome("b", "convoy", "FABLE", "FAILED", 3.0, 0),
        )
    )
    assert summary["policies"]["FABLE"]["successful"] == 1
    assert summary["policies"]["FABLE"]["mean_elapsed_seconds"] == 2.0


def test_all_smoke_policies_produce_explicit_plans() -> None:
    manifest = load_manifest(ROOT / "evaluation/manifests/smoke.yaml")
    rows = [run_planning_cell(cell)[0] for cell in manifest.cells]
    assert {row.policy for row in rows} == {
        "FABLE", "B1_STATIC", "B3_RESOURCE", "B4_GREEDY"
    }
    assert all(row.status == "SUCCESS" for row in rows)
    assert all(row.planned_provider_count > 0 for row in rows)


def test_experiment_catalog_has_typed_rows() -> None:
    rows = load_experiment_catalog(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv"
    )
    assert rows
    assert isinstance(rows[0], ExperimentRecord)
    assert rows[0].campaign_year in {2024, 2025, 2026}
    assert sum(group_counts(rows).values()) == len(rows)
