import json
from pathlib import Path

from evaluation.pilot import (
    PilotManifest,
    classify_failure,
    generate_pilot_report,
)


def _manifest() -> PilotManifest:
    return PilotManifest.model_validate(
        {
            "schema_version": "fable.bounded_pilot.v1",
            "pilot_id": "test_pilot",
            "settings": {"repetitions": 2, "model_id": "yolov8s.pt"},
            "cases": [
                {
                    "case_id": "positive",
                    "family": "robbery",
                    "experiment_id": "experiment",
                },
                {
                    "case_id": "control",
                    "family": "robbery",
                    "scenario": "scenario",
                    "variant": "Cross-sensor robbery",
                    "expected_positive": False,
                    "repetitions": 1,
                },
            ],
        }
    )


def test_failure_layer_distinguishes_runtime_and_graph() -> None:
    assert classify_failure(
        {"classification": "FALSE_NEGATIVE", "error": "readiness timeout"}
    ) == "RUNTIME"
    assert classify_failure(
        {
            "classification": "FALSE_NEGATIVE",
            "watch_registered": True,
            "admitted": True,
            "progress_statuses": {"APPLIED": 2},
        }
    ) == "SEMANTIC_GRAPH"
    assert classify_failure(
        {
            "classification": "FALSE_NEGATIVE",
            "watch_registered": True,
            "admitted": True,
            "variant": "Pass-follow-clear convoy",
            "vehicle_predicates_by_id": {"PASSES": 2, "FOLLOWS": 4},
        }
    ) == "SEMANTIC_GRAPH"


def test_report_scores_positive_and_control(tmp_path: Path) -> None:
    for name, classification in (
        ("positive-r01", "TRUE_POSITIVE"),
        ("control-r01", "NOT_DETECTED"),
    ):
        case_id = name.rsplit("-r", 1)[0]
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "classification": classification,
                    "pilot": {"case_id": case_id, "repetition": 1},
                    "provenance": {
                        "configuration_digest": "config",
                        "model_digest": "model",
                        "runner_arguments": {"model_id": "yolov8s.pt"},
                    },
                }
            ),
            encoding="utf-8",
        )

    report = generate_pilot_report(tmp_path, _manifest())

    assert report["planned_runs"] == 3
    assert report["completed_runs"] == 2
    assert report["by_family"]["robbery"]["pass_rate"] == 1.0
    assert report["by_family"]["robbery"]["positive_recall"] == 1.0
    assert report["by_family"]["robbery"]["control_specificity"] == 1.0
    assert report["configuration_consistent"] is True
    assert report["model_ids"] == ["yolov8s.pt"]
