import json

from evaluation.report import generate_evaluation_report


def test_report_generates_deterministic_csv_and_exclusions(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "plan_decision.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "baseline_id": "FABLE",
                "trace_id": "trace-1",
                "planning_latency_ms": 2.5,
                "predicted_transfer_bytes": 100,
            }
        )
        + "\n"
    )
    output = tmp_path / "report"
    report = generate_evaluation_report((run, tmp_path / "missing"), output)
    assert report["included_run_ids"] == ["run-1"]
    assert report["record_counts"]["plan_decision"] == 1
    assert report["excluded_input_count"] == 1
    assert (output / "summary.json").is_file()
    assert "FABLE" in (output / "summary.csv").read_text()
    assert "does not exist" in (output / "run_exclusions.csv").read_text()
    assert (output / "confidence_intervals.csv").is_file()
    assert (output / "paired_comparisons.csv").is_file()
