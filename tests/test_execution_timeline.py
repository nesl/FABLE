from __future__ import annotations

import json

from evaluation.execution_timeline import write_execution_timeline


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compact_timeline_reports_plan_deltas_and_provider_changes(tmp_path):
    _write(
        tmp_path / "plan_decision.jsonl",
        [
            {
                "monotonic_timestamp_ns": 1_000_000_000,
                "selected_node_ids": ["sensor"],
                "activated_provider_keys": ["yolo@sensor"],
                "replan_trigger": "INITIAL_ADMISSION",
                "reason": "initial",
            },
            {
                "monotonic_timestamp_ns": 3_000_000_000,
                "selected_node_ids": ["site"],
                "activated_provider_keys": ["yolo@site"],
                "replan_trigger": "RESOURCE_CHANGE",
                "reason": "contention",
            },
        ],
    )
    _write(
        tmp_path / "provider_command.jsonl",
        [{
            "monotonic_timestamp_ns": 3_100_000_000, "command": "ACTIVATE",
            "provider_id": "yolo", "node_id": "site",
        }],
    )
    _write(tmp_path / "provider_lifecycle.jsonl", [])
    result = {
        "disturbance_results": [{
            "applied_offset_s": 30, "action": "APPLY",
            "condition": "E1", "transition_id": "gpu-load",
        }]
    }

    report = write_execution_timeline(result, tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "execution_changes.jsonl").read_text().splitlines()]

    assert report["event_count"] == 4
    change = next(row for row in rows if row["event"] == "PLAN_CHANGED")
    assert change["added_nodes"] == "site"
    assert change["removed_nodes"] == "sensor"
    assert change["trigger"] == "RESOURCE_CHANGE"
    assert any(row["event_kind"] == "DISTURBANCE" for row in rows)
