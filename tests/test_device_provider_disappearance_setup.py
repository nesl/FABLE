from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/manifests/adaptation/device_provider_disappearance"


def test_device_provider_failure_targets_orin11_yolo_for_middle_45_percent():
    trace = json.loads((OUT / "f1.json").read_text(encoding="utf-8"))
    assert [(row["offset_s"], row["action"], row["target_id"]) for row in trace["transitions"]] == [
        (60.0, "FAIL_PROVIDER", "dvpg_gq_orin_11:yolo_vehicle_fast_640"),
        (114.0, "RESTORE_PROVIDER", "dvpg_gq_orin_11:yolo_vehicle_fast_640"),
    ]


def test_device_provider_pilot_is_paired_and_realtime():
    rows = [
        json.loads(line)
        for line in (OUT / "device_provider_disappearance_pilot_6.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 6
    assert {row["disturbance_profile_id"] for row in rows} == {"N0", "F1"}
    assert all(row["playback_mode"] == "realtime" for row in rows)
    assert {row["baseline_id"] for row in rows} == {
        "B1_STATIC_WHOLE_EVENT", "B3_TASK_RESOURCE_ADAPTIVE", "FABLE"
    }
