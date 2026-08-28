from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "evaluation/manifests/adaptation/compute_contention"


def test_compute_contention_trace_uses_replay_midpoint_schedule() -> None:
    trace = json.loads((MANIFEST_DIR / "e1.json").read_text(encoding="utf-8"))
    assert trace["anchor"] == "TRACE_START"
    assert trace["initial_network_profile"] == "N0"
    assert [(row["offset_s"], row["action"]) for row in trace["transitions"]] == [
        (60.0, "APPLY_COMPUTE_CONTENTION"),
        (114.0, "CLEAR_COMPUTE_CONTENTION"),
    ]


def test_compute_contention_pilot_is_balanced_and_realtime() -> None:
    rows = [
        json.loads(line)
        for line in (MANIFEST_DIR / "compute_contention_pilot_10.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 10
    assert {row["disturbance_profile_id"] for row in rows} == {"N0", "E1"}
    assert all(row["playback_mode"] == "realtime" for row in rows)
    counts = {
        condition: sum(row["disturbance_profile_id"] == condition for row in rows)
        for condition in ("N0", "E1")
    }
    assert counts == {"N0": 5, "E1": 5}


def test_contention_invocations_are_isolated_and_argument_array_based() -> None:
    source = (ROOT / "evaluation/compute_contention.py").read_text(encoding="utf-8")
    assert source.count('"--network", "none"') == 2
    assert "shell=False" in source
    assert "shell=True" not in source


def test_compute_pilot_records_two_gpu_partition() -> None:
    summary = json.loads(
        (MANIFEST_DIR / "compute_contention_pilot_10.summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["disturbed_tier"] == "site_local"
    assert summary["unaffected_tier"] == "device"
    assert summary["gpu_partition"]["device_gpu_uuid"] != summary["gpu_partition"]["site_gpu_uuid"]


def test_contention_is_not_charged_to_evaluated_system() -> None:
    from evaluation.resource_monitor import _container_category

    assert _container_category("fable-gpu-contention") == "disturbance_workload"
    assert _container_category("yolo-detector-orin11") == "evaluated_system"
