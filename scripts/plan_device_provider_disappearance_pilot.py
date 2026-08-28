#!/usr/bin/env python3
"""Plan a paired Orin11 YOLO-disappearance adaptation pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.condition_trace import midpoint_disturbance_schedule
from evaluation.experiments.matrix import PlannedRun
from evaluation.experiments.specs import ExperimentQuestion
from evaluation.schemas import BaselineId, EvaluationMode
from fable.common.ids import deterministic_id


OUT = ROOT / "evaluation/manifests/adaptation/device_provider_disappearance"
EXPERIMENT = "20260415-pass-follow-clear-convoy-3-car-convoy-11-r011"
REPLAY_DURATION_SECONDS = 120.0
TARGET = "dvpg_gq_orin_11:yolo_vehicle_fast_640"
POLICIES = (
    BaselineId.B1_STATIC_WHOLE_EVENT,
    BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
    BaselineId.FABLE,
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failed_at, restored_at = midpoint_disturbance_schedule(
        REPLAY_DURATION_SECONDS
    )
    traces = {
        "N0": {
            "schema_version": "fable.condition_trace.v1",
            "trace_id": "device-provider-n0",
            "initial_network_profile": "N0",
            "initial_compute_profile": "N0",
            "anchor": "TRACE_START",
            "transitions": [],
            "duration_s": 180.0,
            "random_seed": 33001,
        },
        "F1": {
            "schema_version": "fable.condition_trace.v1",
            "trace_id": "device-provider-orin11-yolo-f1",
            "initial_network_profile": "N0",
            "initial_compute_profile": "N0",
            "anchor": "TRACE_START",
            "transitions": [
                {
                    "transition_id": "orin11-yolo-unavailable",
                    "offset_s": failed_at,
                    "action": "FAIL_PROVIDER",
                    "target_id": TARGET,
                },
                {
                    "transition_id": "orin11-yolo-restored",
                    "offset_s": restored_at,
                    "action": "RESTORE_PROVIDER",
                    "target_id": TARGET,
                },
            ],
            "duration_s": 180.0,
            "random_seed": 33001,
        },
    }
    paths: dict[str, Path] = {}
    for condition, trace in traces.items():
        path = OUT / f"{condition.lower()}.json"
        path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        paths[condition] = path

    rows = []
    for condition in ("N0", "F1"):
        for policy in POLICIES:
            identity = {
                "experiment": EXPERIMENT,
                "condition": condition,
                "policy": policy.value,
                "target": TARGET,
            }
            rows.append(PlannedRun(
                run_id=deterministic_id("eval_run", identity, length=32),
                question=ExperimentQuestion.RQ3_OPERATING_ADAPTATION,
                experiment_id=EXPERIMENT,
                baseline_id=policy,
                mode=EvaluationMode.FULL_STACK,
                network_profile_id="good_network",
                network_profile_path=str(
                    ROOT / "netwaggle/configs/profiles/site_evaluation_29node/N0.json"
                ),
                repetition=1,
                random_seed=33001,
                playback_mode="realtime",
                provider_profile_version="device-provider-disappearance-v1",
                disturbance_profile_id=condition,
                condition_trace_id=traces[condition]["trace_id"],
                condition_trace_path=str(paths[condition]),
                ce_start_offset_seconds=0,
                provider_execution_mode="real",
                vlm_mode="replayed_response",
                campaign_year=2026,
            ))

    manifest = OUT / "device_provider_disappearance_pilot_6.jsonl"
    manifest.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "fable.device_provider_disappearance_plan.v1",
        "manifest": str(manifest),
        "experiment_id": EXPERIMENT,
        "replay_duration_seconds": REPLAY_DURATION_SECONDS,
        "target": TARGET,
        "conditions": ["N0", "F1"],
        "systems": [item.value for item in POLICIES],
        "planned_cells": len(rows),
        "schedule_fraction": {"failure": 0.50, "restore": 0.95},
        "schedule_seconds": {"failure": failed_at, "restore": restored_at},
        "expected_fallback": "yolo_vehicle_fast_640@x86server",
        "playback_mode": "realtime",
        "execution_order": "condition-major; B1, B3, FABLE",
    }
    (OUT / "device_provider_disappearance_pilot_6.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
