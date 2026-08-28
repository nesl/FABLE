#!/usr/bin/env python3
"""Build the paired nominal/degraded fan-out validation matrix."""

from __future__ import annotations

import json
from pathlib import Path

from fable.common.ids import deterministic_id


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "evaluation/manifests/adaptation/rq3a_updated/network_fanout_validation_4.jsonl"
)
NOMINAL_OUTPUT = OUTPUT.with_name("network_fanout_validation_nominal_2.jsonl")
DEGRADED_OUTPUT = OUTPUT.with_name("network_fanout_validation_degraded_2.jsonl")
TRACE = ROOT / "evaluation/manifests/adaptation/rq3a_updated/n1_c1_test_suv_orin14.json"
EXPERIMENT = "20260413-pass-follow-clear-convoy-c1-test-suv-r004"
BASELINES = ("FABLE", "B1_STATIC_WHOLE_EVENT")


def main() -> int:
    rows = []
    for condition in ("nominal", "degraded"):
        for baseline in BASELINES:
            trace_id = (
                "rq3a-updated-n1-c1-test-suv-orin14"
                if condition == "degraded"
                else None
            )
            identity = {
                "experiment_id": EXPERIMENT,
                "baseline_id": baseline,
                "condition": condition,
                "repetition": 1,
            }
            row = {
                    "schema_version": "fable.planned_run.v1",
                    "run_id": deterministic_id("eval_run", identity, length=32),
                    "question": "RQ3_OPERATING_ADAPTATION",
                    "experiment_id": EXPERIMENT,
                    "baseline_id": baseline,
                    "mode": "FULL_STACK",
                    "network_profile_id": "good_network",
                    "network_profile_path": str(
                        ROOT
                        / "netwaggle/configs/profiles/site_evaluation_29node/N0.json"
                    ),
                    "spatial_metrics_enabled": False,
                    "repetition": 1,
                    "random_seed": 31004,
                    "playback_mode": "realtime",
                    "provider_profile_version": "rq3a-network-fanout-repair-v1",
                    "disturbance_profile_id": "N1" if trace_id else "N0",
                    "ce_start_offset_seconds": 0.0,
                    "provider_execution_mode": "real",
                    "vlm_mode": "replayed_response",
                    "warnings": [
                        "2026 trace has no calibrated spatial prior; exclude topology-based spatial metrics."
                    ],
                    "campaign_year": 2026,
                    "replay_supported_sensor_ids": [],
                    "unavailable_mobile_sensor_ids": [],
                    "topology_deployment_ids": [],
                }
            if trace_id:
                row["condition_trace_id"] = trace_id
                row["condition_trace_path"] = str(TRACE)
            rows.append(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    NOMINAL_OUTPUT.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
            if not row.get("condition_trace_id")
        ),
        encoding="utf-8",
    )
    DEGRADED_OUTPUT.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
            if row.get("condition_trace_id")
        ),
        encoding="utf-8",
    )
    print(OUTPUT)
    print(NOMINAL_OUTPUT)
    print(DEGRADED_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
