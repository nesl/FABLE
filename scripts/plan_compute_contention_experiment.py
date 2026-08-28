#!/usr/bin/env python3
"""Generate the bounded site-local GPU contention pilot."""

from __future__ import annotations

import json
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.experiments.matrix import PlannedRun
from evaluation.experiments.specs import ExperimentQuestion
from evaluation.gpu_partition import resolve_gpu_partition
from evaluation.condition_trace import midpoint_disturbance_schedule
from evaluation.schemas import BaselineId, EvaluationMode
from fable.common.ids import deterministic_id

OUT = ROOT / "evaluation/manifests/adaptation/compute_contention"
# This CE has executable detector/tracker/evaluator realizations at both the
# device and site-local tiers, so GPU-1 contention can cause a meaningful
# placement change instead of merely slowing a site-only identity operator.
EXPERIMENT = "20260415-pass-follow-clear-convoy-3-car-convoy-11-r011"
POLICIES = (
    BaselineId.B1_STATIC_WHOLE_EVENT,
    BaselineId.B2_FRONTIER_FIXED_REALIZATION,
    BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
    BaselineId.B4_GREEDY_FRONTIER,
    BaselineId.FABLE,
)
PILOT_POLICIES = (
    BaselineId.B1_STATIC_WHOLE_EVENT,
    BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
    BaselineId.FABLE,
)
SITE_HEAVY_POLICIES = (
    BaselineId.B1_STATIC_WHOLE_EVENT,
    BaselineId.FABLE,
)


def experiment_duration_seconds(experiment_id: str) -> float:
    """Return the labeled raw replay duration for one experiment."""

    labels = ROOT / "evaluation/labels/filtered_complex_event_experiments.csv"
    with labels.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("experiment_id") == experiment_id:
                return float(row["duration_seconds"])
    raise ValueError(f"experiment is absent from {labels}: {experiment_id}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gpu_partition = resolve_gpu_partition()
    replay_duration_s = experiment_duration_seconds(EXPERIMENT)
    contention_start_s, contention_end_s = midpoint_disturbance_schedule(
        replay_duration_s
    )
    traces = {
        "N0": {
            "schema_version": "fable.condition_trace.v1", "trace_id": "compute-contention-n0",
            "initial_network_profile": "N0", "initial_compute_profile": "N0",
            "anchor": "TRACE_START", "transitions": [], "duration_s": 360, "random_seed": 32001,
        },
        "E1": {
            "schema_version": "fable.condition_trace.v1", "trace_id": "compute-contention-e1",
            "initial_network_profile": "N0", "initial_compute_profile": "N0", "anchor": "TRACE_START",
            "transitions": [
                {"transition_id": "gpu-contention-apply", "offset_s": contention_start_s,
                 "action": "APPLY_COMPUTE_CONTENTION", "target_id": "x86server", "profile_id": "E1"},
                {"transition_id": "gpu-contention-clear", "offset_s": contention_end_s,
                 "action": "CLEAR_COMPUTE_CONTENTION", "target_id": "x86server", "profile_id": "N0"},
            ], "duration_s": 360, "random_seed": 32001,
        },
    }
    paths = {}
    for condition, trace in traces.items():
        path = OUT / f"{condition.lower()}.json"
        path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        paths[condition] = path
    rows = []
    for condition in ("N0", "E1"):
        for policy in POLICIES:
            identity = {"experiment": EXPERIMENT, "condition": condition, "policy": policy.value}
            rows.append(PlannedRun(
                run_id=deterministic_id("eval_run", identity, length=32),
                question=ExperimentQuestion.RQ3_OPERATING_ADAPTATION,
                experiment_id=EXPERIMENT, baseline_id=policy, mode=EvaluationMode.FULL_STACK,
                network_profile_id="good_network",
                network_profile_path=str(ROOT / "netwaggle/configs/profiles/site_evaluation_29node/N0.json"),
                repetition=1, random_seed=32001, playback_mode="realtime",
                provider_profile_version="compute-contention-v1",
                disturbance_profile_id=condition, condition_trace_id=traces[condition]["trace_id"],
                condition_trace_path=str(paths[condition]), ce_start_offset_seconds=0,
                provider_execution_mode="real", vlm_mode="replayed_response", campaign_year=2026,
            ))
    manifest = OUT / "compute_contention_pilot_10.jsonl"
    manifest.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    summary = {
        "schema_version": "fable.compute_contention_plan.v1", "manifest": str(manifest),
        "experiment_id": EXPERIMENT, "planned_cells": len(rows),
        "conditions": ["N0", "E1"], "systems": [item.value for item in POLICIES],
        "replay_duration_seconds": replay_duration_s,
        "schedule_seconds": {
            "nominal_until": contention_start_s,
            "contention_until": contention_end_s,
            "recovery_until": replay_duration_s,
        },
        "schedule_anchor": "REPLAY_MIDPOINT_DERIVED_FROM_LABEL_DURATION",
        "playback_mode": "realtime",
        "contention_validation": {"gpu_utilization_percent": [70, 90], "provider_p95_slowdown": [1.8, 2.5]},
        "network_condition": "N0 throughout",
        "gpu_partition": gpu_partition.as_dict(),
        "disturbed_tier": "site_local",
        "unaffected_tier": "device",
    }
    (OUT / "compute_contention_pilot_10.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pilot_rows = [row for row in rows if row.baseline_id in PILOT_POLICIES]
    pilot_manifest = OUT / "compute_contention_pilot_6.jsonl"
    pilot_manifest.write_text(
        "".join(row.model_dump_json() + "\n" for row in pilot_rows),
        encoding="utf-8",
    )
    pilot_summary = {
        **summary,
        "manifest": str(pilot_manifest),
        "planned_cells": len(pilot_rows),
        "systems": [item.value for item in PILOT_POLICIES],
        "purpose": "single-trace B1/B3/FABLE discrimination pilot",
    }
    (OUT / "compute_contention_pilot_6.summary.json").write_text(
        json.dumps(pilot_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    site_heavy_experiment = "20260415-cross-sensor-robbery-robbery-13-r013"
    site_heavy_duration = experiment_duration_seconds(site_heavy_experiment)
    site_start, site_end = midpoint_disturbance_schedule(site_heavy_duration)
    site_trace = {
        "schema_version": "fable.condition_trace.v1",
        "trace_id": "compute-contention-site-heavy-e1",
        "initial_network_profile": "N0",
        "initial_compute_profile": "N0",
        "anchor": "TRACE_START",
        "transitions": [
            {
                "transition_id": "gpu-contention-site-heavy-apply",
                "offset_s": site_start,
                "action": "APPLY_COMPUTE_CONTENTION",
                "target_id": "x86server",
                "profile_id": "E1",
            },
            {
                "transition_id": "gpu-contention-site-heavy-clear",
                "offset_s": site_end,
                "action": "CLEAR_COMPUTE_CONTENTION",
                "target_id": "x86server",
                "profile_id": "N0",
            },
        ],
        "duration_s": 360,
        "random_seed": 32001,
    }
    site_trace_path = OUT / "e1_site_heavy.json"
    site_trace_path.write_text(
        json.dumps(site_trace, indent=2) + "\n", encoding="utf-8"
    )
    site_rows = []
    for policy in SITE_HEAVY_POLICIES:
        identity = {
            "experiment": site_heavy_experiment,
            "condition": "E1_SITE_HEAVY",
            "policy": policy.value,
        }
        site_rows.append(PlannedRun(
            run_id=deterministic_id("eval_run", identity, length=32),
            question=ExperimentQuestion.RQ3_OPERATING_ADAPTATION,
            experiment_id=site_heavy_experiment,
            baseline_id=policy,
            mode=EvaluationMode.FULL_STACK,
            network_profile_id="good_network",
            network_profile_path=str(ROOT / "netwaggle/configs/profiles/site_evaluation_29node/N0.json"),
            repetition=1,
            random_seed=32001,
            playback_mode="realtime",
            provider_profile_version="compute-contention-v1",
            disturbance_profile_id="E1_SITE_HEAVY",
            condition_trace_id=site_trace["trace_id"],
            condition_trace_path=str(site_trace_path),
            ce_start_offset_seconds=0,
            provider_execution_mode="real",
            vlm_mode="replayed_response",
            campaign_year=2026,
        ))
    site_manifest = OUT / "compute_contention_site_heavy_pilot_2.jsonl"
    site_manifest.write_text(
        "".join(row.model_dump_json() + "\n" for row in site_rows),
        encoding="utf-8",
    )
    (OUT / "compute_contention_site_heavy_pilot_2.summary.json").write_text(
        json.dumps({
            "schema_version": "fable.compute_contention_plan.v1",
            "manifest": str(site_manifest),
            "experiment_id": site_heavy_experiment,
            "replay_duration_seconds": site_heavy_duration,
            "schedule_fraction": {"start": 0.5, "end": 0.95},
            "schedule_seconds": {"start": site_start, "end": site_end},
            "systems": [item.value for item in SITE_HEAVY_POLICIES],
            "planned_cells": len(site_rows),
            "playback_mode": "realtime",
            "gpu_partition": gpu_partition.as_dict(),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
