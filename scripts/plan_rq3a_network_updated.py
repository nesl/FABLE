#!/usr/bin/env python3
"""Generate the updated 55-cell RQ3a network-adaptation experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.catalog import ExperimentCatalog  # noqa: E402
from evaluation.condition_trace import midpoint_disturbance_schedule  # noqa: E402
from evaluation.experiments.matrix import PlannedRun  # noqa: E402
from evaluation.experiments.specs import ExperimentQuestion  # noqa: E402
from evaluation.schemas import BaselineId, EvaluationMode  # noqa: E402
from fable.common.ids import deterministic_id  # noqa: E402


MANIFEST_ROOT = ROOT / "evaluation/manifests/adaptation/rq3a_updated"
TOPOLOGY = ROOT / "netwaggle/configs/site_evaluation_29node.json"
POLICIES = (
    BaselineId.B1_STATIC_WHOLE_EVENT,
    BaselineId.B2_FRONTIER_FIXED_REALIZATION,
    BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
    BaselineId.B4_GREEDY_FRONTIER,
    BaselineId.FABLE,
)
CASES = (
    # Five unique nominal controls.
    ("N0_PASS_FOLLOW", "20260415-pass-follow-clear-convoy-3-car-convoy-11-r011", "N0", None),
    ("N0_CONVERGENCE", "20241008-vehicle-convergence-1-r004", "N0", None),
    ("N0_RENDEZVOUS", "20250812-vehicle-rendezvous-brianjulian-1-r026", "N0", None),
    ("N0_ROBBERY", "20260415-cross-sensor-robbery-robbery-13-r013", "N0", None),
    ("N0_STALKING", "20260414-three-visit-stalking-stalking-30-r030", "N0", None),
    # N1 selected-sensor disturbances.
    ("N1_PASS_FOLLOW_ORIN13", "20260415-pass-follow-clear-convoy-3-car-convoy-11-r011", "N1", "s_orin13"),
    ("N1_CONVERGENCE_MOBILE1", "20241008-vehicle-convergence-1-r004", "N1", "s_mobile_archive_1"),
    ("N1_RENDEZVOUS_MOBILE6", "20250812-vehicle-rendezvous-brianjulian-1-r026", "N1", "s_mobile_archive_6"),
    # N2 shared site-backbone disturbances.
    ("N2_PASS_FOLLOW", "20260415-pass-follow-clear-convoy-3-car-convoy-11-r011", "N2", "site_backbone"),
    ("N2_ROBBERY", "20260415-cross-sensor-robbery-robbery-13-r013", "N2", "site_backbone"),
    ("N2_STALKING", "20260414-three-visit-stalking-stalking-30-r030", "N2", "site_backbone"),
)


def condition_trace(
    case_id: str,
    condition: str,
    target: str | None,
    replay_duration_s: float,
) -> dict:
    transitions = []
    if condition == "N1":
        disturbed_at, restored_at = midpoint_disturbance_schedule(replay_duration_s)
        transitions = [
            {
                "transition_id": f"{case_id.lower()}:degrade",
                "offset_s": disturbed_at,
                "action": "APPLY_NETWORK_PROFILE",
                "target_id": f"sensor_uplink:{target}",
                "profile_id": "L1",
            },
            {
                "transition_id": f"{case_id.lower()}:restore",
                "offset_s": restored_at,
                "action": "RESTORE_NETWORK_PROFILE",
                "target_id": f"sensor_uplink:{target}",
                "profile_id": "N0",
            },
        ]
    elif condition == "N2":
        disturbed_at, restored_at = midpoint_disturbance_schedule(replay_duration_s)
        transitions = [
            {
                "transition_id": f"{case_id.lower()}:degrade",
                "offset_s": disturbed_at,
                "action": "APPLY_NETWORK_PROFILE",
                "target_id": "site_backbone",
                "profile_id": "N2",
            },
            {
                "transition_id": f"{case_id.lower()}:restore",
                "offset_s": restored_at,
                "action": "RESTORE_NETWORK_PROFILE",
                "target_id": "site_backbone",
                "profile_id": "N0",
            },
        ]
    return {
        "schema_version": "fable.condition_trace.v1",
        "trace_id": f"rq3a-updated-{case_id.lower().replace('_', '-')}",
        "initial_network_profile": "N0",
        "initial_compute_profile": "N0",
        "anchor": "TRACE_START",
        "transitions": transitions,
        "duration_s": 360.0,
        "random_seed": 31011,
    }


def main() -> int:
    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv"
    )
    experiments = {item.experiment_id: item for item in catalog.recommended()}
    missing = sorted({case[1] for case in CASES} - set(experiments))
    if missing:
        raise RuntimeError(f"updated RQ3a cases missing from recommended catalog: {missing}")
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    case_rows = []
    for case_id, experiment_id, condition, target in CASES:
        experiment = experiments[experiment_id]
        replay_duration_s = float(experiment.duration_seconds)
        trace = condition_trace(
            case_id, condition, target, replay_duration_s
        )
        trace_path = MANIFEST_ROOT / f"{case_id.lower()}.json"
        trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        schedule = (
            midpoint_disturbance_schedule(replay_duration_s)
            if condition != "N0" else None
        )
        case_rows.append(
            {
                "case_id": case_id,
                "experiment_id": experiment_id,
                "ce_variant": experiment.ce_variant,
                "campaign_year": experiment.campaign_year,
                "condition": condition,
                "target": target,
                "condition_trace_path": str(trace_path),
                "replay_duration_seconds": replay_duration_s,
                "disturbance_start_seconds": schedule[0] if schedule else None,
                "disturbance_end_seconds": schedule[1] if schedule else None,
            }
        )
        for baseline in POLICIES:
            identity = {
                "question": ExperimentQuestion.RQ3_OPERATING_ADAPTATION,
                "case_id": case_id,
                "experiment_id": experiment_id,
                "baseline_id": baseline,
                "random_seed": 31011,
            }
            rows.append(
                PlannedRun(
                    run_id=deterministic_id("eval_run", identity, length=32),
                    question=ExperimentQuestion.RQ3_OPERATING_ADAPTATION,
                    experiment_id=experiment_id,
                    baseline_id=baseline,
                    mode=EvaluationMode.FULL_STACK,
                    network_profile_id="good_network",
                    network_profile_path=str(
                        ROOT / "netwaggle/configs/profiles/site_evaluation_29node/N0.json"
                    ),
                    repetition=1,
                    random_seed=31011,
                    playback_mode="realtime",
                    provider_profile_version="rq3a-network-updated-v1",
                    disturbance_profile_id=condition,
                    condition_trace_id=trace["trace_id"],
                    condition_trace_path=str(trace_path),
                    ce_start_offset_seconds=0.0,
                    provider_execution_mode="real",
                    vlm_mode="replayed_response",
                    warnings=experiment.spatial_notes,
                    campaign_year=experiment.campaign_year,
                )
            )
    manifest = MANIFEST_ROOT / "rq3a_network_updated_55.jsonl"
    manifest.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    pilot_case_ids = {"N0_PASS_FOLLOW", "N1_PASS_FOLLOW_ORIN13"}
    pilot_rows = [
        row
        for row, case in zip(rows, (case for case in CASES for _ in POLICIES))
        if case[0] in pilot_case_ids
    ]
    pilot_manifest = MANIFEST_ROOT / "rq3a_network_updated_pass_follow_pilot_10.jsonl"
    pilot_manifest.write_text(
        "".join(row.model_dump_json() + "\n" for row in pilot_rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "fable.rq3a_network_updated_plan.v1",
        "manifest": str(manifest),
        "topology": str(TOPOLOGY),
        "systems": [item.value for item in POLICIES],
        "runs_per_system": 11,
        "planned_runs": len(rows),
        "pilot_manifest": str(pilot_manifest),
        "pilot_runs": len(pilot_rows),
        "repetitions": 1,
        "playback_mode": "realtime",
        "schedule_anchor": "TRACE_START",
        "schedule": "per-trace fractional window: 50% through 95% of replay",
        "execution_order": "case-major; all five systems adjacent",
        "cases": case_rows,
        "preflight_required_for_disturbed_cells": True,
    }
    (MANIFEST_ROOT / "rq3a_network_updated_55.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
