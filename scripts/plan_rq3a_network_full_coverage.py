#!/usr/bin/env python3
"""Generate the full-coverage RQ3 network matrix.

The discriminating disturbed cases remain exactly those selected by the
updated RQ3 plan.  Four additional nominal controls ensure that the campaign
visits one trace from every authored CE family before any later trace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.catalog import ExperimentCatalog  # noqa: E402
from evaluation.experiments.matrix import PlannedRun  # noqa: E402
from evaluation.experiments.specs import ExperimentQuestion  # noqa: E402
from evaluation.schemas import EvaluationMode  # noqa: E402
from fable.common.ids import deterministic_id  # noqa: E402
from scripts.plan_rq3a_network_updated import (  # noqa: E402
    CASES as UPDATED_CASES,
    MANIFEST_ROOT,
    POLICIES,
    TOPOLOGY,
    condition_trace,
)


ADDITIONAL_NOMINAL_CONTROLS = (
    ("N0_ROUTE_CONVOY", "20241008-route-convoy-18-r030", "N0", None),
    ("N0_TWO_VEHICLE_CHASE", "20241009-two-vehicle-chase-18-r021", "N0", None),
    ("N0_ROBBERY_WITH_ALARM", "20250812-robbery-with-alarm-burglary-a-r012", "N0", None),
    ("N0_TALKING_RENDEZVOUS", "20250812-talking-rendezvous-rendezvous-brianjulian-1-r029", "N0", None),
)
CASES = (*UPDATED_CASES[:5], *ADDITIONAL_NOMINAL_CONTROLS, *UPDATED_CASES[5:])


def main() -> int:
    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv"
    )
    experiments = {item.experiment_id: item for item in catalog.recommended()}
    missing = sorted({case[1] for case in CASES} - set(experiments))
    if missing:
        raise RuntimeError(f"full-coverage RQ3 cases missing from catalog: {missing}")

    rows: list[PlannedRun] = []
    case_rows: list[dict[str, object]] = []
    for case_id, experiment_id, condition, target in CASES:
        experiment = experiments[experiment_id]
        duration = float(experiment.duration_seconds)
        trace = condition_trace(case_id, condition, target, duration)
        trace_path = MANIFEST_ROOT / f"{case_id.lower()}.json"
        trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        case_rows.append(
            {
                "case_id": case_id,
                "experiment_id": experiment_id,
                "ce_variant": experiment.ce_variant,
                "campaign_year": experiment.campaign_year,
                "condition": condition,
                "target": target,
                "condition_trace_path": str(trace_path),
                "replay_duration_seconds": duration,
            }
        )
        for baseline in POLICIES:
            identity = {
                "question": ExperimentQuestion.RQ3_OPERATING_ADAPTATION,
                "case_id": case_id,
                "experiment_id": experiment_id,
                "baseline_id": baseline,
                "random_seed": 31012,
                "matrix": "full_coverage_v2",
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
                    random_seed=31012,
                    playback_mode="realtime",
                    provider_profile_version="rq3a-network-full-coverage-v2",
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

    manifest = MANIFEST_ROOT / "rq3a_network_full_coverage_75.jsonl"
    manifest.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "schema_version": "fable.rq3a_network_full_coverage_plan.v1",
        "manifest": str(manifest),
        "topology": str(TOPOLOGY),
        "systems": [item.value for item in POLICIES],
        "runs_per_system": len(CASES),
        "planned_runs": len(rows),
        "unique_ce_family_controls": 9,
        "disturbed_cases": sum(case[2] != "N0" for case in CASES),
        "repetitions": 1,
        "playback_mode": "realtime",
        "execution_order": "ce-round-robin; condition then baseline within trace",
        "cases": case_rows,
        "preflight_required_for_disturbed_cells": True,
    }
    summary_path = MANIFEST_ROOT / "rq3a_network_full_coverage_75.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
