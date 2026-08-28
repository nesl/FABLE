#!/usr/bin/env python3
"""Generate the uniform nominal/single-sensor-disconnect RQ3 network matrix."""

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
from evaluation.schemas import BaselineId, EvaluationMode  # noqa: E402
from fable.common.ids import deterministic_id  # noqa: E402


OUTPUT_ROOT = ROOT / "evaluation/manifests/adaptation/rq3a_single_disconnect_full"
TOPOLOGY = ROOT / "netwaggle/configs/site_evaluation_29node.json"
NETWORK_PROFILE = (
    ROOT / "netwaggle/configs/profiles/site_evaluation_29node/N0.json"
)
POLICIES = (
    BaselineId.B1_STATIC_WHOLE_EVENT,
    BaselineId.B2_FRONTIER_FIXED_REALIZATION,
    BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
    BaselineId.B4_GREEDY_FRONTIER,
    BaselineId.FABLE,
)

# One previously validated representative trace per authored CE family.  The
# disconnected sensor is the dominant TRUE-observation source in a successful
# FABLE execution of that trace; this is provenance, not a runtime oracle.
CASES = (
    ("20260415-pass-follow-clear-convoy-3-car-convoy-11-r011", "s_orin13"),
    ("20241008-vehicle-convergence-1-r004", "s_orin1"),
    ("20250812-vehicle-rendezvous-brianjulian-1-r026", "s_mob6"),
    ("20260415-cross-sensor-robbery-robbery-13-r013", "s_orin15"),
    ("20260414-three-visit-stalking-stalking-30-r030", "s_orin16"),
    ("20241008-route-convoy-18-r030", "s_orin4"),
    ("20241009-two-vehicle-chase-18-r021", "s_orin1"),
    ("20250812-robbery-with-alarm-burglary-a-r012", "s_mob4"),
    ("20250812-talking-rendezvous-rendezvous-brianjulian-1-r029", "s_mob6"),
)


def _condition_trace(experiment_id: str, switch_id: str, duration: float) -> dict:
    start = round(duration * 0.50, 3)
    end = round(duration * 0.95, 3)
    slug = experiment_id.replace("_", "-")
    return {
        "schema_version": "fable.condition_trace.v1",
        "trace_id": f"rq3-network-single-disconnect-{slug}",
        "initial_network_profile": "N0",
        "initial_compute_profile": "N0",
        "anchor": "TRACE_START",
        "transitions": [
            {
                "transition_id": f"{slug}:disconnect",
                "offset_s": start,
                "action": "FAIL_LINK",
                "target_id": f"link:{switch_id}:s_edge",
            },
            {
                "transition_id": f"{slug}:restore",
                "offset_s": end,
                "action": "RESTORE_LINK",
                "target_id": f"link:{switch_id}:s_edge",
            },
        ],
        "duration_s": max(duration + 60.0, end + 1.0),
        "random_seed": 31032,
    }


def main() -> int:
    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv"
    )
    experiments = {item.experiment_id: item for item in catalog.recommended()}
    topology = json.loads(TOPOLOGY.read_text(encoding="utf-8"))
    topology_links = {
        frozenset((str(item["from"]), str(item["to"])))
        for item in topology["links"]
    }
    missing = sorted(set(dict(CASES)) - set(experiments))
    if missing:
        raise RuntimeError(f"representative experiments missing from catalog: {missing}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[PlannedRun] = []
    case_rows: list[dict[str, object]] = []
    for experiment_id, switch_id in CASES:
        experiment = experiments[experiment_id]
        duration = float(experiment.duration_seconds)
        if frozenset((switch_id, "s_edge")) not in topology_links:
            raise RuntimeError(f"topology has no link between {switch_id} and s_edge")
        trace = _condition_trace(experiment_id, switch_id, duration)
        trace_path = OUTPUT_ROOT / f"disconnect_{experiment_id}.json"
        trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        case_rows.append(
            {
                "experiment_id": experiment_id,
                "ce_variant": experiment.ce_variant,
                "campaign_year": experiment.campaign_year,
                "duration_seconds": duration,
                "disconnect_switch": switch_id,
                "disconnect_link": f"link:{switch_id}:s_edge",
                "disconnect_start_seconds": round(duration * 0.50, 3),
                "disconnect_end_seconds": round(duration * 0.95, 3),
                "selection_basis": "dominant TRUE-observation source in a prior successful FABLE run",
            }
        )
        for condition in ("N0", "SENSOR_DISCONNECT"):
            for baseline in POLICIES:
                identity = {
                    "matrix": "rq3_single_disconnect_full_v1",
                    "experiment_id": experiment_id,
                    "condition": condition,
                    "baseline": baseline.value,
                    "seed": 31032,
                }
                kwargs: dict[str, object] = {}
                if condition == "SENSOR_DISCONNECT":
                    kwargs.update(
                        condition_trace_id=trace["trace_id"],
                        condition_trace_path=str(trace_path),
                    )
                rows.append(
                    PlannedRun(
                        run_id=deterministic_id("eval_run", identity, length=32),
                        question=ExperimentQuestion.RQ3_OPERATING_ADAPTATION,
                        experiment_id=experiment_id,
                        baseline_id=baseline,
                        mode=EvaluationMode.FULL_STACK,
                        network_profile_id="good_network",
                        network_profile_path=str(NETWORK_PROFILE),
                        spatial_metrics_enabled=False,
                        repetition=1,
                        random_seed=31032,
                        playback_mode="realtime",
                        provider_profile_version="rq3-network-single-disconnect-v1",
                        disturbance_profile_id=condition,
                        ce_start_offset_seconds=0.0,
                        provider_execution_mode="real",
                        vlm_mode="replayed_response",
                        warnings=experiment.spatial_notes,
                        campaign_year=experiment.campaign_year,
                        **kwargs,
                    )
                )

    manifest = OUTPUT_ROOT / "rq3_network_single_disconnect_full_90.jsonl"
    manifest.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "schema_version": "fable.rq3_network_single_disconnect_plan.v1",
        "manifest": str(manifest),
        "topology": str(TOPOLOGY),
        "planned_runs": len(rows),
        "ce_families": len(CASES),
        "conditions": ["N0", "SENSOR_DISCONNECT"],
        "systems": [item.value for item in POLICIES],
        "runs_per_trace": 10,
        "repetitions": 1,
        "playback_mode": "realtime",
        "playback_speed": 1.0,
        "event_match_tolerance_seconds": 30.0,
        "disconnect_schedule": "50% through 95% of each labeled trace duration",
        "execution_order": "ce-round-robin; all conditions and baselines adjacent per trace",
        "offline_evidence_policy": "drop; no broker store-and-forward",
        "execution_timeline": "execution_changes.jsonl and execution_changes.csv per run",
        "cases": case_rows,
    }
    summary_path = OUTPUT_ROOT / "rq3_network_single_disconnect_full_90.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
