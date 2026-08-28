#!/usr/bin/env python3
"""Plan all recommended RQ3 network cells not covered by the 9-trace campaign."""

from __future__ import annotations

import json
import sys
from collections import Counter
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
PRIOR_MANIFEST = OUTPUT_ROOT / "rq3_network_single_disconnect_full_90.jsonl"
TOPOLOGY = ROOT / "netwaggle/configs/site_evaluation_29node.json"
NETWORK_PROFILE = ROOT / "netwaggle/configs/profiles/site_evaluation_29node/N0.json"
POLICIES = (
    BaselineId.B1_STATIC_WHOLE_EVENT,
    BaselineId.B2_FRONTIER_FIXED_REALIZATION,
    BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
    BaselineId.B4_GREEDY_FRONTIER,
    BaselineId.FABLE,
)

# Within each CE family, collection sessions repeat the same physical setup.
# These switches come from the validated representative trace used by the
# preceding 90-cell campaign. This mapping is fixed before evaluation.
SWITCH_BY_VARIANT = {
    "Pass-follow-clear convoy": "s_orin13",
    "Vehicle convergence": "s_orin1",
    "Vehicle rendezvous": "s_mob6",
    "Cross-sensor robbery": "s_orin15",
    "Three-visit stalking": "s_orin16",
    "Route convoy": "s_orin4",
    "Two-vehicle chase": "s_orin1",
    "Robbery with alarm": "s_mob4",
    "Talking/rendezvous": "s_mob6",
}


def _semantic_key(run: PlannedRun) -> tuple[str, str, str, int]:
    return (
        run.experiment_id,
        run.disturbance_profile_id,
        run.baseline_id.value,
        run.repetition,
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
    prior_runs = tuple(
        PlannedRun.model_validate_json(line)
        for line in PRIOR_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    prior_keys = {_semantic_key(run) for run in prior_runs}
    prior_experiments = {run.experiment_id for run in prior_runs}

    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv"
    )
    recommended = tuple(catalog.recommended())
    variants = {item.ce_variant for item in recommended}
    if variants != set(SWITCH_BY_VARIANT):
        raise RuntimeError(
            "disconnect-source mapping does not exactly cover catalog variants: "
            f"missing={sorted(variants - set(SWITCH_BY_VARIANT))}, "
            f"extra={sorted(set(SWITCH_BY_VARIANT) - variants)}"
        )

    topology = json.loads(TOPOLOGY.read_text(encoding="utf-8"))
    topology_links = {
        frozenset((str(item["from"]), str(item["to"])))
        for item in topology["links"]
    }
    rows: list[PlannedRun] = []
    case_rows: list[dict[str, object]] = []
    for experiment in recommended:
        # Exclude the complete trace block, including cells that were still
        # pending when this continuation was generated.
        if experiment.experiment_id in prior_experiments:
            continue
        switch_id = SWITCH_BY_VARIANT[experiment.ce_variant]
        if frozenset((switch_id, "s_edge")) not in topology_links:
            raise RuntimeError(f"topology has no link between {switch_id} and s_edge")
        duration = float(experiment.duration_seconds)
        trace = _condition_trace(experiment.experiment_id, switch_id, duration)
        trace_path = OUTPUT_ROOT / f"disconnect_{experiment.experiment_id}.json"
        trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        case_rows.append(
            {
                "experiment_id": experiment.experiment_id,
                "ce_variant": experiment.ce_variant,
                "campaign_year": experiment.campaign_year,
                "duration_seconds": duration,
                "disconnect_switch": switch_id,
                "disconnect_link": f"link:{switch_id}:s_edge",
                "disconnect_start_seconds": round(duration * 0.50, 3),
                "disconnect_end_seconds": round(duration * 0.95, 3),
                "selection_basis": "fixed CE-family mapping from the validated representative trace",
            }
        )
        for condition in ("N0", "SENSOR_DISCONNECT"):
            for baseline in POLICIES:
                identity = {
                    "matrix": "rq3_single_disconnect_full_v1",
                    "experiment_id": experiment.experiment_id,
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
                run = PlannedRun(
                    run_id=deterministic_id("eval_run", identity, length=32),
                    question=ExperimentQuestion.RQ3_OPERATING_ADAPTATION,
                    experiment_id=experiment.experiment_id,
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
                if _semantic_key(run) in prior_keys:
                    raise RuntimeError(f"continuation overlaps prior cell: {_semantic_key(run)}")
                rows.append(run)

    manifest = OUTPUT_ROOT / "rq3_network_single_disconnect_continuation_740.jsonl"
    manifest.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    counts = Counter(item["ce_variant"] for item in case_rows)
    summary = {
        "schema_version": "fable.rq3_network_single_disconnect_continuation.v1",
        "manifest": str(manifest),
        "prior_manifest": str(PRIOR_MANIFEST),
        "exclusion_policy": "exclude every cell belonging to any trace in the prior 90-cell manifest",
        "excluded_traces": sorted(prior_experiments),
        "excluded_trace_count": len(prior_experiments),
        "excluded_cell_count": len(prior_runs),
        "recommended_catalog_traces": len(recommended),
        "continuation_traces": len(case_rows),
        "planned_runs": len(rows),
        "runs_per_trace": 10,
        "conditions": ["N0", "SENSOR_DISCONNECT"],
        "systems": [item.value for item in POLICIES],
        "trace_counts_by_ce_variant": dict(sorted(counts.items())),
        "playback_mode": "realtime",
        "playback_speed": 1.0,
        "disconnect_schedule": "50% through 95% of each labeled trace duration",
        "execution_order": "ce-round-robin",
        "recommended_output_root": "/media/brianw/Extreme SSD2/fable_results/rq3_network_single_disconnect_continuation_20260807",
        "cases": case_rows,
    }
    summary_path = OUTPUT_ROOT / "rq3_network_single_disconnect_continuation_740.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if len(prior_runs) != 90 or len(case_rows) != 74 or len(rows) != 740:
        raise RuntimeError(
            f"unexpected matrix sizes: prior={len(prior_runs)}, "
            f"traces={len(case_rows)}, cells={len(rows)}"
        )
    print(json.dumps({k: summary[k] for k in (
        "recommended_catalog_traces", "excluded_trace_count",
        "excluded_cell_count", "continuation_traces", "planned_runs",
        "trace_counts_by_ce_variant", "manifest",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
