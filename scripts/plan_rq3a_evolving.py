#!/usr/bin/env python3
"""Generate the bounded, trace-major evolving-condition RQ3a matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.catalog import ExperimentCatalog  # noqa: E402
from evaluation.experiments.matrix import (  # noqa: E402
    PlannedRun,
    _event_family,
    build_run_matrix,
)
from evaluation.experiments.specs import ExperimentQuestion  # noqa: E402
from evaluation.schemas import BaselineId  # noqa: E402
from fable.common.ids import deterministic_id  # noqa: E402


TRACE_ROOT = ROOT / "evaluation/manifests/adaptation"
TRACE_INPUTS = (
    ("sensor-uplink", str(TRACE_ROOT / "rq3a_short_sensor_uplink.json")),
    ("wan", str(TRACE_ROOT / "rq3a_short_wan.json")),
    ("compute", str(TRACE_ROOT / "rq3a_short_compute.json")),
    ("provider-failure-robbery", str(TRACE_ROOT / "rq3a_short_provider_failure.json")),
    ("provider-failure-convoy", str(TRACE_ROOT / "rq3a_short_provider_failure_convoy.json")),
)

# Order is part of the experiment contract: dynamically changing evidence
# frontiers run first, followed by controls whose evidence requirements are
# largely fixed after admission.
DYNAMIC_EXPERIMENT_IDS = (
    "20260414-cross-sensor-robbery-robbery-32-r032",
    "20241008-vehicle-convergence-1-r004",
    "20250812-vehicle-rendezvous-brianjulian-2-r027",
    "20260414-three-visit-stalking-stalking-28-r028",
)
NON_DYNAMIC_EXPERIMENT_IDS = (
    "20250812-robbery-with-alarm-burglary-a-r009",
    "20241008-route-convoy-3-r014",
)
REPRESENTATIVE_EXPERIMENT_IDS = DYNAMIC_EXPERIMENT_IDS + NON_DYNAMIC_EXPERIMENT_IDS
FOCUSED_TRACE_INPUTS = (
    ("wan", str(TRACE_ROOT / "rq3a_short_wan.json")),
    ("compute", str(TRACE_ROOT / "rq3a_short_compute.json")),
)
RQ1_PILOT_EXPERIMENT_IDS = (
    "20241008-vehicle-convergence-1-r004",
    "20250812-vehicle-rendezvous-brianjulian-1-r026",
    "20260414-three-visit-stalking-stalking-30-r030",
    "20260415-cross-sensor-robbery-robbery-13-r013",
    "20260415-pass-follow-clear-convoy-3-car-convoy-11-r011",
    "20241008-route-convoy-18-r030",
    "20241009-two-vehicle-chase-18-r021",
    "20250812-robbery-with-alarm-burglary-a-r012",
    "20250812-talking-rendezvous-rendezvous-brianjulian-1-r029",
)
RQ1_PILOT_DYNAMIC_EXPERIMENT_IDS = RQ1_PILOT_EXPERIMENT_IDS[:5]
RQ1_PAIRED_TRACE_INPUTS = (
    ("n0-control", str(TRACE_ROOT / "rq3a_rq1_paired_n0.json")),
    ("wan", str(TRACE_ROOT / "rq3a_short_wan.json")),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=31000)
    parser.add_argument(
        "--focused",
        action="store_true",
        help="Generate only live-validated WAN/uplink cells with active-demand overlap.",
    )
    parser.add_argument(
        "--rq1-paired-pilot",
        action="store_true",
        help=(
            "Generate a trace-major N0/WAN pilot over the matching 2026 RQ1 "
            "dynamic-evidence examples."
        ),
    )
    parser.add_argument(
        "--include-b1",
        action="store_true",
        help=(
            "Add B1 static whole-event placement as a non-adaptive reference. "
            "The primary RQ3a systems remain B2, B3, and FABLE."
        ),
    )
    parser.add_argument(
        "--include-b4",
        action="store_true",
        help=(
            "Add B4 greedy frontier planning as an adaptation ablation. It "
            "receives the same condition traces as B2, B3, and FABLE."
        ),
    )
    args = parser.parse_args()
    if args.focused and args.rq1_paired_pilot:
        parser.error("--focused and --rq1-paired-pilot are mutually exclusive")
    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv"
    )
    experiments = {item.experiment_id: item for item in catalog.recommended()}
    missing = sorted(set(REPRESENTATIVE_EXPERIMENT_IDS) - set(experiments))
    if missing:
        raise RuntimeError(f"RQ3a representative experiments missing: {missing}")
    selected_experiment_ids = (
        RQ1_PILOT_EXPERIMENT_IDS
        if args.rq1_paired_pilot
        else REPRESENTATIVE_EXPERIMENT_IDS
    )
    representative_catalog = ExperimentCatalog(
        experiments[experiment_id]
        for experiment_id in selected_experiment_ids
    )
    trace_inputs = TRACE_INPUTS
    offsets = (0.0, 20.0, 45.0)
    if args.focused:
        trace_inputs = FOCUSED_TRACE_INPUTS
        # Focused traces are admission-relative; replay offsets are not needed
        # to manufacture overlap with an active request.
        offsets = (0.0,)
    elif args.rq1_paired_pilot:
        trace_inputs = RQ1_PAIRED_TRACE_INPUTS
        offsets = (0.0,)
    candidates = build_run_matrix(
        representative_catalog,
        ExperimentQuestion.RQ3_OPERATING_ADAPTATION,
        repetitions=1,
        seed=args.seed,
        max_traces=len(selected_experiment_ids),
        condition_traces=trace_inputs,
        ce_start_offsets=offsets,
        provider_execution_mode="real",
        vlm_mode="replayed_response",
    )
    selected = list(candidates)
    extra_baselines = []
    if args.include_b1:
        extra_baselines.append(BaselineId.B1_STATIC_WHOLE_EVENT)
    if args.include_b4:
        extra_baselines.append(BaselineId.B4_GREEDY_FRONTIER)
    if extra_baselines:
        # RQ3a's registered systems intentionally contain only B2, B3, and
        # FABLE. Clone each B2 treatment so optional references receive the
        # identical trace, seed, offset, and provider execution mode without
        # changing the canonical RQ3a experiment specification.
        extra_rows = []
        for row in candidates:
            if row.baseline_id != BaselineId.B2_FRONTIER_FIXED_REALIZATION:
                continue
            for baseline in extra_baselines:
                payload = row.model_dump(mode="json")
                payload["baseline_id"] = baseline.value
                payload["run_id"] = deterministic_id(
                    "eval_run",
                    {"source_run_id": row.run_id, "baseline_id": baseline.value},
                    length=32,
                )
                extra_rows.append(PlannedRun.model_validate(payload))
        selected.extend(extra_rows)
    trace_rank = {value: index for index, value in enumerate(selected_experiment_ids)}
    condition_rank = {value[0]: index for index, value in enumerate(trace_inputs)}
    policy_rank = {
        "B1_STATIC_WHOLE_EVENT": 0,
        "B2_FRONTIER_FIXED_REALIZATION": 1,
        "B3_TASK_RESOURCE_ADAPTIVE": 2,
        "B4_GREEDY_FRONTIER": 3,
        "FABLE": 4,
    }
    selected.sort(
        key=lambda row: (
            trace_rank[row.experiment_id],
            condition_rank.get(row.condition_trace_id, 999),
            policy_rank[row.baseline_id.value],
            row.repetition,
        )
    )
    expected_runs = 54 if args.rq1_paired_pilot else (36 if args.focused else 270)
    expected_runs = expected_runs * (3 + len(extra_baselines)) // 3
    if len(selected) != expected_runs:
        raise RuntimeError(
            f"expected {expected_runs} evolving-condition runs, generated {len(selected)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(item.model_dump_json() + "\n" for item in selected),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "fable.rq3a_evolving_plan.v1",
        "planned_runs": len(selected),
        "systems": sorted({item.baseline_id.value for item in selected}),
        "experiments": list(selected_experiment_ids),
        "dynamic_experiments": list(
            RQ1_PILOT_DYNAMIC_EXPERIMENT_IDS
            if args.rq1_paired_pilot
            else DYNAMIC_EXPERIMENT_IDS
        ),
        "non_dynamic_control_experiments": list(
            RQ1_PILOT_EXPERIMENT_IDS[len(RQ1_PILOT_DYNAMIC_EXPERIMENT_IDS):]
            if args.rq1_paired_pilot
            else NON_DYNAMIC_EXPERIMENT_IDS
        ),
        "condition_traces": sorted(
            {item.condition_trace_id for item in selected if item.condition_trace_id}
        ),
        "ce_start_offsets_seconds": sorted(
            {item.ce_start_offset_seconds for item in selected}
        ),
        "random_seed": args.seed,
        "focused_active_demand": args.focused,
        "rq1_paired_pilot": args.rq1_paired_pilot,
        "b1_static_reference_included": args.include_b1,
        "b4_greedy_ablation_included": args.include_b4,
        "attribution_rule": (
            "A disturbed failure is network-attributable only when the matching "
            "N0 cell for experiment, baseline, seed, playback mode, and provider "
            "configuration is TRUE_POSITIVE."
            if args.rq1_paired_pilot
            else None
        ),
        "execution_order": (
            "trace-major; dynamic first; "
            + ", ".join(
                baseline
                for baseline in ("B1", "B2", "B3", "B4", "FABLE")
                if baseline not in {"B1" if not args.include_b1 else "", "B4" if not args.include_b4 else ""}
            )
            + " per condition"
        ),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
