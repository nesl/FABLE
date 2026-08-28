#!/usr/bin/env python3
"""Generate the RQ1 baseline pilot matching the paired RQ3a traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_IDS = {
    "20241008-route-convoy-18-r030",
    "20241008-vehicle-convergence-1-r004",
    "20241009-two-vehicle-chase-18-r021",
    "20250812-robbery-with-alarm-burglary-a-r012",
    "20250812-talking-rendezvous-rendezvous-brianjulian-1-r029",
    "20250812-vehicle-rendezvous-brianjulian-1-r026",
    "20260414-three-visit-stalking-stalking-30-r030",
    "20260415-cross-sensor-robbery-robbery-13-r013",
    "20260415-pass-follow-clear-convoy-3-car-convoy-11-r011",
}

LEGACY_TRACE_REPLACEMENTS = {
    "20250812-talking-rendezvous-rendezvous-brianjulian-3-r031":
        "20250812-talking-rendezvous-rendezvous-brianjulian-1-r029",
}


def replace_invalid_trace(row: dict) -> dict:
    replacement = LEGACY_TRACE_REPLACEMENTS.get(row["experiment_id"])
    if replacement is None:
        return row
    updated = dict(row)
    updated["experiment_id"] = replacement
    identity = dict(updated)
    identity.pop("run_id", None)
    updated["run_id"] = "eval_run_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:32]
    return updated


def with_baseline(row: dict, baseline_id: str) -> dict:
    """Clone a trace treatment while deriving an immutable baseline run ID."""

    updated = dict(row)
    updated["baseline_id"] = baseline_id
    identity = dict(updated)
    identity.pop("run_id", None)
    updated["run_id"] = "eval_run_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:32]
    return updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "evaluation/manifests/workloads/rq1_lease_controlled_45.jsonl",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help=(
            "Baseline to include (repeatable). Defaults to the authored "
            "B0/B1/FABLE matching pilot."
        ),
    )
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [replace_invalid_trace(row) for row in rows]
    selected = [row for row in rows if row["experiment_id"] in EXPERIMENT_IDS]
    expected_baselines = set(args.baseline) or {
        "B0_PRODUCE_ALL",
        "B1_STATIC_WHOLE_EVENT",
        "B2_FRONTIER_FIXED_REALIZATION",
        "B3_TASK_RESOURCE_ADAPTIVE",
        "B4_GREEDY_FRONTIER",
        "FABLE",
    }
    # Older RQ1 manifests predate B4. All treatment-independent fields are
    # trace properties, so derive a missing policy cell from that trace's
    # FABLE row rather than dropping the requested baseline from the pilot.
    by_experiment = {}
    for row in selected:
        by_experiment.setdefault(row["experiment_id"], []).append(row)
    expanded = []
    for experiment_id, experiment_rows in by_experiment.items():
        by_baseline = {row["baseline_id"]: row for row in experiment_rows}
        exemplar = by_baseline.get("FABLE") or experiment_rows[0]
        for baseline_id in expected_baselines:
            expanded.append(
                by_baseline.get(baseline_id)
                or with_baseline(exemplar, baseline_id)
            )
    selected = expanded
    expected_rows = len(EXPERIMENT_IDS) * len(expected_baselines)
    if len(selected) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} matching RQ1 rows, found {len(selected)}"
        )
    for experiment_id in EXPERIMENT_IDS:
        actual = {
            row["baseline_id"]
            for row in selected
            if row["experiment_id"] == experiment_id
        }
        if actual != expected_baselines:
            raise RuntimeError(
                f"RQ1 baseline coverage mismatch for {experiment_id}: {sorted(actual)}"
            )
    selected.sort(
        key=lambda row: (
            row["experiment_id"], row["repetition"], row["baseline_id"]
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "fable.rq1_matching_pilot.v1",
        "planned_runs": len(selected),
        "experiments": sorted(EXPERIMENT_IDS),
        "baselines": sorted(expected_baselines),
        "repetitions": [1],
        "playback_modes": sorted({row["playback_mode"] for row in selected}),
        "source_manifest": str(args.source.resolve()),
        "execution_order": "trace-major; all baselines adjacent for each trace",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
