#!/usr/bin/env python3
"""Consolidate a completed RQ1 campaign with explicit corrected-result overlays."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


_IGNORED_RESULT_NAMES = {"campaign.json", "campaign-report.json", "plan.json", "report.json"}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_files(root: Path):
    for path in sorted(root.rglob("*.json")):
        if path.name in _IGNORED_RESULT_NAMES:
            continue
        if "-playable.json" in path.name or ".invalid-readiness.json" in path.name:
            continue
        yield path


def _read_results(root: Path, *, default_baseline: str | None = None):
    results = {}
    for path in _result_files(root):
        try:
            item = _load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        experiment_id = item.get("experiment_id")
        if not experiment_id or not item.get("classification"):
            continue
        baseline = str(item.get("baseline") or default_baseline or "")
        if not baseline:
            continue
        results[(baseline, str(experiment_id))] = (item, path.resolve())
    return results


def _write_csv(path: Path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    campaign_dir = args.campaign_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign_dir / "run_matrix.jsonl"
    planned = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]

    original = {}
    for baseline_dir in sorted(path for path in campaign_dir.iterdir() if path.is_dir()):
        original.update(_read_results(baseline_dir, default_baseline=baseline_dir.name))

    overlays = {}
    correction_rows = []
    for overlay_root in args.overlay:
        resolved = overlay_root.resolve()
        for key, value in _read_results(resolved, default_baseline="FABLE").items():
            previous = original.get(key)
            overlays[key] = value
            correction_rows.append(
                {
                    "baseline_id": key[0],
                    "experiment_id": key[1],
                    "original_classification": previous[0].get("classification") if previous else "COVERAGE_GAP",
                    "corrected_classification": value[0].get("classification"),
                    "overlay_source": str(value[1]),
                }
            )

    effective = dict(original)
    effective.update(overlays)
    outcome_rows = []
    counts = defaultdict(Counter)
    by_variant = defaultdict(Counter)
    by_year = defaultdict(Counter)
    for run in sorted(planned, key=lambda item: (item["baseline_id"], item["experiment_id"])):
        key = (str(run["baseline_id"]), str(run["experiment_id"]))
        value = effective.get(key)
        item, source = value if value else ({}, None)
        classification = str(item.get("classification") or "COVERAGE_GAP")
        variant = str(item.get("variant") or "")
        year = item.get("campaign_year") or run.get("campaign_year")
        counts[key[0]][classification] += 1
        by_variant[(key[0], variant)][classification] += 1
        by_year[(key[0], str(year))][classification] += 1
        outcome_rows.append(
            {
                "baseline_id": key[0],
                "experiment_id": key[1],
                "campaign_year": year,
                "variant": variant,
                "classification": classification,
                "corrected_overlay": key in overlays,
                "result_source": str(source) if source else "",
            }
        )

    summary_rows = []
    for baseline, baseline_counts in sorted(counts.items()):
        total = sum(baseline_counts.values())
        executed = total - baseline_counts["COVERAGE_GAP"]
        tp = baseline_counts["TRUE_POSITIVE"]
        summary_rows.append(
            {
                "baseline_id": baseline,
                "planned": total,
                "executed": executed,
                "true_positive": tp,
                "false_negative": baseline_counts["FALSE_NEGATIVE"],
                "runtime_failure": baseline_counts["RUNTIME_FAILURE"],
                "coverage_gap": baseline_counts["COVERAGE_GAP"],
                "true_positive_rate_executed": round(tp / executed, 6) if executed else 0.0,
            }
        )

    _write_csv(output_dir / "summary.csv", summary_rows, tuple(summary_rows[0]))
    _write_csv(output_dir / "outcomes.csv", outcome_rows, tuple(outcome_rows[0]))
    _write_csv(
        output_dir / "corrections.csv",
        sorted(correction_rows, key=lambda row: (row["baseline_id"], row["experiment_id"])),
        ("baseline_id", "experiment_id", "original_classification", "corrected_classification", "overlay_source"),
    )
    document = {
        "schema_version": "fable.rq1_consolidated_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_dir": str(campaign_dir),
        "manifest": str(manifest_path.resolve()),
        "overlay_roots": [str(path.resolve()) for path in args.overlay],
        "planned_runs": len(planned),
        "correction_count": len(correction_rows),
        "summary": summary_rows,
        "by_variant": {
            f"{baseline}|{variant}": dict(value)
            for (baseline, variant), value in sorted(by_variant.items())
        },
        "by_year": {
            f"{baseline}|{year}": dict(value)
            for (baseline, year), value in sorted(by_year.items())
        },
        "summary_csv": str((output_dir / "summary.csv").resolve()),
        "outcomes_csv": str((output_dir / "outcomes.csv").resolve()),
        "corrections_csv": str((output_dir / "corrections.csv").resolve()),
    }
    (output_dir / "report.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
