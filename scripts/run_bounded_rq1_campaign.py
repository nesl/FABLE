#!/usr/bin/env python3
"""Run the bounded, resumable broad RQ1 campaign."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.catalog import ExperimentCatalog
from evaluation.experiments.matrix import build_run_matrix, write_planned_runs
from evaluation.experiments.specs import ExperimentQuestion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--ready-seconds", type=float, default=30)
    parser.add_argument(
        "--mobile-root",
        type=Path,
        default=Path("/media/brianw/Extreme SSD3"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv",
        transition_model_path=(
            ROOT / "evaluation/labels/site_sensor_transition_model_2024_2025.json"
        ),
    )
    runs = build_run_matrix(
        catalog,
        ExperimentQuestion.RQ1_END_TO_END,
        repetitions=1,
        seed=args.seed,
        playback_mode="realtime",
    )
    manifest = write_planned_runs(runs, output_dir / "run_matrix.jsonl")
    grouped: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        grouped[run.baseline_id.value].add(run.experiment_id)
    campaign = {
        "schema_version": "fable.bounded_rq1_campaign.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "manifest": str(manifest),
        "planned_runs": len(runs),
        "unique_traces": len({run.experiment_id for run in runs}),
        "baselines": {
            baseline: len(experiment_ids)
            for baseline, experiment_ids in sorted(grouped.items())
        },
    }
    (output_dir / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    failures = 0
    for baseline, experiment_ids in sorted(grouped.items()):
        baseline_dir = output_dir / baseline / "repetition-01"
        command = [
            sys.executable,
            str(ROOT / "scripts/run_full_ce_suite.py"),
            "--output-dir",
            str(baseline_dir),
            "--baseline",
            baseline,
            "--max-seconds",
            str(args.max_seconds),
            "--ready-seconds",
            str(args.ready_seconds),
            "--playback-mode",
            "realtime",
            "--mobile-root",
            str(args.mobile_root),
        ]
        for experiment_id in sorted(experiment_ids):
            command.extend(("--experiment-id", experiment_id))
        print(
            f"[campaign] starting baseline={baseline} traces={len(experiment_ids)}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=ROOT, check=False)
        print(
            f"[campaign] finished baseline={baseline} returncode={completed.returncode}",
            flush=True,
        )
        if completed.returncode:
            failures += 1
    summary = {
        **campaign,
        "finished_at": datetime.now(UTC).isoformat(),
        "failed_baseline_suites": failures,
    }
    (output_dir / "campaign-report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
