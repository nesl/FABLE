#!/usr/bin/env python3
"""Rerun only failed cells from the 2026-08-05 RQ1 baseline pilot."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FAILED_CELLS = {
    "B1_STATIC_WHOLE_EVENT": (
        "20241008-vehicle-convergence-1-r004",
        "20250812-talking-rendezvous-rendezvous-brianjulian-1-r029",
        "20250812-vehicle-rendezvous-brianjulian-1-r026",
        "20260414-three-visit-stalking-stalking-30-r030",
        "20260415-cross-sensor-robbery-robbery-13-r013",
    ),
    "B2_FRONTIER_FIXED_REALIZATION": (
        "20241008-vehicle-convergence-1-r004",
        "20250812-talking-rendezvous-rendezvous-brianjulian-1-r029",
        "20250812-vehicle-rendezvous-brianjulian-1-r026",
    ),
    "B3_TASK_RESOURCE_ADAPTIVE": (
        "20241008-vehicle-convergence-1-r004",
        "20250812-talking-rendezvous-rendezvous-brianjulian-1-r029",
        "20260414-three-visit-stalking-stalking-30-r030",
        "20260415-cross-sensor-robbery-robbery-13-r013",
    ),
}


def main() -> int:
    output_root = (
        ROOT / "evaluation/results/rq1_failed_baseline_repairs_20260805_v2"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for baseline, experiment_ids in FAILED_CELLS.items():
        output_dir = output_root / baseline / "repetition-01"
        command = [
            sys.executable,
            str(ROOT / "scripts/run_full_ce_suite.py"),
            "--output-dir",
            str(output_dir),
            "--baseline",
            baseline,
            "--max-seconds",
            "300",
            "--ready-seconds",
            "30",
            "--playback-mode",
            "realtime",
            "--mobile-root",
            "/media/brianw/Extreme SSD3",
        ]
        for experiment_id in experiment_ids:
            command.extend(("--experiment-id", experiment_id))
        print(
            f"[baseline-repair] starting {baseline} ({len(experiment_ids)} cells)",
            flush=True,
        )
        completed = subprocess.run(command, cwd=ROOT, check=False)
        results.append(
            {
                "baseline_id": baseline,
                "experiment_ids": list(experiment_ids),
                "returncode": completed.returncode,
                "output_dir": str(output_dir),
            }
        )
        print(
            f"[baseline-repair] finished {baseline}: {completed.returncode}",
            flush=True,
        )
    report = {
        "schema_version": "fable.failed_baseline_repair_campaign.v1",
        "finished_at": datetime.now(UTC).isoformat(),
        "playback_mode": "realtime",
        "playback_speed": 1.0,
        "results": results,
    }
    (output_root / "campaign-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 1 if any(item["returncode"] for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
