#!/usr/bin/env python3
"""Run corrected RQ3a single- and multi-disturbance stages sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--ready-seconds", type=float, default=30)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stages = (
        ("single_disturbance", [
            sys.executable, str(ROOT / "scripts/run_rq3_campaigns.py"),
            "--root", str(root / "single_disturbance"), "--only", "rq3a",
            "--rq3a-manifest", str(ROOT / "evaluation/manifests/adaptation/rq3a_focused_lease_controlled_matrix.jsonl"),
            "--max-seconds", str(args.max_seconds), "--ready-seconds", str(args.ready_seconds),
        ]),
        ("multi_disturbance", [
            sys.executable, str(ROOT / "scripts/run_rq3a_mixed_workload.py"),
            "--matrix", str(ROOT / "evaluation/manifests/workloads/rq3a_mixed_480s_matrix.jsonl"),
            "--output-dir", str(root / "multi_disturbance"),
        ]),
    )
    events = []
    failures = 0
    for name, command in stages:
        started = datetime.now(UTC).isoformat()
        result = subprocess.run(command, cwd=ROOT, check=False)
        events.append({"stage": name, "returncode": result.returncode,
                       "status": "complete" if result.returncode == 0 else "failed",
                       "started_at": started, "finished_at": datetime.now(UTC).isoformat()})
        failures += int(result.returncode != 0)
    report = {"schema_version": "fable.rq3a_only_unattended.v1",
              "events": events, "failed_stages": failures,
              "finished_at": datetime.now(UTC).isoformat()}
    (root / "campaign-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
