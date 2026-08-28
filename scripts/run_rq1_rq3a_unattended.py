#!/usr/bin/env python3
"""Resumable sequential RQ1 -> RQ3a-short -> RQ3a-mixed campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completed(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--ready-seconds", type=float, default=30)
    parser.add_argument("--continue-after-failure", action="store_true", default=True)
    parser.add_argument(
        "--mixed-executor",
        type=Path,
        default=ROOT / "scripts/run_rq3a_mixed_workload.py",
        help="Persistent-clock executor for rq3a_mixed_480s_matrix.jsonl.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rq1_manifest = ROOT / "evaluation/manifests/workloads/rq1_lease_controlled_45.jsonl"
    short_manifest = ROOT / "evaluation/manifests/adaptation/rq3a_focused_lease_controlled_matrix.jsonl"
    mixed_manifest = ROOT / "evaluation/manifests/workloads/rq3a_mixed_480s.json"
    for path in (PYTHON, rq1_manifest, short_manifest, mixed_manifest):
        if not path.exists():
            parser.error(f"required path is missing: {path}")
    validation = subprocess.run(
        [str(PYTHON), str(ROOT / "scripts/validate_mixed_workload.py"), str(mixed_manifest)],
        cwd=ROOT, check=False, text=True, capture_output=True,
    )
    preflight = {
        "schema_version": "fable.rq1_rq3a_preflight.v1",
        "validated_at": datetime.now(UTC).isoformat(),
        "rq1_rows": sum(bool(line.strip()) for line in rq1_manifest.read_text().splitlines()),
        "rq3a_single_rows": sum(bool(line.strip()) for line in short_manifest.read_text().splitlines()),
        "mixed_workload_validation": validation.stdout.strip(),
        "mixed_executor": str(args.mixed_executor.resolve()) if args.mixed_executor else None,
        "mixed_execution_ready": bool(args.mixed_executor and args.mixed_executor.is_file()),
    }
    write_json(root / "preflight.json", preflight)
    if validation.returncode:
        print(validation.stderr, file=sys.stderr)
        return validation.returncode
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    stages = [
        (
            "rq1",
            [str(PYTHON), str(ROOT / "scripts/run_planned_ce_campaign.py"),
             "--manifest", str(rq1_manifest), "--output-dir", str(root / "rq1"),
             "--max-seconds", str(args.max_seconds), "--ready-seconds", str(args.ready_seconds)],
        ),
        (
            "rq3a_single_disturbance",
            [str(PYTHON), str(ROOT / "scripts/run_rq3_campaigns.py"),
             "--root", str(root / "rq3a_single_disturbance"), "--only", "rq3a",
             "--rq3a-manifest", str(short_manifest), "--max-seconds", str(args.max_seconds),
             "--ready-seconds", str(args.ready_seconds)],
        ),
    ]
    if args.mixed_executor and args.mixed_executor.is_file():
        stages.append((
            "rq3a_multi_disturbance",
            [str(PYTHON), str(args.mixed_executor.resolve()),
             "--matrix", str(ROOT / "evaluation/manifests/workloads/rq3a_mixed_480s_matrix.jsonl"),
             "--output-dir", str(root / "rq3a_multi_disturbance")],
        ))
    events = []
    failures = 0
    for name, command in stages:
        checkpoint = root / "checkpoints" / f"{name}.json"
        if completed(checkpoint):
            continue
        started = datetime.now(UTC).isoformat()
        result = subprocess.run(command, cwd=ROOT, check=False)
        row = {"stage": name, "status": "complete" if result.returncode == 0 else "failed",
               "returncode": result.returncode, "started_at": started,
               "finished_at": datetime.now(UTC).isoformat(), "command": command}
        write_json(checkpoint, row)
        events.append(row)
        failures += int(result.returncode != 0)
        if result.returncode and not args.continue_after_failure:
            break
    if not preflight["mixed_execution_ready"]:
        events.append({
            "stage": "rq3a_multi_disturbance",
            "status": "blocked",
            "reason": "persistent-stack multi-request executor is not yet supplied",
        })
        failures += 1
    report = {"schema_version": "fable.rq1_rq3a_unattended.v1",
              "finished_at": datetime.now(UTC).isoformat(), "events": events,
              "failed_or_blocked_stages": failures}
    write_json(root / "campaign-report.json", report)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
