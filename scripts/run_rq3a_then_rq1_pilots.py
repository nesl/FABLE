#!/usr/bin/env python3
"""Run the paired RQ3a pilot and matching RQ1 pilot unattended."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stage_complete(path: Path) -> bool:
    try:
        return load_json(path).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_bounded(command: list[str], timeout_s: float) -> tuple[int, bool]:
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        return process.wait(timeout=timeout_s), False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        return 124, True


def restore_n0(epoch: int) -> dict:
    command = [
        str(PYTHON),
        str(ROOT / "netwaggle/scripts/fable_netwaggle_helper.py"),
        "--kind", "NETWORK_PROFILE",
        "--target", "site_to_cloud",
        "--condition", "N0",
        "--action", "RESTORE",
        "--condition-epoch", str(epoch),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, check=False, text=True, capture_output=True, timeout=75
    )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError:
        response = {"validated": False, "reason": completed.stdout[-1000:]}
    return {
        "returncode": completed.returncode,
        "validated": bool(response.get("validated")),
        "response": response,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=360)
    parser.add_argument("--ready-seconds", type=float, default=60)
    parser.add_argument("--stage-timeout-seconds", type=float, default=21600)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rq3a_manifest = ROOT / "evaluation/manifests/adaptation/rq3a_rq1_paired_pilot.jsonl"
    rq1_manifest = ROOT / "evaluation/manifests/workloads/rq1_matching_rq3a_pilot.jsonl"
    required = (PYTHON, rq3a_manifest, rq1_manifest)
    missing = [str(path) for path in required if not path.is_file()]
    preflight = {
        "schema_version": "fable.rq3a_then_rq1_preflight.v1",
        "checked_at": datetime.now(UTC).isoformat(),
        "missing_paths": missing,
        "rq3a_rows": sum(bool(x.strip()) for x in rq3a_manifest.read_text().splitlines()) if not missing else None,
        "rq1_rows": sum(bool(x.strip()) for x in rq1_manifest.read_text().splitlines()) if not missing else None,
        "execution_order": ["rq3a_paired", "restore_n0", "rq3a_attribution", "rq1_matching"],
        "unattended": True,
    }
    write_json(root / "preflight.json", preflight)
    if missing:
        return 2
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    events = []
    rq3_checkpoint = root / "checkpoints/rq3a_paired.json"
    if not stage_complete(rq3_checkpoint):
        command = [
            str(PYTHON), str(ROOT / "scripts/run_rq3_campaigns.py"),
            "--root", str(root / "rq3a"), "--only", "rq3a",
            "--rq3a-manifest", str(rq3a_manifest),
            "--max-seconds", str(args.max_seconds),
            "--ready-seconds", str(args.ready_seconds),
        ]
        started = datetime.now(UTC).isoformat()
        returncode, timed_out = run_bounded(command, args.stage_timeout_seconds)
        row = {
            "stage": "rq3a_paired", "status": "complete" if returncode == 0 else "failed",
            "returncode": returncode, "timed_out": timed_out, "command": command,
            "started_at": started, "finished_at": datetime.now(UTC).isoformat(),
        }
        write_json(rq3_checkpoint, row)
        events.append(row)

    restoration = restore_n0(900000)
    write_json(root / "n0-before-rq1.json", restoration)
    attribution_command = [
        str(PYTHON), str(ROOT / "scripts/report_rq3a_paired_pilot.py"),
        "--manifest", str(rq3a_manifest),
        "--results-root", str(root / "rq3a"),
        "--output-dir", str(root / "rq3a-analysis"),
    ]
    attribution_rc, attribution_timeout = run_bounded(attribution_command, 300)
    events.append({
        "stage": "rq3a_attribution", "status": "complete" if attribution_rc == 0 else "failed",
        "returncode": attribution_rc, "timed_out": attribution_timeout,
        "finished_at": datetime.now(UTC).isoformat(),
    })

    rq1_checkpoint = root / "checkpoints/rq1_matching.json"
    if restoration["returncode"] == 0 and restoration["validated"] and not stage_complete(rq1_checkpoint):
        command = [
            str(PYTHON), str(ROOT / "scripts/run_planned_ce_campaign.py"),
            "--manifest", str(rq1_manifest),
            "--output-dir", str(root / "rq1"),
            "--max-seconds", str(args.max_seconds),
            "--ready-seconds", str(args.ready_seconds),
        ]
        started = datetime.now(UTC).isoformat()
        returncode, timed_out = run_bounded(command, args.stage_timeout_seconds)
        row = {
            "stage": "rq1_matching", "status": "complete" if returncode == 0 else "failed",
            "returncode": returncode, "timed_out": timed_out, "command": command,
            "started_at": started, "finished_at": datetime.now(UTC).isoformat(),
        }
        write_json(rq1_checkpoint, row)
        events.append(row)
    elif not restoration["validated"]:
        events.append({
            "stage": "rq1_matching", "status": "blocked",
            "reason": "N0 restoration was not validated; nominal RQ1 would be contaminated",
        })

    failed = sum(row.get("status") in {"failed", "blocked"} for row in events)
    report = {
        "schema_version": "fable.rq3a_then_rq1_unattended.v1",
        "finished_at": datetime.now(UTC).isoformat(),
        "events": events,
        "n0_before_rq1": restoration,
        "failed_or_blocked_stages": failed,
        "resumable": True,
    }
    write_json(root / "campaign-report.json", report)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
