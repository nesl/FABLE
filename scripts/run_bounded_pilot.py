#!/usr/bin/env python3
"""Run a manifest-defined live replay pilot sequentially with hard timeouts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot import generate_pilot_report, load_pilot_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case", action="append", dest="case_ids")
    args = parser.parse_args()

    manifest = load_pilot_manifest(args.manifest)
    output_dir = args.output_dir or (
        ROOT / "evaluation/results" / manifest.pilot_id
    )
    selected = set(args.case_ids or ())
    unknown = selected - {case.case_id for case in manifest.cases}
    if unknown:
        parser.error("unknown case(s): " + ", ".join(sorted(unknown)))
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = manifest.settings
    planned: list[tuple[object, int, Path, list[str]]] = []
    for case in manifest.cases:
        if selected and case.case_id not in selected:
            continue
        repetitions = case.repetitions or settings.repetitions
        nodes = case.replay_nodes or settings.replay_nodes
        for repetition in range(1, repetitions + 1):
            output = output_dir / f"{case.case_id}-r{repetition:02d}.json"
            command = [
                sys.executable,
                str(ROOT / "scripts/run_replay_accuracy.py"),
                "--baseline",
                settings.baseline,
                "--model-id",
                settings.model_id,
                "--max-seconds",
                str(settings.max_seconds),
                "--ready-seconds",
                str(settings.ready_seconds),
                "--playback-mode",
                settings.playback_mode,
                "--playback-speed",
                str(settings.playback_speed),
                "--deadline-seconds",
                str(settings.deadline_seconds),
                "--minimum-temporal-iou",
                str(settings.minimum_temporal_iou),
                "--required-ready-services",
                settings.required_ready_services,
                "--replay-nodes",
                ",".join(nodes),
                "--output",
                str(output),
            ]
            if case.experiment_id:
                command.extend(["--experiment-id", case.experiment_id])
            else:
                command.extend(["--scenario", case.scenario, "--variant", case.variant])
            planned.append((case, repetition, output, command))

    plan_path = output_dir / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": "fable.bounded_pilot_plan.v1",
                "pilot_id": manifest.pilot_id,
                "generated_at": datetime.now(UTC).isoformat(),
                "manifest": str(args.manifest),
                "runs": [
                    {
                        "case_id": case.case_id,
                        "repetition": repetition,
                        "output": str(output),
                        "command": command,
                    }
                    for case, repetition, output, command in planned
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"planned {len(planned)} runs; plan: {plan_path}", flush=True)
    if args.dry_run:
        return 0

    pilot_started_monotonic = time.monotonic()
    pilot_started_at = datetime.now(UTC)
    for index, (case, repetition, output, command) in enumerate(planned, 1):
        if output.is_file():
            print(f"[{index}/{len(planned)}] skip existing {output.name}", flush=True)
            continue
        print(
            f"[{index}/{len(planned)}] {case.case_id} repetition {repetition}",
            flush=True,
        )
        timed_out = False
        process_started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=settings.max_seconds + settings.process_grace_seconds,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            returncode = 124
        process_wall_seconds = round(time.monotonic() - process_started, 3)
        if output.is_file():
            document = json.loads(output.read_text(encoding="utf-8"))
        else:
            document = {
                "schema_version": "fable.replay_accuracy_run.v2",
                "classification": "RUNTIME_FAILURE",
                "detected": False,
                "admitted": False,
                "watch_registered": False,
                "error": (
                    "pilot runner hard timeout"
                    if timed_out
                    else "replay driver exited without writing a result"
                ),
            }
        document["pilot"] = {
            "pilot_id": manifest.pilot_id,
            "case_id": case.case_id,
            "family": case.family,
            "repetition": repetition,
            "expected_positive": case.expected_positive,
            "runner_returncode": returncode,
            "hard_timeout": timed_out,
            "process_wall_seconds": process_wall_seconds,
        }
        output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.with_suffix(".stdout.log").write_text(str(stdout), encoding="utf-8")
        output.with_suffix(".stderr.log").write_text(str(stderr), encoding="utf-8")
        print(
            f"  classification={document.get('classification')} "
            f"error={document.get('error', '')!r}",
            flush=True,
        )
        if index < len(planned):
            time.sleep(settings.inter_run_seconds)

    report = generate_pilot_report(output_dir, manifest)
    report["timing"] = {
        "started_at": pilot_started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "total_pilot_wall_seconds": round(
            time.monotonic() - pilot_started_monotonic,
            3,
        ),
        "container_startup_seconds": None,
        "container_startup_note": (
            "Containers are prepared outside run_bounded_pilot.py; use a "
            "stack wrapper to record startup/build duration."
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
