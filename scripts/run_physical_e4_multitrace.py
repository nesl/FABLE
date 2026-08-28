#!/usr/bin/env python3
"""Run the prepared multi-trace physical E4 campaign, calibrating B1 first."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evaluation/manifests/adaptation/physical_e4_multitrace.json"
DEFAULT_OUTPUT = Path("/media/brianw/Extreme SSD2/fable_results/physical_e4_multitrace_20260821")


def invoke(
    spec: Path,
    output: Path,
    *,
    execute: bool,
    hard_cell_timeout: float,
    process_timeout: float,
) -> subprocess.CompletedProcess:
    command = [
        sys.executable, str(ROOT / "scripts/run_physical_e4_pilot.py"),
        "--spec", str(spec), "--output", str(output),
        "--hard-cell-timeout", str(hard_cell_timeout),
    ]
    if execute:
        command.append("--execute")
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=process_timeout,
        env=os.environ.copy(),
    )


def preserve_invoke_diagnostics(
    output: Path, completed: subprocess.CompletedProcess, *, phase: str
) -> dict[str, str]:
    """Persist captured child diagnostics before reducing a run to a row."""

    output.mkdir(parents=True, exist_ok=True)
    stdout_path = output / f"{phase}-invoke.stdout.log"
    stderr_path = output / f"{phase}-invoke.stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    return {"stdout": str(stdout_path), "stderr": str(stderr_path)}


def case_spec(case: dict, calibration: str | None, *, baselines=None, conditions=None) -> dict:
    selected_baselines = baselines or case["baselines"]
    names = conditions or case["conditions"]
    condition_paths = {"nominal": None, **case["condition_trace_paths"]}
    return {
        "schema_version": "fable.physical_e4_pilot.v1",
        "pilot_id": f"physical-e4-multitrace-{case['experiment_id']}",
        "experiment_id": case["experiment_id"],
        "logical_physical_node_id": case["logical_physical_node_id"],
        "replay_sources": case["replay_sources"],
        "playback_mode": "realtime", "playback_speed": 1.0, "repetitions": 1,
        "baselines": selected_baselines,
        "conditions": [
            {"condition_id": name, "condition_trace": condition_paths[name]}
            for name in names
        ],
        "b1_calibration_result": calibration,
        "max_seconds": case["max_seconds"],
        "expected_cells": len(selected_baselines) * len(names),
        "adaptive_baselines": [b for b in selected_baselines if b in {"B3_TASK_RESOURCE_ADAPTIVE", "FABLE"}],
        "required_adaptation_pairs": [
            {"baseline": b, "condition": "compute_contention"}
            for b in selected_baselines if b in {"B3_TASK_RESOURCE_ADAPTIVE", "FABLE"}
            if "compute_contention" in names
        ],
        "require_adaptation_discrimination": "compute_contention" in names,
        "allow_raw_to_trusted_site_edge": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--hard-cell-timeout", type=float, default=390)
    parser.add_argument(
        "--experiment-id",
        action="append",
        default=[],
        help="Run only the named experiment; repeat to select multiple cases.",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.resolve(strict=True).read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    specs_dir = args.output / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    capability = subprocess.run(
        ["ssh", "-i", "/tmp/fable_deploy_key", "-o", "BatchMode=yes", "rpi",
         "sudo -n /usr/local/sbin/fable-physical-net capabilities"],
        cwd=ROOT, text=True, capture_output=True,
    )
    capability_text = capability.stdout + "\n" + capability.stderr
    # The first deployed disconnect helper predates the informational
    # ``capabilities`` verb but advertises its fixed allowlist in the usage
    # response.  Accept either form without attempting a mutation.
    disconnect_ready = "disconnect" in capability_text
    if not disconnect_ready:
        report = {
            "manifest": str(args.manifest.resolve()), "execute": args.execute,
            "ready": False, "blocker": "RPI_DISCONNECT_HELPER_NOT_INSTALLED",
            "install_source": str(ROOT / "scripts/fable_physical_net"),
            "remote_staged_source": "/tmp/fable-physical-net.e4-new",
            "helper_stdout": capability.stdout, "helper_stderr": capability.stderr,
            "rows": [],
        }
        (args.output / "multitrace-run-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2
    selected_ids = set(args.experiment_id)
    available_ids = {case["experiment_id"] for case in manifest["cases"]}
    missing_ids = sorted(selected_ids - available_ids)
    if missing_ids:
        parser.error(f"unknown experiment id(s): {', '.join(missing_ids)}")
    selected_cases = [
        case
        for case in manifest["cases"]
        if not selected_ids or case["experiment_id"] in selected_ids
    ]
    for case in selected_cases:
        experiment_id = case["experiment_id"]
        calibration = case.get("b1_calibration_result")
        case_root = args.output / experiment_id
        requires_b1 = "B1_STATIC_WHOLE_EVENT" in case["baselines"]
        if calibration is None and requires_b1:
            calibration_spec = specs_dir / f"{experiment_id}.calibration.json"
            calibration_spec.write_text(json.dumps(
                case_spec(case, None, baselines=["FABLE"], conditions=["nominal"]),
                indent=2, sort_keys=True,
            ) + "\n", encoding="utf-8")
            calibration_output = case_root / "calibration-prepass"
            if not args.execute:
                rows.append({"experiment_id": experiment_id, "status": "CALIBRATION_REQUIRED"})
                continue
            calibration_timeout = max(
                args.hard_cell_timeout, float(case["hard_cell_timeout_seconds"])
            )
            completed = invoke(
                calibration_spec, calibration_output,
                execute=True,
                hard_cell_timeout=calibration_timeout,
                process_timeout=calibration_timeout + 180.0,
            )
            diagnostics = preserve_invoke_diagnostics(
                calibration_output, completed, phase="calibration"
            )
            calibration_result = calibration_output / "nominal/FABLE/repetition-01" / f"{experiment_id}.json"
            try:
                result = json.loads(calibration_result.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                result = {}
            if completed.returncode or result.get("classification") != "TRUE_POSITIVE":
                rows.append({
                    "experiment_id": experiment_id,
                    "status": "CALIBRATION_FAILED",
                    "returncode": completed.returncode,
                    **diagnostics,
                })
                continue
            calibration = str(calibration_result)
        spec_path = specs_dir / f"{experiment_id}.json"
        spec_path.write_text(json.dumps(case_spec(case, calibration), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        cell_timeout = max(args.hard_cell_timeout, float(case["hard_cell_timeout_seconds"]))
        matrix_cells = len(case["baselines"]) * len(case["conditions"])
        completed = invoke(
            spec_path,
            case_root / "matrix",
            execute=args.execute,
            hard_cell_timeout=cell_timeout,
            # The child pilot applies hard_cell_timeout independently to each
            # cell. The enclosing process must cover every cell plus bounded
            # setup, cleanup, and report-generation overhead.
            process_timeout=cell_timeout * matrix_cells + 600.0,
        )
        diagnostics = preserve_invoke_diagnostics(
            case_root / "matrix", completed, phase="matrix"
        )
        rows.append({
            "experiment_id": experiment_id,
            "status": "COMPLETE" if completed.returncode == 0 else "FAILED",
            "returncode": completed.returncode,
            **diagnostics,
        })
    report = {
        "manifest": str(args.manifest.resolve()),
        "execute": args.execute,
        "selected_experiment_ids": [case["experiment_id"] for case in selected_cases],
        "rows": rows,
    }
    (args.output / "multitrace-run-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(any(row["status"] in {"FAILED", "CALIBRATION_FAILED"} for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
