#!/usr/bin/env python3
"""Prepare or execute the four-CE physical E4 mini-matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evaluation/manifests/adaptation/physical_e4_expanded.json"
DEFAULT_OUTPUT = Path("/media/brianw/Extreme SSD2/fable_results/physical_e4_expanded_20260821")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--hard-cell-timeout", type=float, default=390)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = [(ROOT / item).resolve(strict=True) for item in manifest["case_specs"]]
    total = sum(json.loads(path.read_text(encoding="utf-8"))["expected_cells"] for path in specs)
    if total != int(manifest["expected_cells"]):
        raise SystemExit(f"expected_cells mismatch: manifest={manifest['expected_cells']} specs={total}")

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec_path in specs:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        case_output = args.output / str(spec["experiment_id"])
        command = [
            sys.executable, str(ROOT / "scripts/run_physical_e4_pilot.py"),
            "--spec", str(spec_path), "--output", str(case_output),
            "--hard-cell-timeout", str(args.hard_cell_timeout),
        ]
        if args.execute:
            command.append("--execute")
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, env=os.environ.copy())
        rows.append({
            "experiment_id": spec["experiment_id"],
            "spec": str(spec_path),
            "output": str(case_output),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        })
        if completed.returncode != 0:
            break
    report = {
        "schema_version": "fable.physical_e4_expanded_preparation.v1",
        "manifest": str(manifest_path),
        "execute": args.execute,
        "expected_cells": total,
        "completed_case_preflights": len(rows),
        "valid": len(rows) == len(specs) and all(row["returncode"] == 0 for row in rows),
        "cases": rows,
    }
    (args.output / "expanded-preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
