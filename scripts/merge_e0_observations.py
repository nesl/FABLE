#!/usr/bin/env python3
"""Merge bounded E0 campaign shards without duplicate run identifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.experiments.e0_calibration import CalibrationObservation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_run: dict[str, CalibrationObservation] = {}
    for path in args.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = CalibrationObservation.model_validate(json.loads(line))
            if item.run_id in by_run:
                raise ValueError(f"duplicate calibration run ID: {item.run_id}")
            by_run[item.run_id] = item
    ordered = sorted(by_run.values(), key=lambda item: item.run_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(item.model_dump_json() + "\n" for item in ordered),
        encoding="utf-8",
    )
    print(json.dumps({"observation_count": len(ordered), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
