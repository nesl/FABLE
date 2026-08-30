#!/usr/bin/env python3
"""Run the refactored planning-smoke evaluation manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import load_manifest, summarize_outcomes  # noqa: E402
from evaluation.runner import run_planning_cell  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outcomes = []
    with args.output.open("w", encoding="utf-8") as stream:
        for cell in manifest.cells:
            outcome, detail = run_planning_cell(cell)
            outcomes.append(outcome)
            stream.write(
                json.dumps({"outcome": outcome.as_dict(), "detail": detail}, sort_keys=True)
                + "\n"
            )
            stream.flush()
    summary = summarize_outcomes(outcomes)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(outcomes[0].as_dict()))
        writer.writeheader()
        writer.writerows(row.as_dict() for row in outcomes)
    print(summary_path)
    return int(any(row.status != "SUCCESS" for row in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
