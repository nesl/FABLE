#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.catalog import ExperimentCatalog
from evaluation.experiments.matrix import build_run_matrix
from evaluation.experiments.specs import ExperimentQuestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a controlled FABLE evaluation run matrix.")
    parser.add_argument("question", choices=[item.value for item in ExperimentQuestion])
    parser.add_argument("--output", type=Path, default=Path("evaluation/manifests/workloads/run_matrix.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("evaluation/labels/filtered_complex_event_experiments.csv"))
    parser.add_argument("--transition-model", type=Path, default=Path("evaluation/labels/site_sensor_transition_model_2024_2025.json"))
    args = parser.parse_args()
    catalog = ExperimentCatalog.from_csv(args.labels, transition_model_path=args.transition_model)
    runs = build_run_matrix(catalog, ExperimentQuestion(args.question))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(run.model_dump_json(exclude_none=True) + "\n")
    print(f"wrote {len(runs)} planned runs to {args.output}")


if __name__ == "__main__":
    main()
