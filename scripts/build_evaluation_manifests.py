#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.catalog import ExperimentCatalog
from evaluation.manifests import ManifestBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("evaluation/labels/filtered_complex_event_experiments.csv"),
    )
    parser.add_argument(
        "--transition-model",
        type=Path,
        default=Path("evaluation/labels/site_sensor_transition_model_2024_2025.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/manifests/workloads"),
    )
    args = parser.parse_args()
    catalog = ExperimentCatalog.from_csv(
        args.labels,
        transition_model_path=args.transition_model,
    )
    jsonl_path, summary_path = ManifestBuilder(catalog).write(args.output)
    print(f"wrote {jsonl_path}")
    print(f"wrote {summary_path}")
    print(catalog.summary().model_dump_json(indent=2))


if __name__ == "__main__":
    main()
