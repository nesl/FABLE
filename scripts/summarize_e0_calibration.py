#!/usr/bin/env python3
"""Reduce completed E0 observation JSONL into measured provider/tier profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.experiments.e0_calibration import (  # noqa: E402
    CalibrationObservation,
    summarize_observations,
    write_profiles,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize completed E0 calibration observations."
    )
    parser.add_argument("observations", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation/manifests/providers/calibrated_profiles.json",
    )
    args = parser.parse_args()
    observations = tuple(
        CalibrationObservation.model_validate_json(line)
        for line in args.observations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    profiles = summarize_observations(observations)
    destination = write_profiles(profiles, args.output)
    print(
        json.dumps(
            {
                "schema_version": "fable.calibration_summary.v1",
                "observation_count": len(observations),
                "profile_count": len(profiles),
                "output": str(destination.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
