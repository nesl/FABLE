#!/usr/bin/env python3
"""Promote a complete E0 desktop campaign; never promote partial fixtures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.calibration_promotion import (  # noqa: E402
    CalibrationPromotionPolicy,
    promote_desktop_observations,
)
from evaluation.experiments.e0_calibration import (  # noqa: E402
    CalibrationObservation,
    CalibrationTarget,
    write_profiles,
)
from providers.calibration_worker import worker_capabilities  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("observations", type=Path)
    parser.add_argument("readiness", type=Path)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--minimum-warm", type=int, default=30)
    parser.add_argument("--minimum-cold", type=int, default=10)
    parser.add_argument(
        "--measured-only",
        action="store_true",
        help=(
            "require every production MEASURED_PROVIDER worker target while "
            "leaving validation-only and unimplemented targets explicitly "
            "excluded rather than promoting them"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation/manifests/providers/calibrated_profiles.json",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    observations = tuple(
        CalibrationObservation.model_validate_json(line)
        for line in args.observations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    readiness = json.loads(args.readiness.read_text(encoding="utf-8"))
    operations = worker_capabilities()["operations"]
    targets = tuple(
        CalibrationTarget.model_validate(item["target"])
        for item in readiness["targets"]
        if item["status"] == "READY_CONTAINER"
        and (
            not args.measured_only
            or (
                operations.get(item["target"]["provider_id"], {}).get(
                    "measurement_status"
                )
                == "MEASURED_PROVIDER"
                and item["target"]["input_class"]
                in operations.get(item["target"]["provider_id"], {}).get(
                    "input_classes", ()
                )
            )
        )
    )
    report = promote_desktop_observations(
        observations,
        targets,
        worker_operations=operations,
        policy=CalibrationPromotionPolicy(
            host_id=args.host_id,
            minimum_warm_samples=args.minimum_warm,
            minimum_cold_samples=args.minimum_cold,
        ),
        generated_at=datetime.now(timezone.utc),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if not report.promotable:
        print(report.model_dump_json(), file=sys.stderr)
        return 2
    write_profiles(report.profiles, args.output)
    print(report.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
