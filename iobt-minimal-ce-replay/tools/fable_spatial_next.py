#!/usr/bin/env python3
"""Inspect the qualitative next-sensor prediction used by FABLE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fable.spatial import (  # noqa: E402
    SiteSensorTransitionModel,
    SpatialObservation,
    load_sensor_bindings,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-sensor", required=True)
    parser.add_argument("--heading")
    parser.add_argument("--deployment")
    parser.add_argument("--corridor")
    parser.add_argument("--branch-unresolved", action="store_true")
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "evaluation" / "labels" / "site_sensor_transition_model_2024_2025.json",
    )
    parser.add_argument(
        "--bindings",
        type=Path,
        default=ROOT / "iobt-minimal-ce-replay" / "config" / "fable_spatial_bindings.yaml",
    )
    args = parser.parse_args()
    model = SiteSensorTransitionModel.from_json(args.model)
    bindings = load_sensor_bindings(args.bindings)
    prediction = model.predict(
        SpatialObservation(
            current_sensor_id=args.current_sensor,
            observed_heading=args.heading,
            active_deployment_id=args.deployment,
            corridor_id=args.corridor,
            branch_unresolved=args.branch_unresolved,
            maximum_observation_groups=args.groups,
        ),
        bindings=bindings,
    )
    print(json.dumps(prediction.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
