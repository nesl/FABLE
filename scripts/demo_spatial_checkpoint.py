#!/usr/bin/env python3
"""Demonstrate site-aware next-sensor selection and request compilation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fable.semantic.request_compiler import EventRequestCompiler
from fable.spatial import SiteSensorTransitionModel, SpatialObservation, load_sensor_bindings



def main() -> int:
    compiled = EventRequestCompiler().compile("detect a convoy")
    model = SiteSensorTransitionModel.from_json(
        ROOT / "evaluation" / "labels" / "site_sensor_transition_model_2024_2025.json"
    )
    bindings = load_sensor_bindings(
        ROOT / "iobt-minimal-ce-replay" / "config" / "fable_spatial_bindings.yaml"
    )
    prediction = model.predict(
        SpatialObservation(
            current_sensor_id="orin_6",
            observed_heading="SW",
            active_deployment_id="2025_package_exchange",
            branch_unresolved=True,
        ),
        bindings=bindings,
    )
    print(
        json.dumps(
            {
                "compiled_family": compiled.family_id,
                "graph_name": compiled.graph.name,
                "compile_warnings": compiled.warnings,
                "spatial_prediction": prediction.model_dump(mode="json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
