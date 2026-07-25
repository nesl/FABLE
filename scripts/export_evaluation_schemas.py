#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.schemas import (
    ArtifactEvent,
    ComplexEventResult,
    CoordinationEpisode,
    HypothesisTransition,
    NetworkCondition,
    PlanDecision,
    PredicateObservation,
    ProviderLifecycleEvent,
    ResourceSample,
)

MODELS = (
    PredicateObservation,
    ComplexEventResult,
    ProviderLifecycleEvent,
    ArtifactEvent,
    PlanDecision,
    HypothesisTransition,
    NetworkCondition,
    ResourceSample,
    CoordinationEpisode,
)


def main() -> None:
    output = ROOT / "evaluation/schemas/json"
    output.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        name = model.__name__
        path = output / f"{name}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
