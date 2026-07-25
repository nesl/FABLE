#!/usr/bin/env python3
"""Regenerate deterministic Phase-7 planner/profile fixture summaries."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from fable.common.examples import BASE_TIME
from fable.planning import PhysicalAlternativeGraphBuilder
from fable.planning.testing import (
    fake_artifact_catalog,
    fake_deployment,
    fake_follow_demand,
    fake_provider_registry,
)

OUTPUT = ROOT / "tests/phase7_fixtures"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registry = fake_provider_registry()
    demand = fake_follow_demand()
    graph = PhysicalAlternativeGraphBuilder(
        provider_registry=registry,
        artifact_catalog=fake_artifact_catalog(),
        deployment=fake_deployment(),
    ).build((demand,), now=BASE_TIME)
    (OUTPUT / "vehicle_planning_summary.json").write_text(
        json.dumps(
            {
                "demand": demand.model_dump(mode="json"),
                "candidate_chains": sorted(
                    chain.chain_id for chain in registry.candidate_chains(demand)
                ),
                "realized_chains": sorted({item.chain_id for item in graph.alternatives}),
                "pruned": [item.model_dump(mode="json") for item in graph.pruned],
                "tracker_recovery": "replay detection_set.v1 into a fresh tracker session",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
