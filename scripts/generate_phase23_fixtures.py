#!/usr/bin/env python3
"""Generate compact deterministic fake-data fixtures for Phase 2 and Phase 3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fable.common.base import to_jsonable  # noqa: E402
from fable.common.examples import BASE_TIME  # noqa: E402
from fable.planning import AlternativeBuildConfig, PhysicalAlternativeGraphBuilder  # noqa: E402
from fable.planning.testing import (  # noqa: E402
    fake_artifact_catalog,
    fake_deployment,
    fake_follow_demand,
    fake_provider_registry,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    output = PROJECT_ROOT / "tests" / "phase23_fixtures"
    demand = fake_follow_demand()
    graph = PhysicalAlternativeGraphBuilder(
        provider_registry=fake_provider_registry(),
        artifact_catalog=fake_artifact_catalog(),
        deployment=fake_deployment(),
        config=AlternativeBuildConfig(
            max_external_assignments_per_chain=4,
            max_placement_variants_per_assignment=2,
            max_alternatives_per_chain=2,
            max_total_alternatives=6,
        ),
    ).build((demand,), now=BASE_TIME)
    write_json(output / "follows_demand.json", demand)
    write_json(output / "physical_alternative_graph.json", graph)
    write_json(output / "deployment_nodes.json", list(fake_deployment().nodes.values()))
    write_json(output / "artifacts.json", list(fake_artifact_catalog().artifacts))
    print(f"wrote fixtures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
