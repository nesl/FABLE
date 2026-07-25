#!/usr/bin/env python3
"""Generate fake physical graphs and planner traces for Phase-4 tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fable.common.base import to_jsonable  # noqa: E402
from fable.common.examples import BASE_TIME  # noqa: E402
from fable.planning import BeamSearchConfig, BoundedLabelPlanner  # noqa: E402
from fable.planning.phase4_testing import continuation_trap_graph  # noqa: E402
from fable.planning.testing import (  # noqa: E402
    fake_artifact_catalog,
    fake_deployment,
    fake_provider_registry,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    providers = fake_provider_registry()
    artifacts = fake_artifact_catalog()
    deployment = fake_deployment()
    graph, demand = continuation_trap_graph(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
    )
    greedy = BoundedLabelPlanner(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
        config=BeamSearchConfig(beam_width=1, fallback_count=0),
    ).search(
        graph,
        (demand,),
        now=BASE_TIME,
        required_checkpoint_consumers=("multi_object_tracker",),
    )
    wider = BoundedLabelPlanner(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
        config=BeamSearchConfig(beam_width=2, fallback_count=1),
    ).search(
        graph,
        (demand,),
        now=BASE_TIME,
        required_checkpoint_consumers=("multi_object_tracker",),
    )

    output = PROJECT_ROOT / "tests" / "phase4_fixtures"
    write_json(output / "continuation_trap_graph.json", graph)
    write_json(output / "continuation_trap_demand.json", demand)
    write_json(output / "beam_width_1_result.json", greedy)
    write_json(output / "beam_width_2_result.json", wider)
    print(f"wrote fixtures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
