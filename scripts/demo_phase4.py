#!/usr/bin/env python3
"""Demonstrate Phase-4 bounded label search and the continuation trap."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fable.common.base import to_jsonable  # noqa: E402
from fable.common.examples import BASE_TIME  # noqa: E402
from fable.planning import (  # noqa: E402
    BeamSearchConfig,
    BoundedLabelPlanner,
    PhysicalAlternativeGraphBuilder,
)
from fable.planning.phase4_testing import continuation_trap_graph  # noqa: E402
from fable.planning.testing import (  # noqa: E402
    fake_artifact_catalog,
    fake_deployment,
    fake_follow_demand,
    fake_provider_registry,
)


def main() -> int:
    providers = fake_provider_registry()
    artifacts = fake_artifact_catalog()
    deployment = fake_deployment()
    demand = fake_follow_demand()
    graph = PhysicalAlternativeGraphBuilder(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
    ).build((demand,), now=BASE_TIME)

    planner = BoundedLabelPlanner(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
        config=BeamSearchConfig(beam_width=8, fallback_count=2),
    )
    result = planner.search(graph, (demand,), now=BASE_TIME)

    trap_graph, trap_demand = continuation_trap_graph(
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
        trap_graph,
        (trap_demand,),
        now=BASE_TIME,
        required_checkpoint_consumers=("multi_object_tracker",),
    )
    wider = BoundedLabelPlanner(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
        config=BeamSearchConfig(beam_width=2, fallback_count=1),
    ).search(
        trap_graph,
        (trap_demand,),
        now=BASE_TIME,
        required_checkpoint_consumers=("multi_object_tracker",),
    )

    summary = {
        "normal_search": {
            "alternatives": len(graph.alternatives),
            "selected_label": result.trace.selected_label_id,
            "selected_chains": (
                result.selected.selected_chain_ids if result.selected else ()
            ),
            "fallbacks": result.trace.fallback_label_ids,
            "cost": result.selected.label.cost if result.selected else None,
            "oracle": result.trace.oracle,
            "pruning": Counter(
                record.code.value
                for boundary in result.trace.boundaries
                for record in boundary.pruning_records
            ),
        },
        "continuation_trap": {
            "beam_1_selected": greedy.trace.selected_label_id,
            "beam_1_oracle": greedy.trace.oracle,
            "beam_2_selected": wider.trace.selected_label_id,
            "beam_2_alternative": (
                wider.selected.selected_alternative_ids if wider.selected else ()
            ),
            "beam_2_continuations": (
                wider.selected.label.continuation_output_types if wider.selected else ()
            ),
            "beam_2_oracle": wider.trace.oracle,
        },
    }
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
