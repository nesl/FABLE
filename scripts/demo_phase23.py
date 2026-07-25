#!/usr/bin/env python3
"""Replay the Phase-1 convoy seed and construct Phase-2/3 physical alternatives."""

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
from fable.planning import PhysicalAlternativeGraphBuilder  # noqa: E402
from fable.planning.testing import (  # noqa: E402
    fake_artifact_catalog,
    fake_deployment,
    fake_follow_demand,
    fake_provider_registry,
)


def main() -> int:
    demand = fake_follow_demand()
    builder = PhysicalAlternativeGraphBuilder(
        provider_registry=fake_provider_registry(),
        artifact_catalog=fake_artifact_catalog(),
        deployment=fake_deployment(),
    )
    graph = builder.build((demand,), now=BASE_TIME)
    summary = {
        "semantic_demand": {
            "predicate": demand.semantic_predicate.predicate_id,
            "bound_roles": demand.bound_roles,
            "unbound_roles": demand.unbound_roles,
            "forkable_roles": demand.binding_policy.forkable_roles,
            "eligible_sources": demand.eligible_source_ids,
            "desired_continuations": demand.desired_continuation_types,
        },
        "physical_alternative_graph": {
            "graph_id": graph.graph_id,
            "checkpoint_ids": graph.checkpoint_ids,
            "demand_ids": graph.demand_ids,
            "alternatives": len(graph.alternatives),
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "pruned": len(graph.pruned),
            "chains": Counter(alt.chain_id for alt in graph.alternatives),
            "execution_modes": Counter(alt.execution_mode.value for alt in graph.alternatives),
        },
        "example_alternatives": [
            {
                "alternative_id": alternative.alternative_id,
                "chain_id": alternative.chain_id,
                "execution_mode": alternative.execution_mode,
                "completion_ms": alternative.estimated_completion_ms,
                "transfer_bytes": alternative.estimated_transfer_bytes,
                "placements": [
                    {
                        "step": step.step_id,
                        "provider": step.provider_id,
                        "node": step.node_id,
                    }
                    for step in alternative.step_placements
                ],
                "continuations": alternative.continuation_output_types,
            }
            for alternative in graph.alternatives[:3]
        ],
    }
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
