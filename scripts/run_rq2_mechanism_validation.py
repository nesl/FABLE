#!/usr/bin/env python3
"""Bounded RQ2 mechanism check for joint continuation-aware planning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.baselines.models import BaselinePlanningCase  # noqa: E402
from evaluation.baselines.policies import (  # noqa: E402
    ExhaustiveOraclePolicy,
    FablePolicy,
    GreedyFrontierPolicy,
)
from fable.common.examples import BASE_TIME  # noqa: E402
from fable.common.schemas import ContinuationRequirement  # noqa: E402
from fable.planning import BeamSearchConfig, BoundedLabelPlanner  # noqa: E402
from fable.planning.phase4_testing import fake_follow_alternative_graph  # noqa: E402
from fable.planning.testing import (  # noqa: E402
    fake_artifact_catalog,
    fake_deployment,
    fake_follow_demand,
    fake_provider_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    providers = fake_provider_registry()
    artifacts = fake_artifact_catalog()
    deployment = fake_deployment()
    graph, demand = fake_follow_alternative_graph(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
    )
    cheap = min(
        (item for item in graph.alternatives if not item.continuation_output_types),
        key=lambda item: (item.estimated_completion_ms, item.alternative_id),
    )
    rich = min(
        (
            item
            for item in graph.alternatives
            if "pair_trajectory.v1" in item.continuation_output_types
        ),
        key=lambda item: (item.estimated_completion_ms, item.alternative_id),
    )
    cheap = cheap.model_copy(
        update={
            "estimated_completion_ms": 10,
        }
    )
    rich = rich.model_copy(
        update={
            "estimated_completion_ms": 20,
        }
    )
    graph = graph.model_copy(update={"alternatives": (cheap, rich)})
    demand = demand.model_copy(
        update={
            "continuation_requirements": (
                ContinuationRequirement(
                    artifact_type="pair_trajectory.v1",
                    required_until=demand.deadline.latest_useful_completion,
                ),
            )
        }
    )
    case = BaselinePlanningCase(
        run_id="rq2-mechanism-continuation-trap",
        trace_id="controlled-continuation-trap",
        request_id="rq2-mechanism-request",
        event_family="route_convoy",
        frontier_demands=(demand,),
        all_task_demands=(demand,),
        frontier_graph=graph,
        whole_event_graph=graph,
        now=BASE_TIME,
    )
    planner = BoundedLabelPlanner(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
        config=BeamSearchConfig(beam_width=2, fallback_count=1),
    )
    policies = (
        GreedyFrontierPolicy(),
        FablePolicy(planner),
        ExhaustiveOraclePolicy(planner),
    )
    decisions = {item.baseline_id.value: item.plan(case) for item in policies}
    greedy = decisions["B4_GREEDY_FRONTIER"]
    fable = decisions["FABLE"]
    oracle = decisions["O1_EXHAUSTIVE_ORACLE"]
    valid = (
        greedy.selected_alternative_ids == (cheap.alternative_id,)
        and fable.selected_alternative_ids == (rich.alternative_id,)
        and fable.selected_alternative_ids == oracle.selected_alternative_ids
        and "pair_trajectory.v1" in fable.continuation_types
    )
    result = {
        "schema_version": "fable.rq2_mechanism_validation.v1",
        "valid": valid,
        "controlled_case": True,
        "claim": (
            "FABLE preserves a required checkpoint continuation that immediate "
            "greedy selection discards, and matches the bounded exhaustive oracle."
        ),
        "decisions": {
            key: value.model_dump(mode="json") for key, value in decisions.items()
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "mechanism_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"valid": valid, "output": str(args.output)}, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
