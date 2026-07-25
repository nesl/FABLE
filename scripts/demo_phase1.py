#!/usr/bin/env python3
"""Replay deterministic fake convoy and robbery traces through the Phase-1 runtime."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fable.common.enums import HypothesisLifecycle
from fable.common.examples import convoy_graph, robbery_graph
from fable.common.time import EventTimeInterval, SourceWatermark, WatermarkSnapshot
from fable.semantic import (
    CanonicalBindingManager,
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
    seed_result_from_spec,
)

FIXTURE_DIR = PROJECT_ROOT / "tests" / "phase1_fixtures"


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_trace(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def spec(item: dict) -> ScriptedResultSpec:
    return ScriptedResultSpec(
        node_key=item["node_key"],
        source_id=item["source_id"],
        event_time_interval=EventTimeInterval(
            start=parse_time(item["start"]),
            end=parse_time(item["end"]),
        ),
        introduced=item.get("introduced", {}),
        validated=item.get("validated", {}),
        truth=item.get("truth", "TRUE"),
        occurrence_id=item.get("occurrence_id"),
    )


def make_runtime(trace: dict) -> SemanticRuntime:
    graph = convoy_graph() if trace["graph"] == "convoy" else robbery_graph()
    bindings = CanonicalBindingManager()
    for alias in trace.get("aliases", []):
        bindings.register_alias(
            entity_type=alias["entity_type"],
            source_id=alias["source_id"],
            local_entity_id=alias["local_id"],
            canonical_entity_id=alias["canonical_id"],
        )
    return SemanticRuntime(
        graph,
        config=SemanticRuntimeConfig(
            request_id=trace["request_id"],
            hypothesis_horizon_ms=300_000,
            deadline_offset_ms=300_000,
            lateness_policy={"allowed_lateness_ms": 1000},
        ),
        bindings=bindings,
    )


def run_trace(name: str) -> dict:
    trace = load_trace(name)
    runtime = make_runtime(trace)
    result_specs = [spec(item) for item in trace["results"]]
    seed_transition = runtime.seed(seed_result_from_spec(runtime, result_specs[0]))
    hypothesis_id = seed_transition.hypothesis_ids[0]
    transitions = [seed_transition.status.value]

    for item, result_spec in zip(trace["results"][1:], result_specs[1:]):
        transition = runtime.apply(
            predicate_result_from_spec(runtime, hypothesis_id, result_spec)
        )
        transitions.append(transition.status.value)
        if transition.hypothesis_ids and transition.status.value == "FORKED":
            hypothesis_id = transition.hypothesis_ids[0]

    if trace.get("watermarks"):
        sources = {}
        latest = None
        for source_id, payload in trace["watermarks"].items():
            event_time = parse_time(payload["event_time"])
            latest = event_time if latest is None else max(latest, event_time)
            sources[source_id] = SourceWatermark(
                source_id=source_id,
                event_time=event_time,
                operational_coverage=payload["operational_coverage"],
            )
        closed = runtime.close_temporal_windows(
            WatermarkSnapshot(
                generated_at=latest + timedelta(seconds=1),
                sources=sources,
            )
        )
        transitions.extend(item.status.value for item in closed)

    hypothesis = runtime.get_hypothesis(hypothesis_id)
    frontier = runtime.get_frontier(hypothesis_id)
    return {
        "trace": name,
        "transitions": transitions,
        "final_hypothesis_id": str(hypothesis_id),
        "lifecycle": hypothesis.lifecycle.value,
        "bindings": {
            role: binding.canonical_entity_id
            for role, binding in hypothesis.role_bindings.items()
        },
        "structural_branches": list(hypothesis.structural_branch_ids),
        "frontier": []
        if frontier is None
        else [
            runtime.graph.nodes_by_id[node_id].authored_key
            for node_id in frontier.snapshot.enabled_node_ids
        ],
    }


def main() -> int:
    summaries = [
        run_trace("convoy_trace.json"),
        run_trace("robbery_trace.json"),
    ]
    print(json.dumps(summaries, indent=2, sort_keys=True))
    if any(item["lifecycle"] != HypothesisLifecycle.COMPLETED.value for item in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
