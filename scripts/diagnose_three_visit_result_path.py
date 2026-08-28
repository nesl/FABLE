#!/usr/bin/env python3
"""Exercise three-visit result delivery without replaying sensor media."""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.live_execution import AuthoritativeLiveExecution
from fable.common.examples import BASE_TIME
from fable.common.ids import uuid7
from fable.common.time import EventTimeInterval
from fable.distributed.models import ControlEventType, ResultInboxRecord
from fable.distributed.topics import result_topic
from fable.planning import DemandCompiler, default_predicate_registry
from fable.planning.testing import fake_deployment
from fable.semantic import (
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
    seed_result_from_spec,
    uncalibrated_repeated_pass_graph,
)
from tests.fake_phase6_data import make_stack, wait_until


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("debug/three_visit_result_correlation.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request_id = "synthetic-three-visit-correlation"
    vehicle = "dvpg_gq_orin_11:synthetic-session:0"
    reference = "camera_fov:dvpg_gq_orin_11"
    runtime = SemanticRuntime(
        uncalibrated_repeated_pass_graph(
            visit_count=3,
            minimum_return_gap_ms=30_000,
            identity_confirmation=True,
        ),
        config=SemanticRuntimeConfig(
            request_id=request_id,
            hypothesis_horizon_ms=600_000,
            deadline_offset_ms=600_000,
        ),
    )
    compiler = DemandCompiler(
        predicate_registry=default_predicate_registry(),
        deployment=fake_deployment(),
    )
    seed = seed_result_from_spec(
        runtime,
        ScriptedResultSpec(
            node_key="first_visit",
            source_id="orin11_camera",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME + timedelta(seconds=1),
            ),
            introduced={"vehicle": vehicle, "visit_reference": reference},
        ),
    )
    hypothesis_id = runtime.seed(seed).hypothesis_ids[-1]
    transitions = []

    def apply_result(result):
        transition = runtime.apply(result)
        transitions.append(transition)
        return transition

    emitted = []
    with tempfile.TemporaryDirectory(prefix="fable-three-visit-") as tmp:
        stack = make_stack(
            Path(tmp),
            nodes=("sensor_a",),
            on_result=apply_result,
        )
        try:
            agent = stack.agents["sensor_a"]

            def deliver(result, boundary: str) -> None:
                wrapper = agent.forward_result(
                    result=result,
                    provider_instance_id=f"synthetic-{boundary}",
                    attempt_id=uuid7(),
                    topic=result_topic(request_id, result.semantic_predicate.predicate_id),
                )
                emitted.append(
                    {
                        "boundary": "NODE_RESULT_CREATED",
                        "logical_boundary": boundary,
                        "message_id": str(wrapper.message_id),
                        "result_id": str(result.result_id),
                        "occurrence_id": result.occurrence_id,
                        "predicate_id": result.semantic_predicate.predicate_id,
                        "hypothesis_id": str(result.hypothesis_id),
                        "expected_hypothesis_version": result.expected_hypothesis_version,
                        "demand_id": str(result.demand_id),
                        "frontier_id": str(result.frontier_id),
                        "checkpoint_id": str(result.checkpoint_id),
                        "graph_node_id": str(result.graph_node_id),
                        "event_time_start": result.event_time_interval.start.isoformat(),
                        "event_time_end": result.event_time_interval.end.isoformat(),
                    }
                )
                expected = len(emitted)
                if not wait_until(
                    lambda: len(stack.store.list("result_inbox", ResultInboxRecord))
                    >= expected
                ):
                    raise RuntimeError(f"result never reached inbox: {result.result_id}")

            # Reproduce the real trace's first noisy successor observation:
            # it targets the correct return frontier but is too close to the
            # seed to satisfy the 30-second absence guard.  A rejection here
            # must not poison the same demand or suppress its later result.
            early_duplicate = predicate_result_from_spec(
                runtime,
                hypothesis_id,
                ScriptedResultSpec(
                    node_key="return_visit",
                    source_id="orin11_camera",
                    event_time_interval=EventTimeInterval(
                        start=BASE_TIME + timedelta(seconds=1),
                        end=BASE_TIME + timedelta(seconds=2),
                    ),
                    introduced={"visit_vehicle_2": vehicle},
                    validated={"visit_reference": reference},
                ),
            )
            deliver(early_duplicate, "early_duplicate_before_gap")
            if transitions[-1].status.value != "REJECTED":
                raise RuntimeError("pre-gap synthetic visit was not rejected")

            for index, (node_key, variable, seconds) in enumerate(
                (
                    ("return_visit", "visit_vehicle_2", 78),
                    ("return_visit_2", "visit_vehicle_3", 152),
                ),
                start=2,
            ):
                visit = predicate_result_from_spec(
                    runtime,
                    hypothesis_id,
                    ScriptedResultSpec(
                        node_key=node_key,
                        source_id="orin11_camera",
                        event_time_interval=EventTimeInterval(
                            start=BASE_TIME + timedelta(seconds=seconds),
                            end=BASE_TIME + timedelta(seconds=seconds + 1),
                        ),
                        introduced={variable: vehicle},
                        validated={"visit_reference": reference},
                    ),
                )
                deliver(visit, f"visit_{index}")
                hypothesis_id = transitions[-1].hypothesis_ids[-1]

                hypothesis = runtime.get_hypothesis(hypothesis_id)
                frontier = runtime.get_frontier(hypothesis_id)
                if frontier is None:
                    raise RuntimeError("identity frontier was not produced")
                demands = compiler.compile_frontier(
                    graph=runtime.graph,
                    hypothesis=hypothesis,
                    frontier=frontier,
                )
                if len(demands) != 1:
                    raise RuntimeError(f"expected one identity demand, got {len(demands)}")
                identity = AuthoritativeLiveExecution._reflexive_identity_result(
                    demands[0], now=BASE_TIME + timedelta(seconds=seconds + 1)
                )
                if identity is None:
                    raise RuntimeError("equal scoped identities did not resolve reflexively")
                deliver(identity, f"identity_{index}")
                hypothesis_id = transitions[-1].hypothesis_ids[-1]

            events = [
                event.model_dump(mode="json")
                for event in stack.store.list_events()
                if event.event_type
                in {
                    ControlEventType.RESULT_RECEIVED,
                    ControlEventType.RESULT_APPLICATION_STARTED,
                    ControlEventType.RESULT_APPLIED,
                    ControlEventType.RESULT_SEMANTICALLY_REJECTED,
                    ControlEventType.RESULT_DUPLICATE,
                }
            ]
            inbox = [
                item.model_dump(mode="json")
                for item in stack.store.list("result_inbox", ResultInboxRecord)
            ]
            final_hypothesis = runtime.get_hypothesis(hypothesis_id)
            report = {
                "schema_version": "fable.three_visit_result_correlation.v1",
                "request_id": request_id,
                "passed": final_hypothesis.lifecycle.value == "COMPLETED",
                "final_hypothesis_id": str(hypothesis_id),
                "final_lifecycle": final_hypothesis.lifecycle.value,
                "node_boundaries": emitted,
                "orchestrator_events": events,
                "result_inbox": inbox,
                "semantic_transitions": [
                    item.model_dump(mode="json") for item in transitions
                ],
            }
        finally:
            stack.stop()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "final_lifecycle": report["final_lifecycle"],
        "node_results": len(report["node_boundaries"]),
        "orchestrator_events": len(report["orchestrator_events"]),
        "inbox_records": len(report["result_inbox"]),
        "output": str(args.output),
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
