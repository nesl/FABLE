from __future__ import annotations

import unittest
from datetime import timedelta

from fable.common.enums import HypothesisLifecycle, ResultKind
from fable.common.examples import BASE_TIME
from fable.common.time import EventTimeInterval
from fable.semantic import (
    ApplyStatus,
    AuthoredGraphBuilder,
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
    seed_result_from_spec,
)


class Phase1CompositeRuntimeTests(unittest.TestCase):
    def test_k_of_n_completes_after_k_children(self) -> None:
        builder = AuthoredGraphBuilder(namespace="test.kofn", name="K-of-N")
        p1 = builder.primitive("p1", name="P1", predicate_id="P1", checkpoint=True)
        p2 = builder.primitive("p2", name="P2", predicate_id="P2", checkpoint=True)
        p3 = builder.primitive("p3", name="P3", predicate_id="P3", checkpoint=True)
        root = builder.k_of_n("root", (p1, p2, p3), k=2, name="Two of three")
        graph = builder.root(root).compile()
        runtime = SemanticRuntime(
            graph,
            config=SemanticRuntimeConfig(request_id="kofn", hypothesis_horizon_ms=60_000),
        )
        seed = seed_result_from_spec(
            runtime,
            ScriptedResultSpec(
                node_key="p1",
                source_id="source",
                event_time_interval=EventTimeInterval(start=BASE_TIME, end=BASE_TIME),
            ),
        )
        hypothesis_id = runtime.seed(seed).hypothesis_ids[0]
        frontier = runtime.get_frontier(hypothesis_id)
        enabled = {runtime.graph.nodes_by_id[node].authored_key for node in frontier.snapshot.enabled_node_ids}
        self.assertEqual(enabled, {"p2", "p3"})
        p2_result = predicate_result_from_spec(
            runtime,
            hypothesis_id,
            ScriptedResultSpec(
                node_key="p2",
                source_id="source",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=1),
                    end=BASE_TIME + timedelta(seconds=1),
                ),
            ),
        )
        self.assertEqual(runtime.apply(p2_result).status, ApplyStatus.APPLIED)
        self.assertEqual(runtime.get_hypothesis(hypothesis_id).lifecycle, HypothesisLifecycle.COMPLETED)

    def test_duration_accumulates_contiguous_intervals(self) -> None:
        builder = AuthoredGraphBuilder(namespace="test.duration", name="Duration")
        state = builder.primitive(
            "state",
            name="State",
            predicate_id="STATE",
            result_kind=ResultKind.STATE_OBSERVATION,
        )
        root = builder.duration("root", state, minimum_ms=5_000, name="State for five seconds")
        runtime = SemanticRuntime(
            builder.root(root).compile(),
            config=SemanticRuntimeConfig(request_id="duration", hypothesis_horizon_ms=60_000),
        )
        seed = seed_result_from_spec(
            runtime,
            ScriptedResultSpec(
                node_key="state",
                source_id="source",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME,
                    end=BASE_TIME + timedelta(seconds=2),
                ),
            ),
        )
        hypothesis_id = runtime.seed(seed).hypothesis_ids[0]
        self.assertEqual(runtime.get_hypothesis(hypothesis_id).lifecycle, HypothesisLifecycle.ACTIVE)
        second = predicate_result_from_spec(
            runtime,
            hypothesis_id,
            ScriptedResultSpec(
                node_key="state",
                source_id="source",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=2),
                    end=BASE_TIME + timedelta(seconds=6),
                ),
                occurrence_id="state_interval_2",
            ),
        )
        self.assertEqual(runtime.apply(second).status, ApplyStatus.APPLIED)
        self.assertEqual(runtime.get_hypothesis(hypothesis_id).lifecycle, HypothesisLifecycle.COMPLETED)


if __name__ == "__main__":
    unittest.main()
