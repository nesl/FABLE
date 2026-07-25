from __future__ import annotations

import unittest
from datetime import timedelta

from fable.common.enums import HypothesisLifecycle
from fable.common.examples import convoy_graph, robbery_graph
from fable.common.time import SourceWatermark, WatermarkSnapshot
from fable.semantic import (
    ApplyStatus,
    CanonicalBindingManager,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
    seed_result_from_spec,
)
from tests.fake_phase1_data import load_trace, parse_timestamp, result_specs


class Phase1FakeTraceTests(unittest.TestCase):
    def _runtime(self, trace_name: str):
        trace = load_trace(trace_name)
        graph = convoy_graph() if trace["graph"] == "convoy" else robbery_graph()
        bindings = CanonicalBindingManager()
        for alias in trace.get("aliases", []):
            bindings.register_alias(
                entity_type=alias["entity_type"],
                source_id=alias["source_id"],
                local_entity_id=alias["local_id"],
                canonical_entity_id=alias["canonical_id"],
            )
        runtime = SemanticRuntime(
            graph,
            config=SemanticRuntimeConfig(
                request_id=trace["request_id"],
                hypothesis_horizon_ms=300_000,
                deadline_offset_ms=300_000,
                lateness_policy={"allowed_lateness_ms": 1000},
            ),
            bindings=bindings,
        )
        return trace, runtime

    def test_robbery_fake_trace_completes(self) -> None:
        trace, runtime = self._runtime("robbery_trace.json")
        specs = result_specs(trace)
        created = runtime.seed(seed_result_from_spec(runtime, specs[0]))
        hypothesis_id = created.hypothesis_ids[0]
        for spec in specs[1:]:
            self.assertEqual(
                runtime.apply(predicate_result_from_spec(runtime, hypothesis_id, spec)).status,
                ApplyStatus.APPLIED,
            )
        self.assertEqual(
            runtime.get_hypothesis(hypothesis_id).lifecycle,
            HypothesisLifecycle.COMPLETED,
        )

    def test_convoy_fake_trace_closes_absence(self) -> None:
        trace, runtime = self._runtime("convoy_trace.json")
        specs = result_specs(trace)
        parent_id = runtime.seed(seed_result_from_spec(runtime, specs[0])).hypothesis_ids[0]
        fork = runtime.apply(predicate_result_from_spec(runtime, parent_id, specs[1]))
        child_id = fork.hypothesis_ids[0]
        watermark_data = trace["watermarks"]["camera_mobile"]
        watermark = WatermarkSnapshot(
            generated_at=parse_timestamp(watermark_data["event_time"]) + timedelta(seconds=1),
            sources={
                "camera_mobile": SourceWatermark(
                    source_id="camera_mobile",
                    event_time=parse_timestamp(watermark_data["event_time"]),
                    operational_coverage=watermark_data["operational_coverage"],
                )
            },
        )
        transitions = runtime.close_temporal_windows(watermark)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(
            runtime.get_hypothesis(child_id).lifecycle,
            HypothesisLifecycle.COMPLETED,
        )


if __name__ == "__main__":
    unittest.main()
