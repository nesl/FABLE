from __future__ import annotations

import unittest
from datetime import timedelta

from fable.common.examples import BASE_TIME, robbery_graph
from fable.common.time import EventTimeInterval
from fable.semantic import (
    ApplyStatus,
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
    seed_result_from_spec,
)


class Phase1ResultApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = SemanticRuntime(
            robbery_graph(),
            config=SemanticRuntimeConfig(
                request_id="result_application",
                hypothesis_horizon_ms=240_000,
                deadline_offset_ms=240_000,
            ),
        )
        seed = seed_result_from_spec(
            self.runtime,
            ScriptedResultSpec(
                node_key="entry",
                source_id="camera",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME,
                    end=BASE_TIME + timedelta(seconds=1),
                ),
                introduced={"person": "person_1", "location": "store"},
            ),
        )
        self.hypothesis_id = self.runtime.seed(seed).hypothesis_ids[0]

    def test_duplicate_result_is_suppressed(self) -> None:
        result = predicate_result_from_spec(
            self.runtime,
            self.hypothesis_id,
            ScriptedResultSpec(
                node_key="gunshot",
                source_id="microphone",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=3),
                    end=BASE_TIME + timedelta(seconds=3),
                ),
            ),
        )
        first = self.runtime.apply(result)
        second = self.runtime.apply(result)
        self.assertEqual(first.status, ApplyStatus.APPLIED)
        self.assertEqual(second.status, ApplyStatus.DUPLICATE)

    def test_stale_expected_version_is_rejected(self) -> None:
        gunshot = predicate_result_from_spec(
            self.runtime,
            self.hypothesis_id,
            ScriptedResultSpec(
                node_key="gunshot",
                source_id="microphone",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=3),
                    end=BASE_TIME + timedelta(seconds=3),
                ),
                occurrence_id="occ_gunshot",
            ),
        )
        stale_other_branch = predicate_result_from_spec(
            self.runtime,
            self.hypothesis_id,
            ScriptedResultSpec(
                node_key="threat",
                source_id="microphone",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=4),
                    end=BASE_TIME + timedelta(seconds=5),
                ),
                occurrence_id="occ_threat_stale",
            ),
        )
        self.assertEqual(self.runtime.apply(gunshot).status, ApplyStatus.APPLIED)
        transition = self.runtime.apply(stale_other_branch)
        self.assertEqual(transition.status, ApplyStatus.STALE)


if __name__ == "__main__":
    unittest.main()
