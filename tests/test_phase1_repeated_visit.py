from __future__ import annotations

import unittest
from datetime import timedelta

from fable.common.enums import HypothesisLifecycle
from fable.common.examples import BASE_TIME
from fable.common.time import EventTimeInterval
from fable.semantic import (
    ApplyStatus,
    CanonicalBindingManager,
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
    repeated_visit_graph,
    seed_result_from_spec,
)


class Phase1RepeatedVisitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bindings = CanonicalBindingManager()
        self.bindings.register_alias(
            entity_type="vehicle",
            source_id="camera_gate",
            local_entity_id="track_first",
            canonical_entity_id="vehicle_A",
        )
        self.bindings.register_alias(
            entity_type="vehicle",
            source_id="camera_gate",
            local_entity_id="track_return",
            canonical_entity_id="vehicle_A",
        )
        self.bindings.register_alias(
            entity_type="vehicle",
            source_id="camera_gate",
            local_entity_id="track_other",
            canonical_entity_id="vehicle_B",
        )
        self.runtime = SemanticRuntime(
            repeated_visit_graph(return_window_ms=300_000),
            config=SemanticRuntimeConfig(
                request_id="repeated_visit_phase1",
                hypothesis_horizon_ms=600_000,
                deadline_offset_ms=600_000,
            ),
            bindings=self.bindings,
        )

    def seed_and_depart(self):
        seed = seed_result_from_spec(
            self.runtime,
            ScriptedResultSpec(
                node_key="first_visit",
                source_id="camera_gate",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME,
                    end=BASE_TIME + timedelta(seconds=1),
                ),
                introduced={"vehicle": "track_first"},
            ),
        )
        hypothesis_id = self.runtime.seed(seed).hypothesis_ids[0]
        departure = predicate_result_from_spec(
            self.runtime,
            hypothesis_id,
            ScriptedResultSpec(
                node_key="departure",
                source_id="camera_gate",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=20),
                    end=BASE_TIME + timedelta(seconds=21),
                ),
                validated={"vehicle": "track_first"},
            ),
        )
        self.assertEqual(self.runtime.apply(departure).status, ApplyStatus.APPLIED)
        return hypothesis_id

    def test_same_canonical_entity_returns_with_new_local_track(self) -> None:
        hypothesis_id = self.seed_and_depart()
        returned = predicate_result_from_spec(
            self.runtime,
            hypothesis_id,
            ScriptedResultSpec(
                node_key="return_visit",
                source_id="camera_gate",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(minutes=2),
                    end=BASE_TIME + timedelta(minutes=2, seconds=1),
                ),
                validated={"vehicle": "track_return"},
            ),
        )
        transition = self.runtime.apply(returned)
        self.assertEqual(transition.status, ApplyStatus.APPLIED)
        hypothesis = self.runtime.get_hypothesis(hypothesis_id)
        self.assertEqual(hypothesis.lifecycle, HypothesisLifecycle.COMPLETED)
        self.assertEqual(
            set(hypothesis.role_bindings["vehicle"].local_entity_ids["camera_gate"]),
            {"track_first", "track_return"},
        )

    def test_different_entity_is_rejected(self) -> None:
        hypothesis_id = self.seed_and_depart()
        returned = predicate_result_from_spec(
            self.runtime,
            hypothesis_id,
            ScriptedResultSpec(
                node_key="return_visit",
                source_id="camera_gate",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(minutes=2),
                    end=BASE_TIME + timedelta(minutes=2, seconds=1),
                ),
                validated={"vehicle": "track_other"},
            ),
        )
        transition = self.runtime.apply(returned)
        self.assertEqual(transition.status, ApplyStatus.REJECTED)

    def test_return_after_window_is_rejected(self) -> None:
        hypothesis_id = self.seed_and_depart()
        returned = predicate_result_from_spec(
            self.runtime,
            hypothesis_id,
            ScriptedResultSpec(
                node_key="return_visit",
                source_id="camera_gate",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(minutes=6),
                    end=BASE_TIME + timedelta(minutes=6, seconds=1),
                ),
                validated={"vehicle": "track_return"},
            ),
        )
        transition = self.runtime.apply(returned)
        self.assertEqual(transition.status, ApplyStatus.REJECTED)
        self.assertIn("maximum delay", transition.reason)

    def test_one_physical_occurrence_can_advance_distinct_hypotheses(self) -> None:
        second_seed = seed_result_from_spec(
            self.runtime,
            ScriptedResultSpec(
                node_key="first_visit",
                source_id="camera_gate",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=2),
                    end=BASE_TIME + timedelta(seconds=3),
                ),
                occurrence_id="second-seed",
                introduced={"vehicle": "track_other"},
            ),
        )
        second_hypothesis = self.runtime.seed(second_seed).hypothesis_ids[0]
        first_hypothesis = self.runtime.seed(
            seed_result_from_spec(
                self.runtime,
                ScriptedResultSpec(
                    node_key="first_visit",
                    source_id="camera_gate",
                    event_time_interval=EventTimeInterval(
                        start=BASE_TIME,
                        end=BASE_TIME + timedelta(seconds=1),
                    ),
                    occurrence_id="first-seed",
                    introduced={"vehicle": "track_first"},
                ),
            )
        ).hypothesis_ids[0]

        def departure(hypothesis_id, vehicle):
            return predicate_result_from_spec(
                self.runtime,
                hypothesis_id,
                ScriptedResultSpec(
                    node_key="departure",
                    source_id="camera_gate",
                    event_time_interval=EventTimeInterval(
                        start=BASE_TIME + timedelta(seconds=20),
                        end=BASE_TIME + timedelta(seconds=21),
                    ),
                    occurrence_id="one-physical-departure",
                    validated={"vehicle": vehicle},
                ),
            )

        first = self.runtime.apply(departure(first_hypothesis, "track_first"))
        second = self.runtime.apply(departure(second_hypothesis, "track_other"))
        self.assertEqual(first.status, ApplyStatus.APPLIED)
        self.assertEqual(second.status, ApplyStatus.APPLIED)



if __name__ == "__main__":
    unittest.main()
