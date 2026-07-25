from __future__ import annotations

import unittest
from datetime import timedelta

from fable.common.enums import HypothesisLifecycle, HypothesisNodeStatus
from fable.common.examples import BASE_TIME, robbery_graph
from fable.common.time import EventTimeInterval
from fable.semantic import (
    ApplyStatus,
    CanonicalBindingManager,
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
    seed_result_from_spec,
)


class Phase1RobberyRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bindings = CanonicalBindingManager()
        self.bindings.register_alias(
            entity_type="person",
            source_id="store_camera",
            local_entity_id="track_person_7",
            canonical_entity_id="person_7",
        )
        self.runtime = SemanticRuntime(
            robbery_graph(),
            config=SemanticRuntimeConfig(
                request_id="robbery_phase1",
                hypothesis_horizon_ms=240_000,
                deadline_offset_ms=240_000,
            ),
            bindings=self.bindings,
        )

    def seed_entry(self):
        result = seed_result_from_spec(
            self.runtime,
            ScriptedResultSpec(
                node_key="entry",
                source_id="store_camera",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME,
                    end=BASE_TIME + timedelta(seconds=2),
                ),
                introduced={
                    "person": "track_person_7",
                    "location": "store_zone",
                },
            ),
        )
        transition = self.runtime.seed(result)
        self.assertEqual(transition.status, ApplyStatus.CREATED)
        return transition.hypothesis_ids[0]

    def test_or_alternatives_share_one_hypothesis_and_cancel_losers(self) -> None:
        hypothesis_id = self.seed_entry()
        frontier = self.runtime.get_frontier(hypothesis_id)
        self.assertIsNotNone(frontier)
        enabled = {
            self.runtime.graph.nodes_by_id[node_id].authored_key
            for node_id in frontier.snapshot.enabled_node_ids
        }
        self.assertEqual(enabled, {"gunshot", "threat", "forced_transfer", "failed_attempt"})
        self.assertEqual(len(frontier.checkpoints), 1)

        gunshot = predicate_result_from_spec(
            self.runtime,
            hypothesis_id,
            ScriptedResultSpec(
                node_key="gunshot",
                source_id="store_microphone",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=5),
                    end=BASE_TIME + timedelta(seconds=5),
                ),
            ),
        )
        transition = self.runtime.apply(gunshot)
        self.assertEqual(transition.status, ApplyStatus.APPLIED)
        self.assertEqual(len(self.runtime.active_hypotheses), 1)
        self.assertEqual(
            set(transition.cancellation.branch_ids),
            {"threat", "forced_transfer", "failed_attempt"},
        )
        cancelled_keys = {
            self.runtime.graph.nodes_by_id[node_id].authored_key
            for node_id in transition.cancellation.node_ids
        }
        self.assertTrue({"threat", "forced_transfer", "failed_attempt"}.issubset(cancelled_keys))

        updated = self.runtime.get_hypothesis(hypothesis_id)
        self.assertEqual(updated.structural_branch_ids, ("gunshot",))
        self.assertEqual(
            updated.node_states[self.runtime.graph.nodes_by_key["threat"].node_id].status,
            HypothesisNodeStatus.INVALIDATED,
        )
        next_frontier = self.runtime.get_frontier(hypothesis_id)
        next_keys = {
            self.runtime.graph.nodes_by_id[node_id].authored_key
            for node_id in next_frontier.snapshot.enabled_node_ids
        }
        self.assertEqual(next_keys, {"departure"})

        departure = predicate_result_from_spec(
            self.runtime,
            hypothesis_id,
            ScriptedResultSpec(
                node_key="departure",
                source_id="store_camera",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=10),
                    end=BASE_TIME + timedelta(seconds=11),
                ),
                validated={"person": "track_person_7"},
            ),
        )
        completed = self.runtime.apply(departure)
        self.assertEqual(completed.status, ApplyStatus.APPLIED)
        self.assertEqual(
            self.runtime.get_hypothesis(hypothesis_id).lifecycle,
            HypothesisLifecycle.COMPLETED,
        )

    def test_failed_or_branch_leaves_other_branches_enabled(self) -> None:
        hypothesis_id = self.seed_entry()
        threat_false = predicate_result_from_spec(
            self.runtime,
            hypothesis_id,
            ScriptedResultSpec(
                node_key="threat",
                source_id="store_microphone",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=4),
                    end=BASE_TIME + timedelta(seconds=6),
                ),
                truth="FALSE",
            ),
        )
        transition = self.runtime.apply(threat_false)
        self.assertEqual(transition.status, ApplyStatus.APPLIED)
        frontier = self.runtime.get_frontier(hypothesis_id)
        enabled = {
            self.runtime.graph.nodes_by_id[node_id].authored_key
            for node_id in frontier.snapshot.enabled_node_ids
        }
        self.assertEqual(enabled, {"gunshot", "forced_transfer", "failed_attempt"})


if __name__ == "__main__":
    unittest.main()
