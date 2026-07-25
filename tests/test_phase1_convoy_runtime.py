from __future__ import annotations

import unittest
from datetime import timedelta

from fable.common.enums import HypothesisLifecycle
from fable.common.examples import BASE_TIME, convoy_graph
from fable.common.time import EventTimeInterval, SourceWatermark, WatermarkSnapshot
from fable.semantic import (
    ApplyStatus,
    CanonicalBindingManager,
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
    seed_result_from_spec,
)


class Phase1ConvoyRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bindings = CanonicalBindingManager()
        self.bindings.register_alias(
            entity_type="vehicle",
            source_id="camera_mobile",
            local_entity_id="track_17",
            canonical_entity_id="vehicle_17",
        )
        self.runtime = SemanticRuntime(
            convoy_graph(),
            config=SemanticRuntimeConfig(
                request_id="convoy_phase1",
                hypothesis_horizon_ms=60_000,
                deadline_offset_ms=60_000,
                lateness_policy={"allowed_lateness_ms": 1000},
            ),
            bindings=self.bindings,
        )

    def seed_leader(self):
        seed = seed_result_from_spec(
            self.runtime,
            ScriptedResultSpec(
                node_key="leader_passes",
                source_id="camera_mobile",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME,
                    end=BASE_TIME + timedelta(seconds=1),
                ),
                introduced={"leader": "track_17"},
            ),
        )
        transition = self.runtime.seed(seed)
        self.assertEqual(transition.status, ApplyStatus.CREATED)
        hypothesis_id = transition.hypothesis_ids[0]
        frontier = self.runtime.get_frontier(hypothesis_id)
        self.assertIsNotNone(frontier)
        enabled_keys = {
            self.runtime.graph.nodes_by_id[node_id].authored_key
            for node_id in frontier.snapshot.enabled_node_ids
        }
        self.assertEqual(enabled_keys, {"follower_follows"})
        return hypothesis_id

    def test_seed_result_creates_hypothesis(self) -> None:
        hypothesis_id = self.seed_leader()
        hypothesis = self.runtime.get_hypothesis(hypothesis_id)
        self.assertEqual(
            hypothesis.role_bindings["leader"].canonical_entity_id,
            "vehicle_17",
        )
        self.assertEqual(hypothesis.version, 0)

    def test_distinct_followers_fork_and_aliases_merge(self) -> None:
        parent_id = self.seed_leader()
        for local_id, canonical_id in (
            ("track_23", "vehicle_23"),
            ("track_99", "vehicle_99"),
            ("track_23_reacquired", "vehicle_23"),
        ):
            self.bindings.register_alias(
                entity_type="vehicle",
                source_id="camera_downstream",
                local_entity_id=local_id,
                canonical_entity_id=canonical_id,
            )

        first = predicate_result_from_spec(
            self.runtime,
            parent_id,
            ScriptedResultSpec(
                node_key="follower_follows",
                source_id="camera_downstream",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=2),
                    end=BASE_TIME + timedelta(seconds=6),
                ),
                introduced={"follower": "track_23"},
                validated={"leader": "vehicle_17"},
                occurrence_id="occ_follower_23",
            ),
        )
        second = predicate_result_from_spec(
            self.runtime,
            parent_id,
            ScriptedResultSpec(
                node_key="follower_follows",
                source_id="camera_downstream",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=2),
                    end=BASE_TIME + timedelta(seconds=6),
                ),
                introduced={"follower": "track_99"},
                validated={"leader": "vehicle_17"},
                occurrence_id="occ_follower_99",
            ),
        )
        alias = predicate_result_from_spec(
            self.runtime,
            parent_id,
            ScriptedResultSpec(
                node_key="follower_follows",
                source_id="camera_downstream",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=2),
                    end=BASE_TIME + timedelta(seconds=6),
                ),
                introduced={"follower": "track_23_reacquired"},
                validated={"leader": "vehicle_17"},
                occurrence_id="occ_follower_23_alias",
            ),
        )

        first_transition = self.runtime.apply(first)
        second_transition = self.runtime.apply(second)
        alias_transition = self.runtime.apply(alias)
        self.assertEqual(first_transition.status, ApplyStatus.FORKED)
        self.assertEqual(second_transition.status, ApplyStatus.FORKED)
        self.assertEqual(alias_transition.status, ApplyStatus.MERGED)

        parent = self.runtime.get_hypothesis(parent_id)
        self.assertEqual(parent.lifecycle, HypothesisLifecycle.FORKED)
        active = self.runtime.active_hypotheses
        self.assertEqual(len(active), 2)
        followers = {
            hypothesis.role_bindings["follower"].canonical_entity_id
            for hypothesis in active
        }
        self.assertEqual(followers, {"vehicle_23", "vehicle_99"})
        vehicle_23 = next(
            hypothesis
            for hypothesis in active
            if hypothesis.role_bindings["follower"].canonical_entity_id == "vehicle_23"
        )
        aliases = vehicle_23.role_bindings["follower"].local_entity_ids["camera_downstream"]
        self.assertEqual(set(aliases), {"track_23", "track_23_reacquired"})


    def test_follower_cannot_resolve_to_bound_leader(self) -> None:
        parent_id = self.seed_leader()
        self.bindings.register_alias(
            entity_type="vehicle",
            source_id="camera_downstream",
            local_entity_id="track_same_vehicle",
            canonical_entity_id="vehicle_17",
        )
        result = predicate_result_from_spec(
            self.runtime,
            parent_id,
            ScriptedResultSpec(
                node_key="follower_follows",
                source_id="camera_downstream",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=2),
                    end=BASE_TIME + timedelta(seconds=6),
                ),
                introduced={"follower": "track_same_vehicle"},
                validated={"leader": "vehicle_17"},
            ),
        )
        transition = self.runtime.apply(result)
        self.assertEqual(transition.status, ApplyStatus.REJECTED)
        self.assertIn("must bind distinct entities", transition.reason)

    def test_absence_requires_coverage_and_closing_watermark(self) -> None:
        parent_id = self.seed_leader()
        self.bindings.register_alias(
            entity_type="vehicle",
            source_id="camera_downstream",
            local_entity_id="track_23",
            canonical_entity_id="vehicle_23",
        )
        follows = predicate_result_from_spec(
            self.runtime,
            parent_id,
            ScriptedResultSpec(
                node_key="follower_follows",
                source_id="camera_downstream",
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=2),
                    end=BASE_TIME + timedelta(seconds=6),
                ),
                introduced={"follower": "track_23"},
                validated={"leader": "vehicle_17"},
            ),
        )
        child_id = self.runtime.apply(follows).hypothesis_ids[0]
        child = self.runtime.get_hypothesis(child_id)
        frontier = self.runtime.get_frontier(child_id)
        self.assertIsNotNone(frontier)
        checkpoint = frontier.checkpoint_for_node(
            self.runtime.graph.nodes_by_key["moving_vehicle"].node_id
        )
        close_time = checkpoint.event_time_interval.end

        early = WatermarkSnapshot(
            generated_at=close_time,
            sources={
                "camera_mobile": SourceWatermark(
                    source_id="camera_mobile",
                    event_time=close_time,
                    operational_coverage=True,
                )
            },
        )
        self.assertEqual(self.runtime.close_temporal_windows(early), ())

        no_coverage = WatermarkSnapshot(
            generated_at=close_time + timedelta(seconds=2),
            sources={
                "camera_mobile": SourceWatermark(
                    source_id="camera_mobile",
                    event_time=close_time + timedelta(seconds=2),
                    operational_coverage=False,
                )
            },
        )
        self.assertEqual(self.runtime.close_temporal_windows(no_coverage), ())

        closed = WatermarkSnapshot(
            generated_at=close_time + timedelta(seconds=2),
            sources={
                "camera_mobile": SourceWatermark(
                    source_id="camera_mobile",
                    event_time=close_time + timedelta(seconds=2),
                    operational_coverage=True,
                )
            },
        )
        transitions = self.runtime.close_temporal_windows(closed)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0].status, ApplyStatus.WINDOW_CLOSED)
        completed = self.runtime.get_hypothesis(child_id)
        self.assertEqual(completed.lifecycle, HypothesisLifecycle.COMPLETED)


if __name__ == "__main__":
    unittest.main()
