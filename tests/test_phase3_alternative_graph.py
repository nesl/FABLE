from __future__ import annotations

import unittest
from collections import Counter
from datetime import timedelta

from fable.common.enums import ExecutionMode
from fable.common.examples import BASE_TIME
from fable.common.time import DeadlineSpec
from fable.planning import AlternativeBuildConfig, PhysicalAlternativeGraphBuilder
from fable.planning.models import ActiveProviderInstance, TransferMode
from fable.planning.testing import (
    fake_artifact_catalog,
    fake_deployment,
    fake_follow_demand,
    fake_provider_registry,
)


class Phase3AlternativeGraphTests(unittest.TestCase):
    def build(self, demand=None, *, config=None, now=BASE_TIME):
        demand = demand or fake_follow_demand()
        builder = PhysicalAlternativeGraphBuilder(
            provider_registry=fake_provider_registry(),
            artifact_catalog=fake_artifact_catalog(),
            deployment=fake_deployment(),
            config=config,
        )
        return builder.build((demand,), now=now)

    def test_follow_demand_expands_joint_provider_placement_and_representation_choices(self) -> None:
        graph = self.build()
        by_chain = Counter(alt.chain_id for alt in graph.alternatives)
        self.assertEqual(
            set(by_chain),
            {
                "follows_local_tracks",
                "follows_local_from_retained_detections",
                "follows_cross_camera_reid",
            },
        )
        self.assertIn(ExecutionMode.LIVE, {alt.execution_mode for alt in graph.alternatives})
        self.assertIn(
            ExecutionMode.RETROSPECTIVE,
            {alt.execution_mode for alt in graph.alternatives},
        )
        placement_signatures = {
            tuple((step.provider_id, step.node_id) for step in alt.step_placements)
            for alt in graph.alternatives
        }
        self.assertGreater(len(placement_signatures), 3)
        self.assertTrue(
            any(
                any(step.provider_id == "track_crop_extractor" for step in alt.step_placements)
                and any(step.provider_id == "vehicle_reid_descriptor" for step in alt.step_placements)
                for alt in graph.alternatives
            )
        )

    def test_raw_inputs_are_not_transferred_when_local_only(self) -> None:
        graph = self.build()
        for alternative in graph.alternatives:
            raw_transfers = [
                transfer
                for transfer in alternative.transfers
                if transfer.data_type in {"raw_video_frames.v1", "audio_segment.v1"}
            ]
            self.assertTrue(all(item.mode == TransferMode.LOCAL for item in raw_transfers))

    def test_graph_is_bounded_to_current_checkpoint_and_demand(self) -> None:
        demand = fake_follow_demand()
        graph = self.build(demand)
        self.assertEqual(graph.demand_ids, (demand.demand_id,))
        self.assertEqual(graph.checkpoint_ids, (demand.checkpoint_id,))
        self.assertTrue(all(alt.demand_id == demand.demand_id for alt in graph.alternatives))
        self.assertTrue(all(alt.checkpoint_id == demand.checkpoint_id for alt in graph.alternatives))

    def test_required_pair_trajectory_prunes_cross_camera_chain(self) -> None:
        demand = fake_follow_demand(required_continuation="pair_trajectory.v1")
        graph = self.build(demand)
        self.assertTrue(graph.alternatives)
        self.assertNotIn(
            "follows_cross_camera_reid",
            {alt.chain_id for alt in graph.alternatives},
        )
        self.assertTrue(
            any(
                item.chain_id == "follows_cross_camera_reid"
                and item.code == "CONTINUATION_UNAVAILABLE"
                for item in graph.pruned
            )
        )

    def test_tight_transfer_budget_prunes_cross_sensor_realizations(self) -> None:
        demand = fake_follow_demand().model_copy(deep=True)
        demand.hard_constraints.maximum_transfer_bytes = 0
        graph = self.build(demand)
        self.assertTrue(graph.alternatives)
        self.assertTrue(
            any(item.code == "TRANSFER_BUDGET_EXCEEDED" for item in graph.pruned)
        )
        self.assertTrue(
            all(alt.estimated_transfer_bytes == 0 for alt in graph.alternatives)
        )


    def test_active_provider_instance_removes_cold_start_for_matching_step(self) -> None:
        demand = fake_follow_demand()
        builder = PhysicalAlternativeGraphBuilder(
            provider_registry=fake_provider_registry(),
            artifact_catalog=fake_artifact_catalog(),
            deployment=fake_deployment(),
            active_providers=(
                ActiveProviderInstance(
                    provider_instance_id="detector_sensor_a",
                    provider_id="yolo_vehicle_fast_640",
                    node_id="sensor_a",
                    output_data_types=("detection_set.v1",),
                ),
            ),
        )
        graph = builder.build((demand,), now=BASE_TIME)
        reused = [
            step
            for alternative in graph.alternatives
            for step in alternative.step_placements
            if step.reused_provider_instance_id == "detector_sensor_a"
        ]
        self.assertTrue(reused)
        self.assertTrue(all(step.startup_ms == 0 for step in reused))

    def test_deadline_infeasible_alternatives_are_pruned(self) -> None:
        demand = fake_follow_demand().model_copy(
            update={
                "deadline": DeadlineSpec(
                    latest_useful_completion=BASE_TIME + timedelta(milliseconds=100)
                )
            }
        )
        graph = self.build(demand, now=BASE_TIME)
        self.assertEqual(graph.alternatives, ())
        self.assertTrue(any(item.code == "DEADLINE_INFEASIBLE" for item in graph.pruned))


if __name__ == "__main__":
    unittest.main()
