from __future__ import annotations

import unittest
from collections import Counter

from pydantic import ValidationError

from fable.common.enums import BindingCapability
from fable.common.examples import BASE_TIME
from fable.common.ids import uuid7
from fable.planning import (
    BeamSearchConfig,
    BoundedLabelPlanner,
    DeploymentGraph,
    PhysicalAlternativeGraph,
    PhysicalAlternativeGraphBuilder,
    PruneCode,
    RepresentationCompatibility,
)
from fable.planning.phase4_testing import continuation_trap_graph
from fable.planning.testing import (
    fake_artifact_catalog,
    fake_deployment,
    fake_follow_demand,
    fake_provider_registry,
)


class Phase4BeamSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.providers = fake_provider_registry()
        self.artifacts = fake_artifact_catalog()
        self.deployment = fake_deployment()
        self.demand = fake_follow_demand()
        self.graph = PhysicalAlternativeGraphBuilder(
            provider_registry=self.providers,
            artifact_catalog=self.artifacts,
            deployment=self.deployment,
        ).build((self.demand,), now=BASE_TIME)

    def planner(self, **updates) -> BoundedLabelPlanner:
        config = BeamSearchConfig(**updates)
        return BoundedLabelPlanner(
            provider_registry=self.providers,
            artifact_catalog=self.artifacts,
            deployment=self.deployment,
            config=config,
        )

    def test_real_graph_returns_deterministic_label_plan_fallbacks_and_trace(self) -> None:
        first = self.planner(beam_width=8, fallback_count=2).search(
            self.graph, (self.demand,), now=BASE_TIME
        )
        second = self.planner(beam_width=8, fallback_count=2).search(
            self.graph, (self.demand,), now=BASE_TIME
        )
        self.assertIsNotNone(first.selected)
        self.assertEqual(first.selected.label_id, second.selected.label_id)
        self.assertEqual(first.trace.selected_label_id, first.selected.label_id)
        self.assertEqual(first.execution_plan.label_id, first.selected.label_id)
        self.assertEqual(first.execution_plan.demand_ids, (self.demand.demand_id,))
        self.assertEqual(len(first.fallbacks), 2)
        self.assertEqual(first.trace.oracle.status.value, "MATCHED")
        codes = Counter(
            record.code
            for boundary in first.trace.boundaries
            for record in boundary.pruning_records
        )
        self.assertGreater(codes[PruneCode.DOMINATED], 0)

    def test_physical_plan_labels_and_extension_state_are_immutable(self) -> None:
        alternative = self.graph.alternatives[0]
        planner = self.planner(run_oracle=False)
        self.assertFalse(
            planner.check_extension(
                None,
                alternative,
                self.demand,
                demand_map={self.demand.demand_id: self.demand},
                now=BASE_TIME,
            )
        )
        state = planner.extend_label(
            None,
            alternative,
            self.demand,
            demand_map={self.demand.demand_id: self.demand},
            now=BASE_TIME,
        )
        original_id = state.label_id
        with self.assertRaises(ValidationError):
            state.label.parent_label_id = "mutated"
        with self.assertRaises(ValidationError):
            state.total_cpu_cores = 0
        self.assertEqual(state.label_id, original_id)

    def test_beam_width_one_is_deterministic_greedy_and_can_hit_continuation_trap(self) -> None:
        graph, demand = continuation_trap_graph(
            provider_registry=self.providers,
            artifact_catalog=self.artifacts,
            deployment=self.deployment,
        )
        planner = self.planner(beam_width=1, fallback_count=0)
        first = planner.search(
            graph,
            (demand,),
            now=BASE_TIME,
            required_checkpoint_consumers=("track_summary_route_evaluator",),
        )
        second = planner.search(
            graph,
            (demand,),
            now=BASE_TIME,
            required_checkpoint_consumers=("track_summary_route_evaluator",),
        )
        self.assertIsNone(first.selected)
        self.assertIsNone(second.selected)
        self.assertEqual(first.trace.search_id, second.trace.search_id)
        self.assertEqual(first.trace.oracle.status.value, "GAP")
        self.assertTrue(
            any(
                record.code == PruneCode.CHECKPOINT_CONTINUATION_INCOMPATIBLE
                for boundary in first.trace.boundaries
                for record in boundary.pruning_records
            )
        )

    def test_larger_beam_avoids_continuation_trap_and_matches_oracle(self) -> None:
        graph, demand = continuation_trap_graph(
            provider_registry=self.providers,
            artifact_catalog=self.artifacts,
            deployment=self.deployment,
        )
        result = self.planner(beam_width=2, fallback_count=1).search(
            graph,
            (demand,),
            now=BASE_TIME,
            required_checkpoint_consumers=("track_summary_route_evaluator",),
        )
        self.assertIsNotNone(result.selected)
        self.assertEqual(
            result.selected.selected_alternative_ids,
            ("alt_continuation_trap_rich_summary",),
        )
        self.assertIn(
            "track_summary.v1",
            result.selected.label.continuation_output_types,
        )
        self.assertEqual(result.trace.oracle.status.value, "MATCHED")

    def test_representation_order_preserves_incomparable_artifacts(self) -> None:
        order = RepresentationCompatibility(self.providers)
        pair = ("pair_trajectory.v1",)
        tracker = ("track_summary.v1",)
        self.assertFalse(order.dominates(pair, tracker))
        self.assertFalse(order.dominates(tracker, pair))
        self.assertFalse(order.comparable(pair, tracker))

    def test_quality_filter_has_explicit_pruning_provenance(self) -> None:
        result = self.planner(
            beam_width=8,
            minimum_quality_score=0.99,
            run_oracle=False,
        ).search(self.graph, (self.demand,), now=BASE_TIME)
        self.assertIsNone(result.selected)
        records = [
            record
            for boundary in result.trace.boundaries
            for record in boundary.pruning_records
        ]
        self.assertTrue(any(item.code == PruneCode.QUALITY_FLOOR for item in records))
        self.assertTrue(all(item.reason for item in records))

    def test_multi_label_capacity_is_checked_against_current_deployment(self) -> None:
        tiny_nodes = tuple(
            node.model_copy(
                update={
                    "capacity": node.capacity.model_copy(
                        update={"cpu_cores": 0.1, "memory_mb": 64, "gpu_memory_mb": 0}
                    )
                }
            )
            for node in self.deployment.nodes.values()
        )
        tiny = DeploymentGraph(
            nodes=tiny_nodes,
            sources=self.deployment.sources.values(),
            links=self.deployment.links,
        )
        planner = BoundedLabelPlanner(
            provider_registry=self.providers,
            artifact_catalog=self.artifacts,
            deployment=tiny,
            config=BeamSearchConfig(run_oracle=False),
        )
        result = planner.search(self.graph, (self.demand,), now=BASE_TIME)
        self.assertIsNone(result.selected)
        self.assertTrue(
            any(
                record.code == PruneCode.CAPACITY_EXCEEDED
                for boundary in result.trace.boundaries
                for record in boundary.pruning_records
            )
        )

    def test_result_schema_mismatch_is_rejected(self) -> None:
        bad = self.graph.alternatives[0].model_copy(
            update={"result_output_type": "wrong_result.v1"}
        )
        graph = PhysicalAlternativeGraph(
            graph_id="physical_graph_bad_schema",
            checkpoint_ids=self.graph.checkpoint_ids,
            demand_ids=self.graph.demand_ids,
            nodes=self.graph.nodes,
            edges=self.graph.edges,
            alternatives=(bad,),
            built_at=BASE_TIME,
        )
        result = self.planner(run_oracle=False).search(
            graph, (self.demand,), now=BASE_TIME
        )
        self.assertIsNone(result.selected)
        self.assertTrue(
            any(
                record.code == PruneCode.RESULT_SCHEMA_INCOMPATIBLE
                for boundary in result.trace.boundaries
                for record in boundary.pruning_records
            )
        )

    def test_declared_binding_capability_mismatch_is_rejected(self) -> None:
        bad_demand = self.demand.model_copy(deep=True)
        bad_demand.binding_policy.role_modes["follower"] = BindingCapability.AGGREGATE
        result = self.planner(run_oracle=False).search(
            self.graph, (bad_demand,), now=BASE_TIME
        )
        self.assertIsNone(result.selected)
        self.assertTrue(
            any(
                record.code == PruneCode.BINDING_CAPABILITY_MISSING
                for boundary in result.trace.boundaries
                for record in boundary.pruning_records
            )
        )

    def test_capacity_is_aggregated_across_demands_in_one_checkpoint(self) -> None:
        base = next(
            alternative
            for alternative in self.graph.alternatives
            if alternative.chain_id == "follows_local_tracks"
        )
        placements = tuple(
            step.model_copy(
                update={
                    "cpu_cores": 0.5,
                    "memory_mb": 128,
                    "gpu_memory_mb": 0,
                }
            )
            for step in base.step_placements
        )
        first_alt = base.model_copy(
            update={
                "alternative_id": "alt_capacity_first",
                "step_placements": placements,
            }
        )
        second_demand = self.demand.model_copy(
            update={"demand_id": uuid7(now_ms=1_900_000_000_001)}
        )
        second_alt = first_alt.model_copy(
            update={
                "alternative_id": "alt_capacity_second",
                "demand_id": second_demand.demand_id,
            }
        )
        graph = PhysicalAlternativeGraph(
            graph_id="physical_graph_capacity_pair",
            checkpoint_ids=self.graph.checkpoint_ids,
            demand_ids=(self.demand.demand_id, second_demand.demand_id),
            nodes=self.graph.nodes,
            edges=self.graph.edges,
            alternatives=(first_alt, second_alt),
            built_at=BASE_TIME,
        )
        single_cpu_by_node: dict[str, float] = {}
        single_memory_by_node: dict[str, int] = {}
        for step in placements:
            single_cpu_by_node[step.node_id] = single_cpu_by_node.get(step.node_id, 0.0) + step.cpu_cores
            single_memory_by_node[step.node_id] = single_memory_by_node.get(step.node_id, 0) + step.memory_mb
        nodes = tuple(
            node.model_copy(
                update={
                    "capacity": node.capacity.model_copy(
                        update={
                            "cpu_cores": max(
                                0.1,
                                single_cpu_by_node.get(node.node_id, node.capacity.cpu_cores) * 1.5,
                            ),
                            "memory_mb": max(
                                64,
                                int(single_memory_by_node.get(node.node_id, node.capacity.memory_mb) * 1.5),
                            ),
                            "gpu_memory_mb": node.capacity.gpu_memory_mb,
                        }
                    )
                }
            )
            for node in self.deployment.nodes.values()
        )
        deployment = DeploymentGraph(
            nodes=nodes,
            sources=self.deployment.sources.values(),
            links=self.deployment.links,
        )
        result = BoundedLabelPlanner(
            provider_registry=self.providers,
            artifact_catalog=self.artifacts,
            deployment=deployment,
            config=BeamSearchConfig(run_oracle=False),
        ).search(graph, (self.demand, second_demand), now=BASE_TIME)
        self.assertIsNone(result.selected)
        self.assertTrue(
            any(
                record.code == PruneCode.CAPACITY_EXCEEDED
                for boundary in result.trace.boundaries
                for record in boundary.pruning_records
            )
        )


if __name__ == "__main__":
    unittest.main()
