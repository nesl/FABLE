from __future__ import annotations

import unittest
from pathlib import Path

from fable.common.examples import BASE_TIME
from fable.planning import (
    BeamSearchConfig,
    BoundedLabelPlanner,
    DemandCompileContext,
    DemandCompiler,
    PhysicalAlternativeGraph,
    PhysicalAlternativeGraphBuilder,
    default_predicate_registry,
)
from fable.planning.testing import (
    fake_artifact_catalog,
    fake_deployment,
    fake_follow_frontier,
    fake_provider_registry,
)
from fable.spatial import SiteSensorTransitionModel, SpatialObservation, SpatialSensorBindings

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "evaluation" / "labels" / "site_sensor_transition_model_2024_2025.json"


class SpatialDemandPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.deployment = fake_deployment()
        self.providers = fake_provider_registry()
        self.artifacts = fake_artifact_catalog()
        self.spatial_model = SiteSensorTransitionModel.from_json(MODEL_PATH)
        self.bindings = SpatialSensorBindings(
            source_ids_by_sensor={
                "orin_6": ("camera_mobile",),
                "orin_5": ("camera_downstream",),
            },
            node_ids_by_sensor={
                "orin_6": ("sensor_a",),
                "orin_5": ("sensor_b",),
            },
        )

    def _demand(self):
        runtime, hypothesis, frontier = fake_follow_frontier()
        node_id = runtime.graph.nodes_by_key["follower_follows"].node_id
        compiler = DemandCompiler(
            predicate_registry=default_predicate_registry(),
            deployment=self.deployment,
            spatial_model=self.spatial_model,
            spatial_bindings=self.bindings,
        )
        demands = compiler.compile_frontier(
            graph=runtime.graph,
            hypothesis=hypothesis,
            frontier=frontier,
            context=DemandCompileContext(
                eligible_source_ids_by_node={
                    node_id: ("camera_mobile", "camera_downstream")
                },
                spatial_observation_by_node={
                    node_id: SpatialObservation(
                        current_sensor_id="orin_6",
                        observed_heading="SW",
                    )
                },
            ),
        )
        return demands[0]

    def test_spatial_prediction_becomes_soft_source_preference(self) -> None:
        demand = self._demand()
        self.assertIsNotNone(demand.spatial_prediction_id)
        self.assertEqual(len(demand.source_preferences), 1)
        self.assertEqual(demand.source_preferences[0].source_id, "camera_downstream")
        self.assertEqual(demand.source_preferences[0].priority_rank, 1)

    def test_alternatives_keep_fallbacks_but_annotate_spatial_penalty(self) -> None:
        demand = self._demand()
        graph = PhysicalAlternativeGraphBuilder(
            provider_registry=self.providers,
            artifact_catalog=self.artifacts,
            deployment=self.deployment,
        ).build((demand,), now=BASE_TIME)
        preferred = [
            alternative
            for alternative in graph.alternatives
            if any(
                item.source_id == "camera_downstream"
                for item in alternative.external_inputs
            )
        ]
        fallback = [
            alternative
            for alternative in graph.alternatives
            if alternative.spatial_preference_penalty == 1000
        ]
        self.assertTrue(preferred)
        self.assertTrue(fallback)
        self.assertLess(
            min(item.spatial_preference_penalty for item in preferred),
            min(item.spatial_preference_penalty for item in fallback),
        )

    def test_beam_search_uses_spatial_penalty_before_soft_cost_tie_breaks(self) -> None:
        demand = self._demand()
        original = PhysicalAlternativeGraphBuilder(
            provider_registry=self.providers,
            artifact_catalog=self.artifacts,
            deployment=self.deployment,
        ).build((demand,), now=BASE_TIME)
        base = original.alternatives[0]
        cheap_unpredicted = base.model_copy(
            update={
                "alternative_id": "alt_spatial_unpredicted",
                "estimated_completion_ms": 1,
                "spatial_preference_penalty": 1000,
                "spatial_preference_reason": "unpredicted fallback",
            }
        )
        slower_predicted = base.model_copy(
            update={
                "alternative_id": "alt_spatial_predicted",
                "estimated_completion_ms": 25,
                "spatial_preference_penalty": 0,
                "spatial_preference_reason": "predicted next sensor",
            }
        )
        graph = PhysicalAlternativeGraph(
            graph_id="physical_graph_spatial_preference",
            checkpoint_ids=original.checkpoint_ids,
            demand_ids=original.demand_ids,
            nodes=original.nodes,
            edges=original.edges,
            alternatives=(cheap_unpredicted, slower_predicted),
            built_at=BASE_TIME,
        )
        result = BoundedLabelPlanner(
            provider_registry=self.providers,
            artifact_catalog=self.artifacts,
            deployment=self.deployment,
            config=BeamSearchConfig(beam_width=2, fallback_count=1, run_oracle=False),
        ).search(graph, (demand,), now=BASE_TIME)
        self.assertIsNotNone(result.selected)
        self.assertEqual(
            result.selected.selected_alternative_ids,
            ("alt_spatial_predicted",),
        )
        self.assertEqual(result.selected.spatial_preference_penalty, 0)


if __name__ == "__main__":
    unittest.main()
