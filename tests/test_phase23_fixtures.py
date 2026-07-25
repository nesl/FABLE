from __future__ import annotations

import unittest
from pathlib import Path

from fable.common.schemas import ArtifactRef, PredicateDemand
from fable.planning.models import DeploymentNode, PhysicalAlternativeGraph


FIXTURES = Path(__file__).parent / "phase23_fixtures"


class Phase23FixtureTests(unittest.TestCase):
    def test_follow_demand_fixture_validates(self) -> None:
        demand = PredicateDemand.model_validate_json(
            (FIXTURES / "follows_demand.json").read_text(encoding="utf-8")
        )
        self.assertEqual(demand.semantic_predicate.predicate_id, "FOLLOWS")

    def test_physical_graph_fixture_validates(self) -> None:
        graph = PhysicalAlternativeGraph.model_validate_json(
            (FIXTURES / "physical_alternative_graph.json").read_text(encoding="utf-8")
        )
        self.assertTrue(graph.alternatives)
        self.assertEqual(set(graph.demand_ids), {alt.demand_id for alt in graph.alternatives})

    def test_artifact_and_deployment_fixtures_validate(self) -> None:
        import json

        artifacts = json.loads((FIXTURES / "artifacts.json").read_text(encoding="utf-8"))
        nodes = json.loads((FIXTURES / "deployment_nodes.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len([ArtifactRef.model_validate(item) for item in artifacts]), 4)
        self.assertGreaterEqual(len([DeploymentNode.model_validate(item) for item in nodes]), 3)


if __name__ == "__main__":
    unittest.main()
