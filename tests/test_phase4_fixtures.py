from __future__ import annotations

import unittest
from pathlib import Path

from fable.common.schemas import PredicateDemand
from fable.planning import PhysicalAlternativeGraph
from fable.planning.search_models import PlanSearchResult


FIXTURE_DIR = Path(__file__).parent / "phase4_fixtures"


class Phase4FixtureTests(unittest.TestCase):
    def test_continuation_trap_graph_and_demand_validate(self) -> None:
        graph = PhysicalAlternativeGraph.model_validate_json(
            (FIXTURE_DIR / "continuation_trap_graph.json").read_text()
        )
        demand = PredicateDemand.model_validate_json(
            (FIXTURE_DIR / "continuation_trap_demand.json").read_text()
        )
        self.assertEqual(graph.demand_ids, (demand.demand_id,))
        self.assertEqual(len(graph.alternatives), 2)

    def test_beam_results_validate_and_capture_expected_gap(self) -> None:
        greedy = PlanSearchResult.model_validate_json(
            (FIXTURE_DIR / "beam_width_1_result.json").read_text()
        )
        wider = PlanSearchResult.model_validate_json(
            (FIXTURE_DIR / "beam_width_2_result.json").read_text()
        )
        self.assertIsNone(greedy.selected)
        self.assertEqual(greedy.trace.oracle.status.value, "GAP")
        self.assertIsNotNone(wider.selected)
        self.assertEqual(wider.trace.oracle.status.value, "MATCHED")


if __name__ == "__main__":
    unittest.main()
