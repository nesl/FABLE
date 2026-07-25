from __future__ import annotations

import unittest

from fable.common.enums import GraphNodeKind
from fable.semantic import all_constructs_graph, repeated_visit_graph


class Phase1BuilderTests(unittest.TestCase):
    def test_all_minimum_constructs_compile(self) -> None:
        graph = all_constructs_graph()
        kinds = {node.kind for node in graph.nodes}
        self.assertTrue(
            {
                GraphNodeKind.PREDICATE,
                GraphNodeKind.AND,
                GraphNodeKind.OR,
                GraphNodeKind.K_OF_N,
                GraphNodeKind.DURATION,
                GraphNodeKind.ABSENT,
                GraphNodeKind.WITHIN,
            }.issubset(kinds)
        )
        self.assertTrue(graph.graph_hash.startswith("sha256:"))

    def test_repeated_visit_uses_one_shared_vehicle_role(self) -> None:
        graph = repeated_visit_graph(return_window_ms=120_000)
        self.assertEqual([role.role_name for role in graph.roles], ["vehicle"])
        enters = [
            node
            for node in graph.nodes
            if node.predicate is not None and node.predicate.predicate_id == "ENTERS"
        ]
        self.assertEqual(len(enters), 2)
        for node in enters:
            self.assertEqual(node.predicate.roles[0].variable, "vehicle")


if __name__ == "__main__":
    unittest.main()
