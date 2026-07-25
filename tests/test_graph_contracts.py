from __future__ import annotations

import unittest

from fable.common.examples import convoy_graph, convoy_graph_draft, robbery_graph
from fable.common.graph import finalize_semantic_graph
from fable.common.serialization import load_versioned


class GraphContractTests(unittest.TestCase):
    def test_convoy_and_robbery_graphs_validate_and_round_trip(self) -> None:
        for graph in (convoy_graph(), robbery_graph()):
            loaded = load_versioned(graph.model_dump_json())
            self.assertEqual(loaded, graph)
            self.assertTrue(graph.graph_hash.startswith("sha256:"))

    def test_identical_graph_input_produces_identical_ids(self) -> None:
        left = finalize_semantic_graph(convoy_graph_draft())
        right = finalize_semantic_graph(convoy_graph_draft())
        self.assertEqual(left.graph_hash, right.graph_hash)
        self.assertEqual(left.graph_id, right.graph_id)
        self.assertEqual(
            [node.node_id for node in left.nodes],
            [node.node_id for node in right.nodes],
        )

    def test_authored_node_order_does_not_change_graph_identity(self) -> None:
        original = convoy_graph_draft()
        reordered = original.model_copy(update={"nodes": tuple(reversed(original.nodes))})
        left = finalize_semantic_graph(original)
        right = finalize_semantic_graph(reordered)
        self.assertEqual(left.graph_hash, right.graph_hash)
        self.assertEqual(left.graph_id, right.graph_id)


if __name__ == "__main__":
    unittest.main()
