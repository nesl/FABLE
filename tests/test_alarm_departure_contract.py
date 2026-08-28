from __future__ import annotations

import yaml

from fable.semantic.definitions.multimodal import alarm_departure_graph


def test_alarm_departure_graph_uses_only_catalog_valid_vehicle_exit() -> None:
    graph = alarm_departure_graph()
    predicates = {
        node.predicate.predicate_id: tuple(
            role.role_name for role in node.predicate.roles
        )
        for node in graph.nodes
        if node.predicate is not None
    }
    assert predicates["EXITS"] == ("vehicle",)
    assert "DEPARTURE_OR_ESCAPE" not in predicates

    raw = yaml.safe_load(open("fable/catalog/default_predicates.yaml", encoding="utf-8"))
    catalog = raw["predicates"] if isinstance(raw, dict) else raw
    exits = next(row for row in catalog if row["predicate_id"] == "EXITS")
    assert tuple(row["role_name"] for row in exits["roles"]) == ("vehicle",)
