"""Deterministic planner search cases for tests and controlled evaluations."""

from __future__ import annotations

from fable.common.examples import BASE_TIME
from fable.common.ids import deterministic_id
from fable.common.schemas import PredicateDemand

from fable.planning.alternative_graph import PhysicalAlternativeGraphBuilder
from fable.planning.models import PhysicalAlternativeGraph
from fable.planning.testing import (
    fake_artifact_catalog,
    fake_deployment,
    fake_follow_demand,
    fake_provider_registry,
)


def fake_follow_alternative_graph(
    *,
    provider_registry=None,
    artifact_catalog=None,
    deployment=None,
    demand: PredicateDemand | None = None,
) -> tuple[PhysicalAlternativeGraph, PredicateDemand]:
    provider_registry = provider_registry or fake_provider_registry()
    artifact_catalog = artifact_catalog or fake_artifact_catalog()
    deployment = deployment or fake_deployment()
    demand = demand or fake_follow_demand()
    graph = PhysicalAlternativeGraphBuilder(
        provider_registry=provider_registry,
        artifact_catalog=artifact_catalog,
        deployment=deployment,
    ).build((demand,), now=BASE_TIME)
    return graph, demand


def continuation_trap_graph(
    *,
    provider_registry=None,
    artifact_catalog=None,
    deployment=None,
    demand: PredicateDemand | None = None,
) -> tuple[PhysicalAlternativeGraph, PredicateDemand]:
    """One cheap narrow representation and one costlier usable continuation.

    Beam width 1 greedily retains the cheap pair-trajectory plan. A downstream
    route/timing consumer can use only a compact ``track_summary.v1`` artifact,
    so beam width 2 is needed to retain the checkpoint-compatible plan.
    """

    graph, demand = fake_follow_alternative_graph(
        provider_registry=provider_registry,
        artifact_catalog=artifact_catalog,
        deployment=deployment,
        demand=demand,
    )
    local = min(
        (
            item
            for item in graph.alternatives
            if item.chain_id == "follows_local_tracks"
        ),
        key=lambda item: (
            item.estimated_completion_ms,
            item.estimated_transfer_bytes,
            item.alternative_id,
        ),
    ).model_copy(
        update={
            "alternative_id": "alt_continuation_trap_cheap_pair",
            "estimated_completion_ms": 10,
            "estimated_transfer_bytes": 0,
        }
    )
    rich = min(
        (
            item
            for item in graph.alternatives
            if item.chain_id == "follows_local_from_retained_detections"
            and "track_summary.v1" in item.continuation_output_types
        ),
        key=lambda item: (
            item.estimated_completion_ms,
            item.estimated_transfer_bytes,
            item.alternative_id,
        ),
    ).model_copy(
        update={
            "alternative_id": "alt_continuation_trap_rich_summary",
            "estimated_completion_ms": 20,
            "estimated_transfer_bytes": 0,
        }
    )
    graph_id = deterministic_id(
        "physical_graph",
        {
            "case": "continuation_trap",
            "demand_id": demand.demand_id,
            "alternatives": (local.alternative_id, rich.alternative_id),
        },
        length=32,
    )
    selected_node_ids = set(local.graph_node_ids) | set(rich.graph_node_ids)
    selected_edge_ids = set(local.graph_edge_ids) | set(rich.graph_edge_ids)
    return (
        PhysicalAlternativeGraph(
            graph_id=graph_id,
            checkpoint_ids=graph.checkpoint_ids,
            demand_ids=graph.demand_ids,
            nodes=tuple(node for node in graph.nodes if node.node_id in selected_node_ids),
            edges=tuple(edge for edge in graph.edges if edge.edge_id in selected_edge_ids),
            alternatives=(local, rich),
            pruned=(),
            built_at=BASE_TIME,
        ),
        demand,
    )
