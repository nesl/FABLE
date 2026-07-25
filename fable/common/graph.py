"""Deterministic finalization of authored semantic graph drafts."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, model_validator

from .base import FableModel, JSONValue, VersionedModel
from .enums import GraphEdgeKind, GraphNodeKind, TemporalGuardKind
from .ids import deterministic_id, graph_edge_id, graph_node_id, sha256_hex, temporal_guard_id
from .schemas import (
    GraphEdge,
    GraphNode,
    RoleDefinition,
    SemanticGraph,
    SemanticPredicate,
    TemporalGuard,
)


class GraphNodeDraft(FableModel):
    key: str = Field(min_length=1)
    kind: GraphNodeKind
    name: str = Field(min_length=1)
    predicate: SemanticPredicate | None = None
    k: int | None = Field(default=None, ge=1)
    checkpoint_boundary: bool = False
    annotations: dict[str, JSONValue] = Field(default_factory=dict)


class TemporalGuardDraft(FableModel):
    key: str = Field(min_length=1)
    kind: TemporalGuardKind
    source_node_keys: tuple[str, ...]
    target_node_key: str | None = None
    minimum_ms: int | None = Field(default=None, ge=0)
    maximum_ms: int | None = Field(default=None, ge=0)
    count: int | None = Field(default=None, ge=1)
    required_source_ids: tuple[str, ...] = ()


class GraphEdgeDraft(FableModel):
    source_node_key: str = Field(min_length=1)
    target_node_key: str = Field(min_length=1)
    kind: GraphEdgeKind
    temporal_guard_keys: tuple[str, ...] = ()
    branch_label: str | None = None


class SemanticGraphDraft(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.semantic_graph_draft.v1"
    schema_version: Literal["fable.semantic_graph_draft.v1"] = SCHEMA_VERSION
    namespace: str = Field(min_length=1)
    graph_version: int = Field(default=1, ge=1)
    name: str = Field(min_length=1)
    description: str | None = None
    root_node_key: str = Field(min_length=1)
    roles: tuple[RoleDefinition, ...] = ()
    nodes: tuple[GraphNodeDraft, ...]
    edges: tuple[GraphEdgeDraft, ...] = ()
    temporal_guards: tuple[TemporalGuardDraft, ...] = ()
    authored_variant_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_keys(self) -> "SemanticGraphDraft":
        node_keys = [node.key for node in self.nodes]
        if len(node_keys) != len(set(node_keys)):
            raise ValueError("draft node keys must be unique")
        node_set = set(node_keys)
        if self.root_node_key not in node_set:
            raise ValueError("root_node_key does not reference a draft node")
        guard_keys = [guard.key for guard in self.temporal_guards]
        if len(guard_keys) != len(set(guard_keys)):
            raise ValueError("draft guard keys must be unique")
        guard_set = set(guard_keys)
        for edge in self.edges:
            if edge.source_node_key not in node_set or edge.target_node_key not in node_set:
                raise ValueError("draft edge references an unknown node key")
            if set(edge.temporal_guard_keys) - guard_set:
                raise ValueError("draft edge references an unknown temporal guard key")
        for guard in self.temporal_guards:
            if set(guard.source_node_keys) - node_set:
                raise ValueError("draft guard references an unknown source node key")
            if guard.target_node_key and guard.target_node_key not in node_set:
                raise ValueError("draft guard references an unknown target node key")
        return self


def _node_hash_payload(node: GraphNodeDraft) -> dict:
    return node.model_dump(mode="python", exclude={"key"}, exclude_none=True)


def _guard_hash_payload(guard: TemporalGuardDraft) -> dict:
    return guard.model_dump(mode="python", exclude={"key"}, exclude_none=True)


def finalize_semantic_graph(draft: SemanticGraphDraft) -> SemanticGraph:
    """Assign stable node/guard/edge identifiers and an immutable graph hash."""

    sorted_nodes = sorted(draft.nodes, key=lambda item: item.key)
    node_ids = {
        node.key: graph_node_id(draft.namespace, node.key, _node_hash_payload(node))
        for node in sorted_nodes
    }

    sorted_guards = sorted(draft.temporal_guards, key=lambda item: item.key)
    guard_ids = {
        guard.key: temporal_guard_id(draft.namespace, guard.key, _guard_hash_payload(guard))
        for guard in sorted_guards
    }

    nodes = tuple(
        GraphNode(
            node_id=node_ids[node.key],
            authored_key=node.key,
            kind=node.kind,
            name=node.name,
            predicate=node.predicate,
            k=node.k,
            checkpoint_boundary=node.checkpoint_boundary,
            annotations=node.annotations,
        )
        for node in sorted_nodes
    )

    guards = tuple(
        TemporalGuard(
            guard_id=guard_ids[guard.key],
            kind=guard.kind,
            source_node_ids=tuple(node_ids[key] for key in guard.source_node_keys),
            target_node_id=node_ids[guard.target_node_key] if guard.target_node_key else None,
            minimum_ms=guard.minimum_ms,
            maximum_ms=guard.maximum_ms,
            count=guard.count,
            required_source_ids=guard.required_source_ids,
        )
        for guard in sorted_guards
    )

    raw_edges = []
    for edge in draft.edges:
        payload = {
            "source_node_id": node_ids[edge.source_node_key],
            "target_node_id": node_ids[edge.target_node_key],
            "kind": edge.kind,
            "temporal_guard_ids": tuple(guard_ids[key] for key in edge.temporal_guard_keys),
            "branch_label": edge.branch_label,
        }
        raw_edges.append(
            GraphEdge(
                edge_id=graph_edge_id(draft.namespace, payload),
                **payload,
            )
        )
    edges = tuple(sorted(raw_edges, key=lambda item: item.edge_id))

    graph_content = {
        "graph_version": draft.graph_version,
        "namespace": draft.namespace,
        "name": draft.name,
        "description": draft.description,
        "root_node_id": node_ids[draft.root_node_key],
        "roles": draft.roles,
        "nodes": nodes,
        "edges": edges,
        "temporal_guards": guards,
        "authored_variant_ids": sorted(draft.authored_variant_ids),
    }
    digest = sha256_hex(graph_content)
    return SemanticGraph(
        graph_id=deterministic_id("graph", {"namespace": draft.namespace, "digest": digest}, length=32),
        graph_hash=f"sha256:{digest}",
        graph_version=draft.graph_version,
        name=draft.name,
        description=draft.description,
        root_node_id=node_ids[draft.root_node_key],
        roles=draft.roles,
        nodes=nodes,
        edges=edges,
        temporal_guards=guards,
        authored_variant_ids=tuple(sorted(draft.authored_variant_ids)),
    )
