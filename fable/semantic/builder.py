"""Authored semantic-graph builder and structural compiler for FABLE Phase 1.

The Phase-0 graph draft objects are intentionally low-level and serialization
oriented.  This module adds a small authoring API for the constructs used by the
semantic runtime while preserving the finalized shared graph as the canonical
representation.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from fable.common.enums import (
    GraphEdgeKind,
    GraphNodeKind,
    ResultKind,
    TemporalGuardKind,
)
from fable.common.graph import (
    GraphEdgeDraft,
    GraphNodeDraft,
    SemanticGraphDraft,
    TemporalGuardDraft,
    finalize_semantic_graph,
)
from fable.common.schemas import PredicateRole, RoleDefinition, SemanticGraph, SemanticPredicate


class GraphCompileError(ValueError):
    """Raised when an authored graph is well-typed as data but structurally invalid."""


@dataclass(frozen=True)
class PredicateRoleSpec:
    """Convenient authoring form for one provider-independent predicate role."""

    role_name: str
    variable: str
    entity_type: str


class AuthoredGraphBuilder:
    """Small deterministic DSL for authored FABLE event graphs.

    Nodes are referred to by authored keys.  Composite helper methods add the
    corresponding child and temporal edges, but never choose providers or
    execution locations.
    """

    def __init__(
        self,
        *,
        namespace: str,
        name: str,
        description: str | None = None,
        graph_version: int = 1,
    ) -> None:
        self.namespace = namespace
        self.name = name
        self.description = description
        self.graph_version = graph_version
        self._roles: dict[str, RoleDefinition] = {}
        self._nodes: dict[str, GraphNodeDraft] = {}
        self._edges: list[GraphEdgeDraft] = []
        self._guards: dict[str, TemporalGuardDraft] = {}
        self._root_key: str | None = None
        self._variant_ids: set[str] = set()

    def role(
        self,
        role_name: str,
        entity_type: str,
        *,
        cardinality_min: int = 1,
        cardinality_max: int | None = 1,
        distinct_from: Iterable[str] = (),
    ) -> "AuthoredGraphBuilder":
        if role_name in self._roles:
            raise GraphCompileError(f"duplicate role {role_name!r}")
        self._roles[role_name] = RoleDefinition(
            role_name=role_name,
            entity_type=entity_type,
            cardinality_min=cardinality_min,
            cardinality_max=cardinality_max,
            distinct_from=tuple(distinct_from),
        )
        return self

    def variant(self, variant_id: str) -> "AuthoredGraphBuilder":
        if not variant_id:
            raise GraphCompileError("variant_id must be non-empty")
        self._variant_ids.add(variant_id)
        return self

    def root(self, node_key: str) -> "AuthoredGraphBuilder":
        self._require_node(node_key)
        self._root_key = node_key
        return self

    def primitive(
        self,
        key: str,
        *,
        name: str,
        predicate_id: str,
        roles: Sequence[PredicateRoleSpec | tuple[str, str, str]] = (),
        result_kind: ResultKind = ResultKind.INSTANT_MATCH,
        parameters: Mapping[str, object] | None = None,
        checkpoint: bool = False,
        annotations: Mapping[str, object] | None = None,
    ) -> str:
        normalized_roles = tuple(
            role
            if isinstance(role, PredicateRoleSpec)
            else PredicateRoleSpec(*role)
            for role in roles
        )
        for role in normalized_roles:
            if role.variable in self._roles:
                expected = self._roles[role.variable].entity_type
                if expected != role.entity_type:
                    raise GraphCompileError(
                        f"predicate role {role.variable!r} has entity type {role.entity_type!r}; "
                        f"graph role requires {expected!r}"
                    )
        predicate = SemanticPredicate(
            predicate_id=predicate_id,
            roles=tuple(
                PredicateRole(
                    role_name=role.role_name,
                    variable=role.variable,
                    entity_type=role.entity_type,
                )
                for role in normalized_roles
            ),
            parameters=dict(parameters or {}),
            result_kind=result_kind,
        )
        self._add_node(
            GraphNodeDraft(
                key=key,
                kind=GraphNodeKind.PREDICATE,
                name=name,
                predicate=predicate,
                checkpoint_boundary=checkpoint,
                annotations=dict(annotations or {}),
            )
        )
        return key

    def sequence(
        self,
        key: str,
        children: Sequence[str],
        *,
        name: str,
        checkpoint: bool = False,
        annotations: Mapping[str, object] | None = None,
    ) -> str:
        self._require_children(children, minimum=1)
        self._add_node(
            GraphNodeDraft(
                key=key,
                kind=GraphNodeKind.AND,
                name=name,
                checkpoint_boundary=checkpoint,
                annotations=dict(annotations or {}),
            )
        )
        self._add_children(key, children, GraphEdgeKind.CHILD)
        for source, target in zip(children, children[1:]):
            self._edges.append(
                GraphEdgeDraft(
                    source_node_key=source,
                    target_node_key=target,
                    kind=GraphEdgeKind.SEQUENCE,
                )
            )
        return key

    def and_group(
        self,
        key: str,
        children: Sequence[str],
        *,
        name: str,
        checkpoint: bool = False,
        annotations: Mapping[str, object] | None = None,
    ) -> str:
        self._require_children(children, minimum=1)
        self._add_node(
            GraphNodeDraft(
                key=key,
                kind=GraphNodeKind.AND,
                name=name,
                checkpoint_boundary=checkpoint,
                annotations=dict(annotations or {}),
            )
        )
        self._add_children(key, children, GraphEdgeKind.CHILD)
        return key

    def or_group(
        self,
        key: str,
        branches: Mapping[str, str],
        *,
        name: str,
        checkpoint: bool = True,
        annotations: Mapping[str, object] | None = None,
    ) -> str:
        if len(branches) < 2:
            raise GraphCompileError("OR requires at least two authored branches")
        self._require_children(tuple(branches.values()), minimum=2)
        self._add_node(
            GraphNodeDraft(
                key=key,
                kind=GraphNodeKind.OR,
                name=name,
                checkpoint_boundary=checkpoint,
                annotations=dict(annotations or {}),
            )
        )
        for label, child in branches.items():
            if not label:
                raise GraphCompileError("OR branch labels must be non-empty")
            self._edges.append(
                GraphEdgeDraft(
                    source_node_key=key,
                    target_node_key=child,
                    kind=GraphEdgeKind.ALTERNATIVE,
                    branch_label=label,
                )
            )
            self._variant_ids.add(label)
        return key

    def k_of_n(
        self,
        key: str,
        children: Sequence[str],
        *,
        k: int,
        name: str,
        checkpoint: bool = True,
        annotations: Mapping[str, object] | None = None,
    ) -> str:
        self._require_children(children, minimum=1)
        if k < 1 or k > len(children):
            raise GraphCompileError("K-of-N requires 1 <= k <= number of children")
        self._add_node(
            GraphNodeDraft(
                key=key,
                kind=GraphNodeKind.K_OF_N,
                name=name,
                k=k,
                checkpoint_boundary=checkpoint,
                annotations=dict(annotations or {}),
            )
        )
        self._add_children(key, children, GraphEdgeKind.CHILD)
        return key

    def duration(
        self,
        key: str,
        child: str,
        *,
        minimum_ms: int,
        name: str,
        checkpoint: bool = True,
        required_source_ids: Iterable[str] = (),
        annotations: Mapping[str, object] | None = None,
    ) -> str:
        self._require_node(child)
        guard_key = f"{key}__duration"
        self._add_node(
            GraphNodeDraft(
                key=key,
                kind=GraphNodeKind.DURATION,
                name=name,
                checkpoint_boundary=checkpoint,
                annotations=dict(annotations or {}),
            )
        )
        self._guards[guard_key] = TemporalGuardDraft(
            key=guard_key,
            kind=TemporalGuardKind.DURATION,
            source_node_keys=(child,),
            target_node_key=key,
            minimum_ms=minimum_ms,
            required_source_ids=tuple(required_source_ids),
        )
        self._edges.append(
            GraphEdgeDraft(
                source_node_key=key,
                target_node_key=child,
                kind=GraphEdgeKind.CHILD,
                temporal_guard_keys=(guard_key,),
            )
        )
        return key

    def absent(
        self,
        key: str,
        child: str,
        *,
        window_ms: int,
        required_source_ids: Iterable[str],
        name: str,
        checkpoint: bool = True,
        annotations: Mapping[str, object] | None = None,
    ) -> str:
        self._require_node(child)
        sources = tuple(required_source_ids)
        if not sources:
            raise GraphCompileError("ABSENT requires at least one coverage source")
        guard_key = f"{key}__absence"
        self._add_node(
            GraphNodeDraft(
                key=key,
                kind=GraphNodeKind.ABSENT,
                name=name,
                checkpoint_boundary=checkpoint,
                annotations=dict(annotations or {}),
            )
        )
        self._guards[guard_key] = TemporalGuardDraft(
            key=guard_key,
            kind=TemporalGuardKind.ABSENCE_WINDOW,
            source_node_keys=(child,),
            target_node_key=key,
            minimum_ms=window_ms,
            required_source_ids=sources,
        )
        self._edges.append(
            GraphEdgeDraft(
                source_node_key=key,
                target_node_key=child,
                kind=GraphEdgeKind.CHILD,
                temporal_guard_keys=(guard_key,),
            )
        )
        return key

    def within(
        self,
        key: str,
        child: str,
        *,
        after: Sequence[str],
        maximum_ms: int,
        name: str,
        minimum_ms: int | None = None,
        checkpoint: bool = True,
        annotations: Mapping[str, object] | None = None,
    ) -> str:
        self._require_node(child)
        self._require_children(after, minimum=1)
        guard_key = f"{key}__within"
        self._add_node(
            GraphNodeDraft(
                key=key,
                kind=GraphNodeKind.WITHIN,
                name=name,
                checkpoint_boundary=checkpoint,
                annotations=dict(annotations or {}),
            )
        )
        self._guards[guard_key] = TemporalGuardDraft(
            key=guard_key,
            kind=TemporalGuardKind.WITHIN,
            source_node_keys=tuple(after),
            target_node_key=child,
            minimum_ms=minimum_ms,
            maximum_ms=maximum_ms,
        )
        self._edges.append(
            GraphEdgeDraft(
                source_node_key=key,
                target_node_key=child,
                kind=GraphEdgeKind.CHILD,
                temporal_guard_keys=(guard_key,),
            )
        )
        return key

    def precedes(
        self,
        source: str,
        target: str,
        *,
        guard_key: str | None = None,
        minimum_ms: int | None = None,
        maximum_ms: int | None = None,
    ) -> "AuthoredGraphBuilder":
        self._require_node(source)
        self._require_node(target)
        keys: tuple[str, ...] = ()
        if minimum_ms is not None or maximum_ms is not None:
            key = guard_key or f"{source}__to__{target}__within"
            if key in self._guards:
                raise GraphCompileError(f"duplicate temporal guard {key!r}")
            self._guards[key] = TemporalGuardDraft(
                key=key,
                kind=TemporalGuardKind.WITHIN,
                source_node_keys=(source,),
                target_node_key=target,
                minimum_ms=minimum_ms,
                maximum_ms=maximum_ms,
            )
            keys = (key,)
        self._edges.append(
            GraphEdgeDraft(
                source_node_key=source,
                target_node_key=target,
                kind=GraphEdgeKind.SEQUENCE,
                temporal_guard_keys=keys,
            )
        )
        return self

    def draft(self) -> SemanticGraphDraft:
        if self._root_key is None:
            raise GraphCompileError("graph root has not been selected")
        return SemanticGraphDraft(
            namespace=self.namespace,
            graph_version=self.graph_version,
            name=self.name,
            description=self.description,
            root_node_key=self._root_key,
            roles=tuple(sorted(self._roles.values(), key=lambda item: item.role_name)),
            nodes=tuple(sorted(self._nodes.values(), key=lambda item: item.key)),
            edges=tuple(self._edges),
            temporal_guards=tuple(sorted(self._guards.values(), key=lambda item: item.key)),
            authored_variant_ids=tuple(sorted(self._variant_ids)),
        )

    def compile(self) -> SemanticGraph:
        return compile_authored_graph(self.draft())

    def _add_node(self, node: GraphNodeDraft) -> None:
        if node.key in self._nodes:
            raise GraphCompileError(f"duplicate node {node.key!r}")
        self._nodes[node.key] = node

    def _add_children(self, parent: str, children: Sequence[str], kind: GraphEdgeKind) -> None:
        for child in children:
            self._edges.append(
                GraphEdgeDraft(
                    source_node_key=parent,
                    target_node_key=child,
                    kind=kind,
                )
            )

    def _require_node(self, key: str) -> None:
        if key not in self._nodes:
            raise GraphCompileError(f"unknown node {key!r}")

    def _require_children(self, keys: Sequence[str], *, minimum: int) -> None:
        if len(keys) < minimum:
            raise GraphCompileError(f"expected at least {minimum} child nodes")
        for key in keys:
            self._require_node(key)


def compile_authored_graph(draft: SemanticGraphDraft) -> SemanticGraph:
    graph = finalize_semantic_graph(draft)
    validate_semantic_graph_structure(graph)
    return graph


def validate_semantic_graph_structure(graph: SemanticGraph) -> None:
    """Validate semantic constraints that do not belong in the wire schema."""

    node_by_id = {node.node_id: node for node in graph.nodes}
    children: dict[str, list[str]] = defaultdict(list)
    alternatives: dict[str, list[str]] = defaultdict(list)
    dependency_edges: list[tuple[str, str]] = []

    for edge in graph.edges:
        if edge.kind == GraphEdgeKind.CHILD:
            children[edge.source_node_id].append(edge.target_node_id)
        elif edge.kind == GraphEdgeKind.ALTERNATIVE:
            alternatives[edge.source_node_id].append(edge.target_node_id)
        elif edge.kind in (GraphEdgeKind.SEQUENCE, GraphEdgeKind.DEPENDS_ON):
            dependency_edges.append((edge.source_node_id, edge.target_node_id))

    for node in graph.nodes:
        ordinary = children.get(node.node_id, [])
        alts = alternatives.get(node.node_id, [])
        if node.kind == GraphNodeKind.PREDICATE and (ordinary or alts):
            raise GraphCompileError(f"predicate node {node.authored_key!r} cannot own children")
        if node.kind == GraphNodeKind.OR:
            if len(alts) < 2 or ordinary:
                raise GraphCompileError(f"OR node {node.authored_key!r} requires >=2 alternative children")
            labels = [
                edge.branch_label
                for edge in graph.edges
                if edge.source_node_id == node.node_id and edge.kind == GraphEdgeKind.ALTERNATIVE
            ]
            if any(label is None for label in labels) or len(labels) != len(set(labels)):
                raise GraphCompileError(f"OR node {node.authored_key!r} requires unique branch labels")
        elif alts:
            raise GraphCompileError(
                f"non-OR node {node.authored_key!r} cannot own ALTERNATIVE edges"
            )
        if node.kind == GraphNodeKind.AND and not ordinary:
            raise GraphCompileError(f"AND node {node.authored_key!r} requires children")
        if node.kind == GraphNodeKind.K_OF_N:
            if not ordinary or node.k is None or node.k > len(ordinary):
                raise GraphCompileError(f"invalid K-of-N node {node.authored_key!r}")
        if node.kind in (GraphNodeKind.DURATION, GraphNodeKind.ABSENT, GraphNodeKind.WITHIN):
            if len(ordinary) != 1:
                raise GraphCompileError(
                    f"{node.kind.value} node {node.authored_key!r} requires exactly one child"
                )

    _assert_acyclic(node_by_id, dependency_edges, "sequence/dependency")
    hierarchy_edges = [
        (edge.source_node_id, edge.target_node_id)
        for edge in graph.edges
        if edge.kind in (GraphEdgeKind.CHILD, GraphEdgeKind.ALTERNATIVE)
    ]
    _assert_acyclic(node_by_id, hierarchy_edges, "composite hierarchy")

    role_by_name = {role.role_name: role for role in graph.roles}
    for role in graph.roles:
        for other in role.distinct_from:
            if other not in role_by_name:
                raise GraphCompileError(
                    f"role {role.role_name!r} is distinct from unknown role {other!r}"
                )
            if other == role.role_name:
                raise GraphCompileError(f"role {role.role_name!r} cannot be distinct from itself")


def _assert_acyclic(
    node_by_id: Mapping[str, object],
    edges: Sequence[tuple[str, str]],
    label: str,
) -> None:
    outgoing: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_by_id}
    for source, target in edges:
        outgoing[source].append(target)
        indegree[target] += 1
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for target in outgoing.get(current, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(node_by_id):
        raise GraphCompileError(f"{label} contains a cycle")
