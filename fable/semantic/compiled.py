"""Indexed view of a finalized FABLE semantic graph."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

from fable.common.enums import GraphEdgeKind, GraphNodeKind
from fable.common.schemas import GraphEdge, GraphNode, SemanticGraph, TemporalGuard


@dataclass(frozen=True)
class CompiledSemanticGraph:
    graph: SemanticGraph

    @cached_property
    def nodes_by_id(self) -> dict[str, GraphNode]:
        return {node.node_id: node for node in self.graph.nodes}

    @cached_property
    def nodes_by_key(self) -> dict[str, GraphNode]:
        return {node.authored_key: node for node in self.graph.nodes}

    @cached_property
    def guards_by_id(self) -> dict[str, TemporalGuard]:
        return {guard.guard_id: guard for guard in self.graph.temporal_guards}

    @cached_property
    def outgoing(self) -> dict[str, tuple[GraphEdge, ...]]:
        grouped: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in self.graph.edges:
            grouped[edge.source_node_id].append(edge)
        return {key: tuple(sorted(value, key=lambda edge: edge.edge_id)) for key, value in grouped.items()}

    @cached_property
    def incoming(self) -> dict[str, tuple[GraphEdge, ...]]:
        grouped: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in self.graph.edges:
            grouped[edge.target_node_id].append(edge)
        return {key: tuple(sorted(value, key=lambda edge: edge.edge_id)) for key, value in grouped.items()}

    def node(self, node_id_or_key: str) -> GraphNode:
        if node_id_or_key in self.nodes_by_id:
            return self.nodes_by_id[node_id_or_key]
        return self.nodes_by_key[node_id_or_key]

    def edges_from(self, node_id: str, *kinds: GraphEdgeKind) -> tuple[GraphEdge, ...]:
        edges = self.outgoing.get(node_id, ())
        return tuple(edge for edge in edges if not kinds or edge.kind in kinds)

    def edges_to(self, node_id: str, *kinds: GraphEdgeKind) -> tuple[GraphEdge, ...]:
        edges = self.incoming.get(node_id, ())
        return tuple(edge for edge in edges if not kinds or edge.kind in kinds)

    def children(self, node_id: str) -> tuple[str, ...]:
        return tuple(
            edge.target_node_id
            for edge in self.edges_from(node_id, GraphEdgeKind.CHILD, GraphEdgeKind.ALTERNATIVE)
        )

    def ordinary_children(self, node_id: str) -> tuple[str, ...]:
        return tuple(edge.target_node_id for edge in self.edges_from(node_id, GraphEdgeKind.CHILD))

    def alternative_children(self, node_id: str) -> tuple[str, ...]:
        return tuple(edge.target_node_id for edge in self.edges_from(node_id, GraphEdgeKind.ALTERNATIVE))

    def parents(self, node_id: str) -> tuple[str, ...]:
        return tuple(
            edge.source_node_id
            for edge in self.edges_to(node_id, GraphEdgeKind.CHILD, GraphEdgeKind.ALTERNATIVE)
        )

    def sequence_predecessors(self, node_id: str) -> tuple[str, ...]:
        return tuple(
            edge.source_node_id
            for edge in self.edges_to(node_id, GraphEdgeKind.SEQUENCE, GraphEdgeKind.DEPENDS_ON)
        )

    def sequence_successors(self, node_id: str) -> tuple[str, ...]:
        return tuple(
            edge.target_node_id
            for edge in self.edges_from(node_id, GraphEdgeKind.SEQUENCE, GraphEdgeKind.DEPENDS_ON)
        )

    def branch_label(self, parent_id: str, child_id: str) -> str | None:
        for edge in self.edges_from(parent_id, GraphEdgeKind.ALTERNATIVE):
            if edge.target_node_id == child_id:
                return edge.branch_label
        return None

    def temporal_guards_for_target(self, node_id: str) -> tuple[TemporalGuard, ...]:
        guard_ids: set[str] = set()
        for edge in self.edges_to(node_id):
            guard_ids.update(edge.temporal_guard_ids)
        for guard in self.graph.temporal_guards:
            if guard.target_node_id == node_id:
                guard_ids.add(guard.guard_id)
        return tuple(self.guards_by_id[guard_id] for guard_id in sorted(guard_ids))

    def temporal_guards_on_parent_child(self, parent_id: str, child_id: str) -> tuple[TemporalGuard, ...]:
        guard_ids: set[str] = set()
        for edge in self.edges_from(parent_id, GraphEdgeKind.CHILD, GraphEdgeKind.ALTERNATIVE):
            if edge.target_node_id == child_id:
                guard_ids.update(edge.temporal_guard_ids)
        return tuple(self.guards_by_id[guard_id] for guard_id in sorted(guard_ids))

    def ancestors(self, node_id: str) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        queue = deque(self.parents(node_id))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            result.append(current)
            queue.extend(self.parents(current))
        return tuple(result)

    def nearest_checkpoint_boundary(self, node_id: str) -> str:
        node = self.nodes_by_id[node_id]
        if node.checkpoint_boundary or node.kind in (
            GraphNodeKind.ABSENT,
            GraphNodeKind.DURATION,
            GraphNodeKind.WITHIN,
        ):
            return node_id
        queue = deque((parent, 1) for parent in self.parents(node_id))
        seen: set[str] = set()
        candidates: list[tuple[int, str]] = []
        while queue:
            current, distance = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            parent = self.nodes_by_id[current]
            if parent.checkpoint_boundary or parent.kind in (
                GraphNodeKind.OR,
                GraphNodeKind.K_OF_N,
                GraphNodeKind.ABSENT,
                GraphNodeKind.DURATION,
                GraphNodeKind.WITHIN,
            ):
                candidates.append((distance, current))
                continue
            queue.extend((ancestor, distance + 1) for ancestor in self.parents(current))
        if not candidates:
            return node_id
        return min(candidates, key=lambda item: (item[0], item[1]))[1]

    def descendants(self, node_id: str) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()
        queue = deque(self.children(node_id))
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            result.append(current)
            queue.extend(self.children(current))
        return tuple(result)

    def branch_subtree(self, or_node_id: str, branch_child_id: str) -> tuple[str, ...]:
        """Return nodes unique to one OR branch before any shared merge point.

        Authored graphs can point several branches at the same subgraph.  A node
        is considered branch-local only when it is reachable from the selected
        branch child and not from another alternative child.
        """

        alternatives = self.alternative_children(or_node_id)
        reachability = {
            child: {child, *self.descendants(child)}
            for child in alternatives
        }
        selected = reachability.get(branch_child_id, {branch_child_id})
        other = set().union(
            *(nodes for child, nodes in reachability.items() if child != branch_child_id)
        ) if len(reachability) > 1 else set()
        return tuple(sorted(selected - other))

    def executable_predicate_nodes(self) -> tuple[str, ...]:
        return tuple(
            node.node_id
            for node in self.graph.nodes
            if node.kind == GraphNodeKind.PREDICATE
        )

    def topological_dependency_order(self) -> tuple[str, ...]:
        indegree = {node.node_id: 0 for node in self.graph.nodes}
        outgoing: dict[str, list[str]] = defaultdict(list)
        for edge in self.graph.edges:
            if edge.kind not in (GraphEdgeKind.SEQUENCE, GraphEdgeKind.DEPENDS_ON):
                continue
            indegree[edge.target_node_id] += 1
            outgoing[edge.source_node_id].append(edge.target_node_id)
        queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        result: list[str] = []
        while queue:
            current = queue.popleft()
            result.append(current)
            for target in sorted(outgoing.get(current, ())):
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        return tuple(result)

    def role_entity_type(self, role_name: str) -> str:
        for role in self.graph.roles:
            if role.role_name == role_name:
                return role.entity_type
        raise KeyError(role_name)

    def role_distinct_from(self, role_name: str) -> tuple[str, ...]:
        for role in self.graph.roles:
            if role.role_name == role_name:
                return role.distinct_from
        return ()

    def predicate_variables(self, node_id: str) -> dict[str, str]:
        node = self.nodes_by_id[node_id]
        if node.predicate is None:
            return {}
        return {role.variable: role.entity_type for role in node.predicate.roles}

    def all_related_checkpoint_nodes(self, boundary_id: str, enabled_nodes: Iterable[str]) -> tuple[str, ...]:
        related = [node_id for node_id in enabled_nodes if self.nearest_checkpoint_boundary(node_id) == boundary_id]
        return tuple(sorted(related))
