"""Active-frontier selection and checkpoint grouping for semantic hypotheses."""

from __future__ import annotations

from collections import defaultdict

from fable.common.enums import CheckpointKind, GraphNodeKind, HypothesisLifecycle, HypothesisNodeStatus
from fable.common.schemas import FrontierSnapshot, Hypothesis, SemanticCheckpoint
from fable.common.time import EventTimeInterval, LatenessPolicy, SourceWatermark, utc_now
from .compiled import CompiledSemanticGraph
from .evaluation import CompositeEvaluator
from .models import DerivedFrontier
from .temporal import TemporalEvaluator

_TERMINAL_NODE_STATES = {
    HypothesisNodeStatus.SATISFIED, HypothesisNodeStatus.FAILED,
    HypothesisNodeStatus.INVALIDATED, HypothesisNodeStatus.EXPIRED,
}

class FrontierDeriver:
    """Select primitive work enabled by already-evaluated semantic state.

    Composite truth propagation and temporal guard evaluation are delegated to
    focused engines; this class owns only frontier/checkpoint selection.
    """

    def __init__(self, graph: CompiledSemanticGraph, *, lateness_policy: LatenessPolicy | None = None) -> None:
        self.graph = graph
        self.lateness_policy = lateness_policy or LatenessPolicy()
        self.composites = CompositeEvaluator(graph)
        self.temporal = TemporalEvaluator(graph, lateness_policy=self.lateness_policy)

    # Compatibility surface used by SemanticRuntime.
    def initialize_node_states(self, hypothesis: Hypothesis) -> None:
        self.composites.initialize_node_states(hypothesis)

    def propagate_composites(self, hypothesis: Hypothesis) -> None:
        self.composites.propagate_composites(hypothesis)

    def checkpoint_interval(self, hypothesis: Hypothesis, boundary_id: str) -> EventTimeInterval:
        return self.temporal.checkpoint_interval(hypothesis, boundary_id)

    def result_obeys_temporal_guards(self, hypothesis: Hypothesis, node_id: str, interval: EventTimeInterval) -> tuple[bool, str]:
        return self.temporal.result_obeys_temporal_guards(hypothesis, node_id, interval)

    def absence_requirements(self, hypothesis: Hypothesis, absent_node_id: str) -> tuple[EventTimeInterval, tuple[str, ...]]:
        return self.temporal.absence_requirements(hypothesis, absent_node_id)
    def derive(
        self,
        hypothesis: Hypothesis,
        *,
        source_watermarks: dict[str, SourceWatermark] | None = None,
    ) -> DerivedFrontier | None:
        """Propagate graph state and expose currently executable predicates."""

        # Derivation is safe on a newly created hypothesis as well as after a
        # result; initialization is idempotent and propagation reaches a fixed
        # point before any leaf is advertised.
        self.initialize_node_states(hypothesis)
        self.propagate_composites(hypothesis)
        if hypothesis.lifecycle != HypothesisLifecycle.ACTIVE:
            # A satisfied/failed root has no physical work, even if some
            # descendant states were previously enabled.
            return None

        enabled = tuple(sorted(self._enabled_predicates(hypothesis)))
        if not enabled:
            return None

        for node_id, state in hypothesis.node_states.items():
            # Reconcile cached status with topology on every derivation. This
            # retires leaves from losing OR branches and re-enables planned work
            # after a physical replan without altering semantic truth.
            if state.status == HypothesisNodeStatus.ENABLED and node_id not in enabled:
                state.status = HypothesisNodeStatus.UNRESOLVED
            if node_id in enabled and state.status in (
                HypothesisNodeStatus.UNRESOLVED,
                HypothesisNodeStatus.PLANNED,
            ):
                state.status = HypothesisNodeStatus.ENABLED
                state.last_updated_at = utc_now()

        grouped: dict[str, list[str]] = defaultdict(list)
        for node_id in enabled:
            # Several leaves can belong to one authored checkpoint (for example
            # concurrent AND children); scheduling must plan them together.
            grouped[self.graph.nearest_checkpoint_boundary(node_id)].append(node_id)

        checkpoints = tuple(
            self._checkpoint_for_group(hypothesis, boundary_id, tuple(sorted(node_ids)))
            for boundary_id, node_ids in sorted(grouped.items())
        )
        snapshot = FrontierSnapshot(
            request_id=hypothesis.request_id,
            graph_hash=hypothesis.graph_hash,
            hypothesis_id=hypothesis.hypothesis_id,
            hypothesis_version=hypothesis.version,
            enabled_node_ids=enabled,
            checkpoint_ids=tuple(checkpoint.checkpoint_id for checkpoint in checkpoints),
            source_watermarks=dict(source_watermarks or {}),
        )
        # Store the frontier identity on the hypothesis so incoming results can
        # be rejected if they refer to a superseded planning boundary.
        hypothesis.frontier_id = snapshot.frontier_id
        return DerivedFrontier(snapshot=snapshot, checkpoints=checkpoints)

    def active_absence_nodes(self, hypothesis: Hypothesis) -> tuple[str, ...]:
        result: list[str] = []
        for node in self.graph.graph.nodes:
            if node.kind != GraphNodeKind.ABSENT:
                continue
            state = hypothesis.node_states.get(node.node_id)
            if state and state.status in _TERMINAL_NODE_STATES:
                continue
            if self._container_and_dependencies_active(hypothesis, node.node_id):
                result.append(node.node_id)
        return tuple(sorted(result))

    def _enabled_predicates(self, hypothesis: Hypothesis) -> set[str]:
        enabled: set[str] = set()
        for node in self.graph.graph.nodes:
            if node.kind != GraphNodeKind.PREDICATE:
                continue
            state = hypothesis.node_states[node.node_id]
            if state.status in _TERMINAL_NODE_STATES or state.status == HypothesisNodeStatus.RUNNING:
                continue
            if self._container_and_dependencies_active(hypothesis, node.node_id):
                enabled.add(node.node_id)
        return enabled

    def _container_and_dependencies_active(self, hypothesis: Hypothesis, node_id: str) -> bool:
        """Return whether sequence predecessors and every ancestor allow work."""

        # Sequence edges are strict gates independent of the surrounding
        # composite: all predecessors must be satisfied first.
        for predecessor_id in self.graph.sequence_predecessors(node_id):
            if hypothesis.node_states[predecessor_id].status != HypothesisNodeStatus.SATISFIED:
                return False

        parents = self.graph.parents(node_id)
        if not parents:
            return True

        active_parent_found = False
        for parent_id in parents:
            parent = self.graph.nodes_by_id[parent_id]
            parent_state = hypothesis.node_states[parent_id]
            if parent_state.status in _TERMINAL_NODE_STATES:
                continue
            if not self._container_and_dependencies_active(hypothesis, parent_id):
                continue
            # Reaching any still-active ordinary parent path is sufficient.
            # Composite truth itself is evaluated by CompositeEvaluator.
            if parent.kind == GraphNodeKind.OR:
                active_parent_found = True
            elif parent.kind in (
                GraphNodeKind.AND,
                GraphNodeKind.K_OF_N,
                GraphNodeKind.DURATION,
                GraphNodeKind.ABSENT,
                GraphNodeKind.WITHIN,
                GraphNodeKind.NAMED_SUBGRAPH,
            ):
                active_parent_found = True
        return active_parent_found

    def _checkpoint_for_group(
        self,
        hypothesis: Hypothesis,
        boundary_id: str,
        node_ids: tuple[str, ...],
    ) -> SemanticCheckpoint:
        """Describe one planning boundary and its success/failure consequences."""

        boundary = self.graph.nodes_by_id[boundary_id]
        # Checkpoint kind is diagnostic/policy metadata derived from authored
        # topology; it does not itself determine semantic completion.
        kind = {
            GraphNodeKind.OR: CheckpointKind.OR_RESOLUTION,
            GraphNodeKind.AND: CheckpointKind.AND_COMPLETION,
            GraphNodeKind.K_OF_N: CheckpointKind.CARDINALITY,
            GraphNodeKind.ABSENT: CheckpointKind.WINDOW_CLOSURE,
            GraphNodeKind.DURATION: CheckpointKind.WINDOW_CLOSURE,
            GraphNodeKind.WITHIN: CheckpointKind.WINDOW_CLOSURE,
        }.get(boundary.kind, CheckpointKind.PRIMITIVE)

        success: set[str] = set(self.graph.sequence_successors(boundary_id))
        if boundary_id not in node_ids:
            for node_id in node_ids:
                success.update(self.graph.sequence_successors(node_id))
        failure: set[str] = set()
        branches: list[str] = []
        if boundary.kind == GraphNodeKind.OR:
            # OR resolution can cancel every losing leaf and reports authored
            # branch labels for the physical control plane.
            for node_id in node_ids:
                label = self.graph.branch_label(boundary_id, node_id)
                if label:
                    branches.append(label)
            failure.update(node_ids)

        artifact_types = boundary.annotations.get("continuation_artifact_types", ())
        if isinstance(artifact_types, str):
            artifact_types = (artifact_types,)

        return SemanticCheckpoint(
            hypothesis_id=hypothesis.hypothesis_id,
            hypothesis_version=hypothesis.version,
            kind=kind,
            node_ids=node_ids,
            event_time_interval=self.checkpoint_interval(hypothesis, boundary_id),
            success_activates_node_ids=tuple(sorted(success)),
            failure_activates_node_ids=tuple(sorted(failure)),
            branch_ids=tuple(sorted(branches)),
            required_artifact_types_after_resolution=tuple(artifact_types),
        )
