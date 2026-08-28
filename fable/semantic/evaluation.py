"""Composite semantic-state evaluation, separate from frontier selection."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from fable.common.enums import GraphNodeKind, HypothesisLifecycle, HypothesisNodeStatus, TemporalGuardKind, TruthValue
from fable.common.schemas import Hypothesis, HypothesisNodeState
from fable.common.time import EventTimeInterval, utc_now
from .compiled import CompiledSemanticGraph

_TERMINAL_NODE_STATES = {
    HypothesisNodeStatus.SATISFIED, HypothesisNodeStatus.FAILED,
    HypothesisNodeStatus.INVALIDATED, HypothesisNodeStatus.EXPIRED,
}

class CompositeEvaluator:
    """Evaluate AND/OR/K-of-N and semantic wrapper states to a fixed point."""

    def __init__(self, graph: CompiledSemanticGraph) -> None:
        self.graph = graph
    def initialize_node_states(self, hypothesis: Hypothesis) -> None:
        for node in self.graph.graph.nodes:
            hypothesis.node_states.setdefault(
                node.node_id,
                HypothesisNodeState(node_id=node.node_id),
            )

    def propagate_composites(self, hypothesis: Hypothesis) -> None:
        """Update AND/OR/K-of-N/wrapper states to a fixed point."""

        self.initialize_node_states(hypothesis)
        changed = True
        while changed:
            changed = False
            for node in self.graph.graph.nodes:
                if node.kind == GraphNodeKind.PREDICATE:
                    continue
                state = hypothesis.node_states[node.node_id]
                old_status = state.status
                new_status = self._composite_status(hypothesis, node.node_id)
                if new_status != old_status:
                    state.status = new_status
                    state.truth = self._truth_from_status(new_status)
                    state.last_updated_at = utc_now()
                    self._inherit_intervals_from_children(hypothesis, node.node_id)
                    changed = True

        root_state = hypothesis.node_states[self.graph.graph.root_node_id]
        if root_state.status == HypothesisNodeStatus.SATISFIED:
            hypothesis.lifecycle = HypothesisLifecycle.COMPLETED
            hypothesis.frontier_id = None
        elif root_state.status in (
            HypothesisNodeStatus.FAILED,
            HypothesisNodeStatus.INVALIDATED,
            HypothesisNodeStatus.EXPIRED,
        ):
            hypothesis.lifecycle = HypothesisLifecycle.INVALIDATED
            hypothesis.frontier_id = None

    def _composite_status(self, hypothesis: Hypothesis, node_id: str) -> HypothesisNodeStatus:
        node = self.graph.nodes_by_id[node_id]
        children = self.graph.children(node_id)
        statuses = [hypothesis.node_states[child].status for child in children]
        current = hypothesis.node_states[node_id].status

        if node.kind == GraphNodeKind.AND:
            if children and all(status == HypothesisNodeStatus.SATISFIED for status in statuses):
                return HypothesisNodeStatus.SATISFIED
            if any(status in (HypothesisNodeStatus.FAILED, HypothesisNodeStatus.EXPIRED) for status in statuses):
                return HypothesisNodeStatus.FAILED
        elif node.kind == GraphNodeKind.OR:
            if any(status == HypothesisNodeStatus.SATISFIED for status in statuses):
                return HypothesisNodeStatus.SATISFIED
            if children and all(status in (
                HypothesisNodeStatus.FAILED,
                HypothesisNodeStatus.INVALIDATED,
                HypothesisNodeStatus.EXPIRED,
            ) for status in statuses):
                return HypothesisNodeStatus.FAILED
        elif node.kind == GraphNodeKind.K_OF_N:
            satisfied = sum(status == HypothesisNodeStatus.SATISFIED for status in statuses)
            possible = sum(status not in (
                HypothesisNodeStatus.FAILED,
                HypothesisNodeStatus.INVALIDATED,
                HypothesisNodeStatus.EXPIRED,
            ) for status in statuses)
            if satisfied >= (node.k or 1):
                return HypothesisNodeStatus.SATISFIED
            if possible < (node.k or 1):
                return HypothesisNodeStatus.FAILED
        elif node.kind == GraphNodeKind.DURATION:
            child_id = children[0]
            intervals = hypothesis.node_states[child_id].event_time_intervals
            minimums = [
                guard.minimum_ms or 0
                for guard in self.graph.temporal_guards_on_parent_child(node_id, child_id)
                if guard.kind == TemporalGuardKind.DURATION
            ]
            required = timedelta(milliseconds=max(minimums or [0]))
            if any(interval.duration >= required for interval in self._merge_intervals(intervals)):
                return HypothesisNodeStatus.SATISFIED
            if hypothesis.node_states[child_id].status in (
                HypothesisNodeStatus.FAILED,
                HypothesisNodeStatus.EXPIRED,
            ):
                return HypothesisNodeStatus.FAILED
        elif node.kind == GraphNodeKind.WITHIN:
            child_status = hypothesis.node_states[children[0]].status
            if child_status == HypothesisNodeStatus.SATISFIED:
                return HypothesisNodeStatus.SATISFIED
            if child_status in (HypothesisNodeStatus.FAILED, HypothesisNodeStatus.EXPIRED):
                return HypothesisNodeStatus.FAILED
        elif node.kind == GraphNodeKind.ABSENT:
            child_status = hypothesis.node_states[children[0]].status
            if current == HypothesisNodeStatus.SATISFIED:
                return current
            if child_status == HypothesisNodeStatus.SATISFIED:
                return HypothesisNodeStatus.FAILED
        return current if current in _TERMINAL_NODE_STATES else HypothesisNodeStatus.UNRESOLVED

    def _inherit_intervals_from_children(self, hypothesis: Hypothesis, node_id: str) -> None:
        intervals = tuple(
            interval
            for child_id in self.graph.children(node_id)
            for interval in hypothesis.node_states[child_id].event_time_intervals
        )
        if intervals:
            unique: dict[tuple[datetime, datetime], EventTimeInterval] = {}
            for interval in intervals:
                unique[(interval.start, interval.end)] = interval
            hypothesis.node_states[node_id].event_time_intervals = tuple(
                sorted(unique.values(), key=lambda item: (item.start, item.end))
            )

    @staticmethod
    def _truth_from_status(status: HypothesisNodeStatus) -> TruthValue:
        if status == HypothesisNodeStatus.SATISFIED:
            return TruthValue.TRUE
        if status in (
            HypothesisNodeStatus.FAILED,
            HypothesisNodeStatus.INVALIDATED,
            HypothesisNodeStatus.EXPIRED,
        ):
            return TruthValue.FALSE
        return TruthValue.UNKNOWN

    @staticmethod
    def _merge_intervals(intervals: Iterable[EventTimeInterval]) -> tuple[EventTimeInterval, ...]:
        ordered = sorted(intervals, key=lambda value: (value.start, value.end))
        if not ordered:
            return ()
        merged = [ordered[0]]
        for interval in ordered[1:]:
            last = merged[-1]
            if interval.start <= last.end:
                merged[-1] = EventTimeInterval(start=last.start, end=max(last.end, interval.end))
            else:
                merged.append(interval)
        return tuple(merged)

