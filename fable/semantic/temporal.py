"""Temporal-guard and semantic-window evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta

from fable.common.enums import GraphNodeKind, TemporalGuardKind
from fable.common.schemas import Hypothesis
from fable.common.time import EventTimeInterval, LatenessPolicy
from .compiled import CompiledSemanticGraph

class TemporalEvaluator:
    """Evaluate temporal guards and derive bounded checkpoint/absence windows."""

    def __init__(self, graph: CompiledSemanticGraph, *, lateness_policy: LatenessPolicy | None = None) -> None:
        self.graph = graph
        self.lateness_policy = lateness_policy or LatenessPolicy()
    def checkpoint_interval(self, hypothesis: Hypothesis, boundary_id: str) -> EventTimeInterval:
        node = self.graph.nodes_by_id[boundary_id]
        anchor = self._activation_anchor(hypothesis, boundary_id)
        end = hypothesis.event_time_window.end

        guards = list(self.graph.temporal_guards_for_target(boundary_id))
        for child_id in self.graph.children(boundary_id):
            guards.extend(self.graph.temporal_guards_on_parent_child(boundary_id, child_id))
            guards.extend(self.graph.temporal_guards_for_target(child_id))

        maximums = [guard.maximum_ms for guard in guards if guard.maximum_ms is not None]
        if maximums:
            end = min(end, anchor + timedelta(milliseconds=min(maximums)))

        if node.kind in (GraphNodeKind.ABSENT, GraphNodeKind.DURATION):
            minimums = [guard.minimum_ms for guard in guards if guard.minimum_ms is not None]
            if minimums:
                end = min(end, anchor + timedelta(milliseconds=max(minimums)))

        if end < anchor:
            end = anchor
        return EventTimeInterval(start=anchor, end=end)

    def result_obeys_temporal_guards(
        self,
        hypothesis: Hypothesis,
        node_id: str,
        interval: EventTimeInterval,
    ) -> tuple[bool, str]:
        guards = list(self.graph.temporal_guards_for_target(node_id))
        for parent_id in self.graph.parents(node_id):
            guards.extend(self.graph.temporal_guards_on_parent_child(parent_id, node_id))

        for guard in guards:
            source_intervals = [
                value
                for source_id in guard.source_node_ids
                for value in hypothesis.node_states[source_id].event_time_intervals
            ]
            if guard.kind == TemporalGuardKind.DURATION:
                if guard.minimum_ms is not None and interval.duration < timedelta(milliseconds=guard.minimum_ms):
                    # Short state intervals are retained for DURATION accumulation.
                    continue
            elif guard.kind in (
                TemporalGuardKind.WITHIN,
                TemporalGuardKind.MAX_GAP,
                TemporalGuardKind.PRECEDES,
            ):
                if not source_intervals:
                    return False, f"temporal guard {guard.guard_id} has no satisfied source interval"
                earliest_source_end = min(value.end for value in source_intervals)
                latest_source_end = max(value.end for value in source_intervals)
                if guard.minimum_ms is not None:
                    lower = latest_source_end + timedelta(milliseconds=guard.minimum_ms)
                    if interval.start < lower:
                        return False, f"result begins before minimum delay for guard {guard.guard_id}"
                if guard.maximum_ms is not None:
                    upper = earliest_source_end + timedelta(milliseconds=guard.maximum_ms)
                    if interval.start > upper:
                        return False, f"result begins after maximum delay for guard {guard.guard_id}"
                if guard.kind == TemporalGuardKind.PRECEDES and interval.start < latest_source_end:
                    return False, f"result violates precedence guard {guard.guard_id}"
            elif guard.kind == TemporalGuardKind.OVERLAPS:
                if not source_intervals or not any(interval.overlaps(value) for value in source_intervals):
                    return False, f"result does not overlap source intervals for guard {guard.guard_id}"
            elif guard.kind == TemporalGuardKind.REPEAT_WITHIN:
                if len(source_intervals) < (guard.count or 1):
                    return False, f"repeat count not reached for guard {guard.guard_id}"
                first = min(value.start for value in source_intervals)
                last = max(value.end for value in source_intervals)
                if guard.maximum_ms is not None and last - first > timedelta(milliseconds=guard.maximum_ms):
                    return False, f"repeat window exceeded for guard {guard.guard_id}"
        return True, ""

    def absence_requirements(
        self,
        hypothesis: Hypothesis,
        absent_node_id: str,
    ) -> tuple[EventTimeInterval, tuple[str, ...]]:
        node = self.graph.nodes_by_id[absent_node_id]
        if node.kind != GraphNodeKind.ABSENT:
            raise ValueError("absence_requirements requires an ABSENT node")
        interval = self.checkpoint_interval(hypothesis, absent_node_id)
        sources: set[str] = set()
        for child_id in self.graph.ordinary_children(absent_node_id):
            for guard in self.graph.temporal_guards_on_parent_child(absent_node_id, child_id):
                if guard.kind == TemporalGuardKind.ABSENCE_WINDOW:
                    sources.update(guard.required_source_ids)
        for guard in self.graph.temporal_guards_for_target(absent_node_id):
            if guard.kind == TemporalGuardKind.ABSENCE_WINDOW:
                sources.update(guard.required_source_ids)
        return interval, tuple(sorted(sources))

    def _activation_anchor(self, hypothesis: Hypothesis, node_id: str) -> datetime:
        predecessors = self.graph.sequence_predecessors(node_id)
        intervals = [
            interval
            for predecessor_id in predecessors
            for interval in hypothesis.node_states[predecessor_id].event_time_intervals
        ]
        if intervals:
            return max(interval.end for interval in intervals)
        for ancestor_id in self.graph.ancestors(node_id):
            predecessors = self.graph.sequence_predecessors(ancestor_id)
            intervals = [
                interval
                for predecessor_id in predecessors
                for interval in hypothesis.node_states[predecessor_id].event_time_intervals
            ]
            if intervals:
                return max(interval.end for interval in intervals)
        return hypothesis.event_time_window.start

