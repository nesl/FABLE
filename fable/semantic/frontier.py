"""Hypothesis propagation, active-frontier derivation, and temporal checkpoints."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable

from fable.common.enums import (
    CheckpointKind,
    GraphNodeKind,
    HypothesisLifecycle,
    HypothesisNodeStatus,
    TemporalGuardKind,
    TruthValue,
)
from fable.common.schemas import (
    FrontierSnapshot,
    Hypothesis,
    HypothesisNodeState,
    SemanticCheckpoint,
)
from fable.common.time import EventTimeInterval, LatenessPolicy, SourceWatermark, utc_now

from .compiled import CompiledSemanticGraph
from .models import DerivedFrontier


_TERMINAL_NODE_STATES = {
    HypothesisNodeStatus.SATISFIED,
    HypothesisNodeStatus.FAILED,
    HypothesisNodeStatus.INVALIDATED,
    HypothesisNodeStatus.EXPIRED,
}


class FrontierDeriver:
    """Derive the semantic work that currently matters for one hypothesis."""

    def __init__(
        self,
        graph: CompiledSemanticGraph,
        *,
        lateness_policy: LatenessPolicy | None = None,
    ) -> None:
        self.graph = graph
        self.lateness_policy = lateness_policy or LatenessPolicy()

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

    def derive(
        self,
        hypothesis: Hypothesis,
        *,
        source_watermarks: dict[str, SourceWatermark] | None = None,
    ) -> DerivedFrontier | None:
        self.initialize_node_states(hypothesis)
        self.propagate_composites(hypothesis)
        if hypothesis.lifecycle != HypothesisLifecycle.ACTIVE:
            return None

        enabled = tuple(sorted(self._enabled_predicates(hypothesis)))
        if not enabled:
            return None

        for node_id, state in hypothesis.node_states.items():
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
        hypothesis.frontier_id = snapshot.frontier_id
        return DerivedFrontier(snapshot=snapshot, checkpoints=checkpoints)

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
        node = self.graph.nodes_by_id[node_id]
        precedence_tolerance = timedelta(
            milliseconds=int(node.annotations.get("precedence_tolerance_ms", 0))
        )
        minimum_delay_tolerance = timedelta(
            milliseconds=int(node.annotations.get("minimum_delay_tolerance_ms", 0))
        )
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
                    lower = (
                        latest_source_end
                        + timedelta(milliseconds=guard.minimum_ms)
                        - minimum_delay_tolerance
                    )
                    if interval.start < lower:
                        return False, f"result begins before minimum delay for guard {guard.guard_id}"
                if guard.maximum_ms is not None:
                    upper = earliest_source_end + timedelta(milliseconds=guard.maximum_ms)
                    if interval.start > upper:
                        return False, f"result begins after maximum delay for guard {guard.guard_id}"
                if (
                    guard.kind == TemporalGuardKind.PRECEDES
                    and interval.start < latest_source_end - precedence_tolerance
                ):
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

    def _checkpoint_for_group(
        self,
        hypothesis: Hypothesis,
        boundary_id: str,
        node_ids: tuple[str, ...],
    ) -> SemanticCheckpoint:
        boundary = self.graph.nodes_by_id[boundary_id]
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
