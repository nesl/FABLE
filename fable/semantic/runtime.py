"""In-memory semantic runtime for FABLE Phase 1.

This module deliberately stops at the semantic-to-physical boundary.  It uses
scripted predicate results, maintains one shared graph plus lightweight
hypotheses, derives frontiers/checkpoints, and applies results with optimistic
version checks and duplicate suppression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
import threading
from typing import Iterable
from uuid import UUID

from fable.common.enums import (
    GraphEdgeKind,
    GraphNodeKind,
    HypothesisLifecycle,
    HypothesisNodeStatus,
    TruthValue,
)
from fable.common.ids import deterministic_id, uuid7
from fable.common.schemas import (
    EntityBinding,
    Hypothesis,
    HypothesisNodeState,
    PredicateResult,
)
from fable.common.time import (
    DeadlineSpec,
    EventTimeInterval,
    WatermarkSnapshot,
    interval_closed_by_watermarks,
    utc_now,
)

from .bindings import BindingError, CanonicalBindingManager
from .compiled import CompiledSemanticGraph
from .frontier import FrontierDeriver
from .models import (
    ApplyStatus,
    CancellationSet,
    DerivedFrontier,
    RuntimeTransition,
    SeedPredicateResult,
    SemanticRuntimeConfig,
)


@dataclass
class _ForkContext:
    base: Hypothesis
    frontier: DerivedFrontier
    child_ids_by_key: dict[str, UUID] = field(default_factory=dict)


class SemanticRuntime:
    """Authoritative Phase-1 state machine for one request graph."""

    def __init__(
        self,
        graph,
        *,
        config: SemanticRuntimeConfig,
        bindings: CanonicalBindingManager | None = None,
    ) -> None:
        if config.request_id == "":
            raise ValueError("request_id must be non-empty")
        self.graph = CompiledSemanticGraph(graph)
        self.config = config
        self.bindings = bindings or CanonicalBindingManager()
        self.frontier_deriver = FrontierDeriver(
            self.graph,
            lateness_policy=config.lateness_policy,
        )
        self._hypotheses: dict[UUID, Hypothesis] = {}
        self._canonical_index: dict[str, UUID] = {}
        self._frontiers: dict[UUID, DerivedFrontier] = {}
        self._fork_contexts: dict[tuple[UUID, int, UUID, str], _ForkContext] = {}
        self._processed_result_ids: set[UUID] = set()
        self._processed_occurrences: set[str] = set()
        self._lock = threading.RLock()

    @property
    def hypotheses(self) -> tuple[Hypothesis, ...]:
        with self._lock:
            return tuple(sorted(self._hypotheses.values(), key=lambda item: str(item.hypothesis_id)))

    @property
    def active_hypotheses(self) -> tuple[Hypothesis, ...]:
        return tuple(
            hypothesis
            for hypothesis in self.hypotheses
            if hypothesis.lifecycle == HypothesisLifecycle.ACTIVE
        )

    def get_hypothesis(self, hypothesis_id: UUID) -> Hypothesis:
        with self._lock:
            return self._hypotheses[hypothesis_id]

    def get_frontier(self, hypothesis_id: UUID) -> DerivedFrontier | None:
        with self._lock:
            hypothesis = self._hypotheses[hypothesis_id]
            if hypothesis.frontier_id is None:
                return None
            return self._frontiers.get(hypothesis.frontier_id)

    def seed(self, result: SeedPredicateResult) -> RuntimeTransition:
        with self._lock:
            duplicate = self._duplicate_transition(result.result_id, result.occurrence_id)
            if duplicate:
                return duplicate
            reason = self._validate_seed_envelope(result)
            if reason:
                return self._reject(result.result_id, reason)
            if result.truth != TruthValue.TRUE:
                self._remember_result(result.result_id, result.occurrence_id)
                return RuntimeTransition(
                    status=ApplyStatus.NOOP,
                    result_id=result.result_id,
                    reason="a non-true seed result cannot create a hypothesis",
                )

            provisional = Hypothesis(
                request_id=self.config.request_id,
                graph_id=self.graph.graph.graph_id,
                graph_hash=self.graph.graph.graph_hash,
                graph_version=self.graph.graph.graph_version,
                anchor_occurrence_id=result.occurrence_id,
                event_time_window=EventTimeInterval(
                    start=result.event_time_interval.start,
                    end=result.event_time_interval.start
                    + timedelta(milliseconds=self.config.hypothesis_horizon_ms),
                ),
                deadline=DeadlineSpec(
                    latest_useful_completion=result.observed_at
                    + timedelta(milliseconds=self.config.deadline_offset_ms)
                ),
                created_at=result.observed_at,
                updated_at=result.observed_at,
            )
            self.frontier_deriver.initialize_node_states(provisional)
            initial_frontier = self.frontier_deriver.derive(provisional)
            if initial_frontier is None or result.graph_node_id not in initial_frontier.snapshot.enabled_node_ids:
                return self._reject(result.result_id, "seed result does not match an enabled seed node")

            try:
                introduced, validated = self.bindings.canonicalize_delta(
                    graph=self.graph,
                    hypothesis=None,
                    node_id=result.graph_node_id,
                    introduced=result.binding_delta.introduced,
                    validated=result.binding_delta.validated,
                    source_id=self._primary_source(result.provenance.source_ids),
                    occurrence_id=result.occurrence_id,
                )
            except BindingError as error:
                return self._reject(result.result_id, str(error))
            if validated:
                return self._reject(result.result_id, "seed results cannot validate pre-existing bindings")

            object.__setattr__(provisional, "role_bindings", introduced)
            provisional = self._revalidate_identity(provisional)
            self._mark_predicate(
                provisional,
                node_id=result.graph_node_id,
                truth=result.truth,
                occurrence_id=result.occurrence_id,
                interval=result.event_time_interval,
            )
            provisional.provenance_result_ids = (result.result_id,)
            provisional.evidence_artifact_ids = tuple(result.artifact_ids)
            self.frontier_deriver.propagate_composites(provisional)
            losing_nodes, losing_branches = self._resolve_satisfied_or_branches(provisional)
            provisional = self._revalidate_identity(provisional)
            self.frontier_deriver.propagate_composites(provisional)

            existing_id = self._canonical_index.get(provisional.canonical_key or "")
            if existing_id is not None:
                existing = self._hypotheses[existing_id]
                self._merge_evidence_only(
                    existing,
                    introduced,
                    result.result_id,
                    result.artifact_ids,
                )
                self._remember_result(result.result_id, result.occurrence_id)
                return RuntimeTransition(
                    status=ApplyStatus.MERGED,
                    hypothesis_ids=(existing_id,),
                    result_id=result.result_id,
                    reason="seed match merged with an existing canonical hypothesis",
                )

            frontier = self._derive_and_store(provisional)
            self._store_new(provisional)
            self._remember_result(result.result_id, result.occurrence_id)
            cancellation = CancellationSet(
                node_ids=tuple(sorted(losing_nodes)),
                branch_ids=tuple(sorted(losing_branches)),
                reason="authored OR branch resolved during seed application" if losing_nodes else "",
            )
            return RuntimeTransition(
                status=ApplyStatus.CREATED,
                hypothesis_ids=(provisional.hypothesis_id,),
                frontiers=() if frontier is None else (frontier,),
                cancellation=cancellation,
                result_id=result.result_id,
            )

    def apply(self, result: PredicateResult) -> RuntimeTransition:
        with self._lock:
            duplicate = self._duplicate_transition(result.result_id, result.occurrence_id)
            if duplicate:
                return duplicate
            envelope_error = self._validate_result_envelope(result)
            if envelope_error:
                return self._reject(result.result_id, envelope_error)

            current = self._hypotheses.get(result.hypothesis_id)
            if current is None:
                return self._reject(result.result_id, "unknown hypothesis")

            context_key = (
                result.hypothesis_id,
                result.expected_hypothesis_version,
                result.checkpoint_id,
                result.graph_node_id,
            )
            fork_context = self._fork_contexts.get(context_key)
            if current.version == result.expected_hypothesis_version and current.lifecycle == HypothesisLifecycle.ACTIVE:
                base = current
                frontier = self.get_frontier(current.hypothesis_id)
            elif fork_context is not None:
                base = fork_context.base
                frontier = fork_context.frontier
            else:
                return RuntimeTransition(
                    status=ApplyStatus.STALE,
                    parent_hypothesis_id=result.hypothesis_id,
                    result_id=result.result_id,
                    reason=(
                        f"expected hypothesis version {result.expected_hypothesis_version}; "
                        f"current version is {current.version}"
                    ),
                )

            if frontier is None:
                return self._reject(result.result_id, "hypothesis has no active frontier")
            if result.frontier_id != frontier.snapshot.frontier_id:
                return RuntimeTransition(
                    status=ApplyStatus.STALE,
                    parent_hypothesis_id=result.hypothesis_id,
                    result_id=result.result_id,
                    reason="result references an obsolete frontier",
                )
            try:
                checkpoint = frontier.checkpoint_for_node(result.graph_node_id)
            except KeyError:
                return self._reject(result.result_id, "result node is not part of the active checkpoint")
            if checkpoint.checkpoint_id != result.checkpoint_id:
                return RuntimeTransition(
                    status=ApplyStatus.STALE,
                    parent_hypothesis_id=result.hypothesis_id,
                    result_id=result.result_id,
                    reason="result references an obsolete checkpoint",
                )

            node = self.graph.nodes_by_id[result.graph_node_id]
            if node.predicate != result.semantic_predicate:
                return self._reject(result.result_id, "result predicate does not match graph node")
            temporal_ok, temporal_reason = self.frontier_deriver.result_obeys_temporal_guards(
                base,
                result.graph_node_id,
                result.event_time_interval,
            )
            if not temporal_ok:
                return self._reject(result.result_id, temporal_reason)

            try:
                introduced, validated = self.bindings.canonicalize_delta(
                    graph=self.graph,
                    hypothesis=base,
                    node_id=result.graph_node_id,
                    introduced=result.binding_delta.introduced,
                    validated=result.binding_delta.validated,
                    source_id=self._primary_source(result.provenance.source_ids),
                    occurrence_id=result.occurrence_id,
                )
            except BindingError as error:
                return self._reject(result.result_id, str(error))

            if result.truth == TruthValue.UNKNOWN:
                self._remember_result(result.result_id, result.occurrence_id)
                return RuntimeTransition(
                    status=ApplyStatus.NOOP,
                    parent_hypothesis_id=result.hypothesis_id,
                    hypothesis_ids=(result.hypothesis_id,),
                    result_id=result.result_id,
                    reason="unknown predicate result does not advance semantic state",
                )

            new_roles = {
                role_name: binding
                for role_name, binding in introduced.items()
                if role_name not in base.role_bindings
            }
            if result.truth == TruthValue.TRUE and new_roles:
                return self._apply_binding_fork(
                    result,
                    base=base,
                    frontier=frontier,
                    introduced=introduced,
                    validated=validated,
                    context_key=context_key,
                )
            if fork_context is not None:
                return RuntimeTransition(
                    status=ApplyStatus.STALE,
                    parent_hypothesis_id=result.hypothesis_id,
                    result_id=result.result_id,
                    reason="only additional binding candidates may be applied to a completed fork base",
                )

            return self._apply_in_place(
                result,
                base=base,
                old_frontier=frontier,
                introduced=introduced,
                validated=validated,
            )

    def close_temporal_windows(self, watermarks: WatermarkSnapshot) -> tuple[RuntimeTransition, ...]:
        """Close coverage-aware ABSENT checkpoints whose event-time windows elapsed."""

        transitions: list[RuntimeTransition] = []
        with self._lock:
            for original in tuple(self.active_hypotheses):
                for absent_node_id in self.frontier_deriver.active_absence_nodes(original):
                    interval, required_sources = self.frontier_deriver.absence_requirements(
                        original,
                        absent_node_id,
                    )
                    child_ids = self.graph.ordinary_children(absent_node_id)
                    positive = any(
                        candidate.overlaps(interval)
                        for child_id in child_ids
                        for candidate in original.node_states[child_id].event_time_intervals
                        if original.node_states[child_id].truth == TruthValue.TRUE
                    )
                    if positive:
                        continue
                    if not interval_closed_by_watermarks(
                        interval,
                        watermarks,
                        required_sources,
                        self.config.lateness_policy,
                        require_operational_coverage=True,
                    ):
                        continue

                    updated = original.model_copy(deep=True)
                    old_frontier = self.get_frontier(original.hypothesis_id)
                    updated.version += 1
                    updated.updated_at = watermarks.generated_at
                    occurrence_id = deterministic_id(
                        "absence",
                        {
                            "hypothesis_id": str(updated.hypothesis_id),
                            "node_id": absent_node_id,
                            "interval": interval,
                            "watermarks": watermarks,
                        },
                        length=32,
                    )
                    self._mark_predicate(
                        updated,
                        node_id=absent_node_id,
                        truth=TruthValue.TRUE,
                        occurrence_id=occurrence_id,
                        interval=interval,
                    )
                    cancelled = set()
                    for child_id in child_ids:
                        state = updated.node_states[child_id]
                        if state.status not in (
                            HypothesisNodeStatus.SATISFIED,
                            HypothesisNodeStatus.FAILED,
                            HypothesisNodeStatus.INVALIDATED,
                            HypothesisNodeStatus.EXPIRED,
                        ):
                            state.status = HypothesisNodeStatus.INVALIDATED
                            state.truth = TruthValue.FALSE
                            cancelled.add(child_id)
                    self.frontier_deriver.propagate_composites(updated)
                    losing_nodes, losing_branches = self._resolve_satisfied_or_branches(updated)
                    updated = self._revalidate_identity(updated)
                    self.frontier_deriver.propagate_composites(updated)
                    new_frontier = self._derive_and_store(
                        updated,
                        source_watermarks=watermarks.sources,
                    )
                    self._replace(original, updated)
                    old_enabled = set(old_frontier.snapshot.enabled_node_ids if old_frontier else ())
                    new_enabled = set(new_frontier.snapshot.enabled_node_ids if new_frontier else ())
                    cancelled.update(old_enabled - new_enabled)
                    cancelled.update(losing_nodes)
                    transition = RuntimeTransition(
                        status=ApplyStatus.WINDOW_CLOSED,
                        parent_hypothesis_id=original.hypothesis_id,
                        hypothesis_ids=(updated.hypothesis_id,),
                        frontiers=() if new_frontier is None else (new_frontier,),
                        cancellation=CancellationSet(
                            node_ids=tuple(sorted(cancelled)),
                            branch_ids=tuple(sorted(losing_branches)),
                            reason="coverage-aware absence window closed",
                        ),
                    )
                    transitions.append(transition)
                    original = updated
        return tuple(transitions)

    def _apply_in_place(
        self,
        result: PredicateResult,
        *,
        base: Hypothesis,
        old_frontier: DerivedFrontier,
        introduced: dict[str, EntityBinding],
        validated: dict[str, EntityBinding],
    ) -> RuntimeTransition:
        updated = base.model_copy(deep=True)
        updated.version += 1
        updated.updated_at = result.processing_completed_at
        updated.frontier_id = None
        updated.role_bindings.update(introduced)
        updated.role_bindings.update(validated)
        updated = self._revalidate_identity(updated)
        updated.provenance_result_ids = tuple(
            dict.fromkeys((*updated.provenance_result_ids, result.result_id))
        )
        updated.evidence_artifact_ids = tuple(
            dict.fromkeys((*updated.evidence_artifact_ids, *result.artifact_ids))
        )
        self._mark_predicate(
            updated,
            node_id=result.graph_node_id,
            truth=result.truth,
            occurrence_id=result.occurrence_id,
            interval=result.event_time_interval,
        )
        self.frontier_deriver.propagate_composites(updated)
        losing_nodes, losing_branches = self._resolve_satisfied_or_branches(updated)
        updated = self._revalidate_identity(updated)
        self.frontier_deriver.propagate_composites(updated)
        new_frontier = self._derive_and_store(updated)
        self._replace(base, updated)
        self._remember_result(result.result_id, result.occurrence_id)

        old_enabled = set(old_frontier.snapshot.enabled_node_ids)
        new_enabled = set(new_frontier.snapshot.enabled_node_ids if new_frontier else ())
        cancelled = (old_enabled - new_enabled) | losing_nodes
        return RuntimeTransition(
            status=ApplyStatus.APPLIED,
            parent_hypothesis_id=base.hypothesis_id,
            hypothesis_ids=(updated.hypothesis_id,),
            frontiers=() if new_frontier is None else (new_frontier,),
            cancellation=CancellationSet(
                node_ids=tuple(sorted(cancelled)),
                branch_ids=tuple(sorted(losing_branches)),
                reason="checkpoint resolution retired obsolete semantic work" if cancelled else "",
            ),
            result_id=result.result_id,
        )

    def _apply_binding_fork(
        self,
        result: PredicateResult,
        *,
        base: Hypothesis,
        frontier: DerivedFrontier,
        introduced: dict[str, EntityBinding],
        validated: dict[str, EntityBinding],
        context_key: tuple[UUID, int, UUID, str],
    ) -> RuntimeTransition:
        context = self._fork_contexts.get(context_key)
        if context is None:
            context = _ForkContext(base=base.model_copy(deep=True), frontier=frontier)
            self._fork_contexts[context_key] = context
            parent = self._hypotheses[base.hypothesis_id].model_copy(deep=True)
            parent.lifecycle = HypothesisLifecycle.FORKED
            parent.version += 1
            parent.updated_at = result.processing_completed_at
            parent.frontier_id = None
            self._replace(base, parent)

        candidate = context.base.model_copy(deep=True)
        candidate.hypothesis_id = uuid7()
        candidate.version = context.base.version + 1
        candidate.lifecycle = HypothesisLifecycle.ACTIVE
        candidate.updated_at = result.processing_completed_at
        candidate.frontier_id = None
        candidate.role_bindings.update(introduced)
        candidate.role_bindings.update(validated)
        candidate = self._revalidate_identity(candidate)
        candidate.provenance_result_ids = tuple(
            dict.fromkeys((*candidate.provenance_result_ids, result.result_id))
        )
        candidate.evidence_artifact_ids = tuple(
            dict.fromkeys((*candidate.evidence_artifact_ids, *result.artifact_ids))
        )
        self._mark_predicate(
            candidate,
            node_id=result.graph_node_id,
            truth=result.truth,
            occurrence_id=result.occurrence_id,
            interval=result.event_time_interval,
        )
        self.frontier_deriver.propagate_composites(candidate)
        losing_nodes, losing_branches = self._resolve_satisfied_or_branches(candidate)
        candidate = self._revalidate_identity(candidate)
        self.frontier_deriver.propagate_composites(candidate)

        existing_id = context.child_ids_by_key.get(candidate.canonical_key or "")
        if existing_id is None:
            indexed = self._canonical_index.get(candidate.canonical_key or "")
            if indexed is not None and indexed != base.hypothesis_id:
                existing_id = indexed
        if existing_id is not None:
            existing = self._hypotheses[existing_id]
            self._merge_evidence_only(
                existing,
                {**introduced, **validated},
                result.result_id,
                result.artifact_ids,
            )
            self._remember_result(result.result_id, result.occurrence_id)
            return RuntimeTransition(
                status=ApplyStatus.MERGED,
                parent_hypothesis_id=base.hypothesis_id,
                hypothesis_ids=(existing_id,),
                result_id=result.result_id,
                reason="provider-local identity resolved to an existing canonical fork",
            )

        new_frontier = self._derive_and_store(candidate)
        self._store_new(candidate)
        context.child_ids_by_key[candidate.canonical_key or ""] = candidate.hypothesis_id
        self._remember_result(result.result_id, result.occurrence_id)
        old_enabled = set(frontier.snapshot.enabled_node_ids)
        new_enabled = set(new_frontier.snapshot.enabled_node_ids if new_frontier else ())
        cancelled = (old_enabled - new_enabled) | losing_nodes
        return RuntimeTransition(
            status=ApplyStatus.FORKED,
            parent_hypothesis_id=base.hypothesis_id,
            hypothesis_ids=(candidate.hypothesis_id,),
            frontiers=() if new_frontier is None else (new_frontier,),
            cancellation=CancellationSet(
                node_ids=tuple(sorted(cancelled)),
                branch_ids=tuple(sorted(losing_branches)),
                reason="new canonical binding created a hypothesis fork",
            ),
            result_id=result.result_id,
        )

    def _mark_predicate(
        self,
        hypothesis: Hypothesis,
        *,
        node_id: str,
        truth: TruthValue,
        occurrence_id: str,
        interval: EventTimeInterval,
    ) -> None:
        state = hypothesis.node_states.setdefault(
            node_id,
            HypothesisNodeState(node_id=node_id),
        )
        state.occurrence_ids = tuple(dict.fromkeys((*state.occurrence_ids, occurrence_id)))
        intervals = list(state.event_time_intervals)
        if not any(value.start == interval.start and value.end == interval.end for value in intervals):
            intervals.append(interval)
        state.event_time_intervals = tuple(
            sorted(intervals, key=lambda value: (value.start, value.end))
        )
        state.last_updated_at = utc_now()
        state.truth = truth

        parents = self.graph.parents(node_id)
        parent_kinds = {self.graph.nodes_by_id[parent].kind for parent in parents}
        if truth == TruthValue.TRUE:
            state.status = HypothesisNodeStatus.SATISFIED
            if GraphNodeKind.DURATION in parent_kinds:
                # The duration wrapper decides whether accumulated intervals are enough.
                duration_parent = next(
                    parent for parent in parents if self.graph.nodes_by_id[parent].kind == GraphNodeKind.DURATION
                )
                self.frontier_deriver.propagate_composites(hypothesis)
                if hypothesis.node_states[duration_parent].status != HypothesisNodeStatus.SATISFIED:
                    state.status = HypothesisNodeStatus.ENABLED
                    state.truth = TruthValue.UNKNOWN
        elif truth == TruthValue.FALSE:
            if GraphNodeKind.ABSENT in parent_kinds:
                # A negative sample does not close an absence window; monitoring continues.
                state.status = HypothesisNodeStatus.ENABLED
            else:
                state.status = HypothesisNodeStatus.FAILED
        else:
            state.status = HypothesisNodeStatus.ENABLED

    def _resolve_satisfied_or_branches(self, hypothesis: Hypothesis) -> tuple[set[str], set[str]]:
        cancelled_nodes: set[str] = set()
        cancelled_branches: set[str] = set()
        selected_branch_ids = set(hypothesis.structural_branch_ids)

        for or_node in self.graph.graph.nodes:
            if or_node.kind != GraphNodeKind.OR:
                continue
            selected_children = [
                child_id
                for child_id in self.graph.alternative_children(or_node.node_id)
                if hypothesis.node_states[child_id].status == HypothesisNodeStatus.SATISFIED
            ]
            if not selected_children:
                continue
            selected_child = sorted(selected_children)[0]
            selected_label = self.graph.branch_label(or_node.node_id, selected_child)
            if selected_label:
                selected_branch_ids.add(selected_label)
            for child_id in self.graph.alternative_children(or_node.node_id):
                if child_id == selected_child:
                    continue
                label = self.graph.branch_label(or_node.node_id, child_id)
                if label:
                    cancelled_branches.add(label)
                for node_id in self.graph.branch_subtree(or_node.node_id, child_id):
                    state = hypothesis.node_states[node_id]
                    if state.status != HypothesisNodeStatus.SATISFIED:
                        state.status = HypothesisNodeStatus.INVALIDATED
                        state.truth = TruthValue.FALSE
                        cancelled_nodes.add(node_id)
        object.__setattr__(
            hypothesis,
            "structural_branch_ids",
            tuple(sorted(selected_branch_ids)),
        )
        return cancelled_nodes, cancelled_branches

    def _derive_and_store(
        self,
        hypothesis: Hypothesis,
        *,
        source_watermarks=None,
    ) -> DerivedFrontier | None:
        frontier = self.frontier_deriver.derive(
            hypothesis,
            source_watermarks=source_watermarks,
        )
        if frontier is not None:
            self._frontiers[frontier.snapshot.frontier_id] = frontier
        return frontier

    def _store_new(self, hypothesis: Hypothesis) -> None:
        self._hypotheses[hypothesis.hypothesis_id] = hypothesis
        if hypothesis.canonical_key:
            self._canonical_index[hypothesis.canonical_key] = hypothesis.hypothesis_id

    def _replace(self, old: Hypothesis, new: Hypothesis) -> None:
        if old.canonical_key and self._canonical_index.get(old.canonical_key) == old.hypothesis_id:
            self._canonical_index.pop(old.canonical_key, None)
        self._hypotheses[new.hypothesis_id] = new
        if new.canonical_key:
            self._canonical_index[new.canonical_key] = new.hypothesis_id

    def _merge_evidence_only(
        self,
        hypothesis: Hypothesis,
        bindings: dict[str, EntityBinding],
        result_id: UUID,
        artifact_ids: Iterable[UUID],
    ) -> None:
        for role_name, candidate in bindings.items():
            existing = hypothesis.role_bindings.get(role_name)
            if existing is None:
                continue
            hypothesis.role_bindings[role_name] = existing.model_copy(
                update={
                    "local_entity_ids": self.bindings.aliases_for(
                        existing.entity_type,
                        existing.canonical_entity_id,
                    )
                }
            )
        hypothesis.provenance_result_ids = tuple(
            dict.fromkeys((*hypothesis.provenance_result_ids, result_id))
        )
        hypothesis.evidence_artifact_ids = tuple(
            dict.fromkeys((*hypothesis.evidence_artifact_ids, *artifact_ids))
        )
        hypothesis.updated_at = utc_now()

    def _revalidate_identity(self, hypothesis: Hypothesis) -> Hypothesis:
        payload = hypothesis.model_dump(mode="python")
        payload["canonical_key"] = None
        return Hypothesis.model_validate(payload)

    def _validate_seed_envelope(self, result: SeedPredicateResult) -> str:
        if result.request_id != self.config.request_id:
            return "seed result request_id does not match runtime"
        if result.graph_hash != self.graph.graph.graph_hash:
            return "seed result graph hash does not match runtime"
        node = self.graph.nodes_by_id.get(result.graph_node_id)
        if node is None or node.kind != GraphNodeKind.PREDICATE:
            return "seed result references an unknown or non-predicate node"
        if node.predicate != result.semantic_predicate:
            return "seed result predicate does not match graph node"
        if result.binding_delta.rejected_roles:
            return "seed result contains rejected roles"
        return ""

    def _validate_result_envelope(self, result: PredicateResult) -> str:
        if result.request_id != self.config.request_id:
            return "result request_id does not match runtime"
        if result.graph_hash != self.graph.graph.graph_hash:
            return "result graph hash does not match runtime"
        if result.graph_node_id not in self.graph.nodes_by_id:
            return "result references an unknown graph node"
        if result.binding_delta.rejected_roles:
            return "result contains rejected roles"
        return ""

    def _duplicate_transition(self, result_id: UUID, occurrence_id: str) -> RuntimeTransition | None:
        if result_id in self._processed_result_ids:
            return RuntimeTransition(
                status=ApplyStatus.DUPLICATE,
                result_id=result_id,
                reason="result_id was already applied",
            )
        if self.config.suppress_duplicate_occurrences and occurrence_id in self._processed_occurrences:
            return RuntimeTransition(
                status=ApplyStatus.DUPLICATE,
                result_id=result_id,
                reason="occurrence_id was already applied",
            )
        return None

    def _remember_result(self, result_id: UUID, occurrence_id: str) -> None:
        self._processed_result_ids.add(result_id)
        if self.config.suppress_duplicate_occurrences:
            self._processed_occurrences.add(occurrence_id)

    @staticmethod
    def _primary_source(source_ids: tuple[str, ...]) -> str:
        if not source_ids:
            raise BindingError("predicate result provenance must include at least one source")
        return source_ids[0]

    @staticmethod
    def _reject(result_id: UUID, reason: str) -> RuntimeTransition:
        return RuntimeTransition(
            status=ApplyStatus.REJECTED,
            result_id=result_id,
            reason=reason,
        )
