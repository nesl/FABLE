"""In-memory semantic state machine for one complex-event request.

Input is a static ``SemanticGraph`` plus normalized ``PredicateResult`` records.
Output is hypothesis/frontier transitions and eventual completion state. This
module deliberately stops at the semantic-to-physical boundary: it never picks
providers or nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
    ensure_utc,
    interval_closed_by_watermarks,
    utc_now,
)

from .bindings import BindingError, CanonicalBindingManager
from .compiled import CompiledSemanticGraph
from .frontier_deriver import FrontierDeriver
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


@dataclass(frozen=True)
class _ApplicationContext:
    """Validated runtime target and concurrency envelope for one result."""

    result: PredicateResult
    base: Hypothesis
    frontier: DerivedFrontier
    fork_context: _ForkContext | None
    context_key: tuple[UUID, int, UUID, str]
    occurrence_scope: tuple[str, str, str]


class SemanticRuntime:
    """Authoritative hypotheses and Active Frontiers for one request graph."""

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
        # Occurrence identities are only idempotent within one semantic
        # consumer.  A physical observation may legitimately advance several
        # concurrently active hypotheses; treating occurrence_id as
        # request-global silently starves every hypothesis after the first.
        self._processed_occurrences: set[tuple[str, str, str]] = set()
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
        """Return current state for one candidate complex-event occurrence."""
        with self._lock:
            return self._hypotheses[hypothesis_id]

    def get_frontier(self, hypothesis_id: UUID) -> DerivedFrontier | None:
        """Return predicates currently useful for the selected hypothesis."""
        with self._lock:
            hypothesis = self._hypotheses[hypothesis_id]
            if hypothesis.frontier_id is None:
                return None
            return self._frontiers.get(hypothesis.frontier_id)

    def invalidate_hypothesis(self, hypothesis_id: UUID) -> bool:
        """Invalidate an active fork and remove its runtime indexes.

        Rolling discovery pools use this operation when a newer camera-local
        candidate replaces an old candidate.  Keeping the invalidated child in
        a fork context or canonical index would cause later observations to be
        merged into a dead hypothesis.
        """

        with self._lock:
            hypothesis = self._hypotheses.get(hypothesis_id)
            if hypothesis is None or hypothesis.lifecycle != HypothesisLifecycle.ACTIVE:
                return False
            hypothesis.lifecycle = HypothesisLifecycle.INVALIDATED
            hypothesis.version += 1
            hypothesis.updated_at = utc_now()
            if hypothesis.frontier_id is not None:
                self._frontiers.pop(hypothesis.frontier_id, None)
                hypothesis.frontier_id = None
            if (
                hypothesis.canonical_key
                and self._canonical_index.get(hypothesis.canonical_key) == hypothesis_id
            ):
                self._canonical_index.pop(hypothesis.canonical_key, None)
            for context in self._fork_contexts.values():
                for key, child_id in tuple(context.child_ids_by_key.items()):
                    if child_id == hypothesis_id:
                        context.child_ids_by_key.pop(key, None)
            return True

    def invalidate_unprogressed_hypothesis(
        self,
        hypothesis_id: UUID,
        *,
        seed_event_time: EventTimeInterval,
        minimum_progress_gap_ms: int,
        force: bool = False,
    ) -> bool:
        """Invalidate a rolling seed only when no successor evidence arrived."""
        with self._lock:
            hypothesis = self._hypotheses.get(hypothesis_id)
            if hypothesis is None or hypothesis.lifecycle != HypothesisLifecycle.ACTIVE:
                return False
            progressed_states = [
                state for state in hypothesis.node_states.values()
                if state.occurrence_ids
            ]
            # A seeded hypothesis has exactly one evidenced primitive. Any
            # second evidenced graph node is meaningful progress and must not
            # be evicted from a bounded candidate pool.
            if len(progressed_states) > 1:
                return False
            age_ms = (
                hypothesis.event_time_window.end - seed_event_time.end
            ).total_seconds() * 1000
            if not force and age_ms < minimum_progress_gap_ms:
                return False
        return self.invalidate_hypothesis(hypothesis_id)

    def start(
        self,
        *,
        event_time_window: EventTimeInterval | None = None,
        observed_at: datetime | None = None,
        anchor_occurrence_id: str | None = None,
    ) -> RuntimeTransition:
        """Create an initial request-scoped hypothesis and expose root predicates.

        Historically Phase 1 required a separately discovered ``SeedPredicateResult``
        before a hypothesis existed.  The deployed controller needs to plan the first
        frontier as well, so this method creates an empty hypothesis whose enabled
        leaves are the authored graph roots.  The first accepted provider observation
        becomes the hypothesis anchor (see ``_anchor_first_observation``), preserving
        occurrence-level identity for independently occurring complex events.

        ``seed()`` remains available for compatibility with existing traces and tests.
        """

        with self._lock:
            active = self.active_hypotheses
            if active:
                # ``start`` is intentionally idempotent. Controllers may ask
                # for the initial frontier more than once (for example after
                # reconnect/recovery); active[0] is the already-created root
                # hypothesis returned to that caller, not a preferred event
                # candidate and not a pruning decision. Before any evidence
                # forks the request, there is exactly one such root.
                existing = active[0]
                frontier = self.get_frontier(existing.hypothesis_id)
                return RuntimeTransition(
                    status=ApplyStatus.NOOP,
                    hypothesis_ids=(existing.hypothesis_id,),
                    frontiers=() if frontier is None else (frontier,),
                    reason="request runtime already has an active initial hypothesis",
                )

            observed = ensure_utc(observed_at or utc_now())
            # The request window is event-time scope.  It is deliberately
            # separate from the processing deadline below: replayed evidence
            # may describe old media while still being processed now.
            window = event_time_window or EventTimeInterval(
                start=observed,
                end=observed + timedelta(milliseconds=self.config.hypothesis_horizon_ms),
            )
            provisional = Hypothesis(
                request_id=self.config.request_id,
                graph_id=self.graph.graph.graph_id,
                graph_hash=self.graph.graph.graph_hash,
                graph_version=self.graph.graph.graph_version,
                anchor_occurrence_id=(
                    anchor_occurrence_id
                    or f"request-start:{self.config.request_id}"
                ),
                event_time_window=window,
                deadline=DeadlineSpec(
                    latest_useful_completion=observed
                    + timedelta(milliseconds=self.config.deadline_offset_ms)
                ),
                created_at=observed,
                updated_at=observed,
            )
            # Allocate every authored node before deriving the frontier so
            # subsequent code can index node state without optional lookups.
            self.frontier_deriver.initialize_node_states(provisional)
            # Derivation evaluates topology and creates checkpoint metadata;
            # storing the hypothesis afterward makes the initial state appear
            # atomically to callers holding the runtime lock.
            frontier = self._derive_and_store(provisional)
            self._store_new(provisional)
            return RuntimeTransition(
                status=ApplyStatus.CREATED,
                hypothesis_ids=(provisional.hypothesis_id,),
                frontiers=() if frontier is None else (frontier,),
                reason="request initialized at authored root frontier",
            )

    def seed(self, result: SeedPredicateResult) -> RuntimeTransition:
        with self._lock:
            # Deduplicate before validation because redelivered broker messages
            # must be harmless even after the original changed runtime state.
            occurrence_scope = ("seed", result.graph_node_id, result.occurrence_id)
            duplicate = self._duplicate_transition(result.result_id, occurrence_scope)
            if duplicate:
                return duplicate
            reason = self._validate_seed_envelope(result)
            if reason:
                return self._reject(result.result_id, reason)
            if result.truth != TruthValue.TRUE:
                self._remember_result(result.result_id, occurrence_scope)
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
            # A seed may only create work that the empty authored graph would
            # actually enable; it cannot jump into the middle of a sequence.
            self.frontier_deriver.initialize_node_states(provisional)
            initial_frontier = self.frontier_deriver.derive(provisional)
            if initial_frontier is None or result.graph_node_id not in initial_frontier.snapshot.enabled_node_ids:
                return self._reject(result.result_id, "seed result does not match an enabled seed node")

            try:
                # Canonicalization converts provider-local identifiers into
                # request-level bindings and enforces role constraints before
                # any durable semantic state is mutated.
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
            # Identity validation may invalidate a candidate (for example two
            # roles authored as distinct resolving to the same entity).
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
            # Resolving an OR branch retires work from losing branches.  The
            # cancellation set returned below lets the physical layer stop it.
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
                self._remember_result(result.result_id, occurrence_scope)
                return RuntimeTransition(
                    status=ApplyStatus.MERGED,
                    hypothesis_ids=(existing_id,),
                    result_id=result.result_id,
                    reason="seed match merged with an existing canonical hypothesis",
                )

            frontier = self._derive_and_store(provisional)
            self._store_new(provisional)
            self._remember_result(result.result_id, occurrence_scope)
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
        """Validate and apply evidence, then derive the next semantic state.

        The algorithm validates the result envelope and optimistic version,
        resolves/forks entity bindings, updates predicate progress, propagates
        AND/OR/sequence/temporal progress, resolves branch cancellation, derives
        a new Active Frontier, and completes the hypothesis when its root is
        satisfied. Runtime state is mutated and a ``RuntimeTransition`` reports
        all affected hypothesis IDs/frontiers.
        """
        with self._lock:
            # Envelope, optimistic-version, active-frontier, checkpoint, and
            # temporal validation are kept together so no partially validated
            # result reaches binding or graph mutation.
            resolved = self._validate_and_resolve_result(result)
            if isinstance(resolved, RuntimeTransition):
                return resolved
            result = resolved.result
            try:
                introduced, validated = self._resolve_bindings(resolved)
            except BindingError as error:
                return self._reject(result.result_id, str(error))

            if result.truth == TruthValue.UNKNOWN:
                self._remember_result(result.result_id, resolved.occurrence_scope)
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
                if role_name not in resolved.base.role_bindings
            }
            # A true result that introduces an entity represents one candidate
            # interpretation, not a mutation of the shared parent.  Forking
            # preserves other provider candidates for the same open frontier.
            if result.truth == TruthValue.TRUE and new_roles:
                return self._apply_binding_fork(
                    result,
                    base=resolved.base,
                    frontier=resolved.frontier,
                    introduced=introduced,
                    validated=validated,
                    context_key=resolved.context_key,
                )
            if resolved.fork_context is not None:
                return RuntimeTransition(
                    status=ApplyStatus.STALE,
                    parent_hypothesis_id=result.hypothesis_id,
                    result_id=result.result_id,
                    reason="only additional binding candidates may be applied to a completed fork base",
                )

            return self._apply_in_place(
                result,
                base=resolved.base,
                old_frontier=resolved.frontier,
                introduced=introduced,
                validated=validated,
            )

    def _validate_and_resolve_result(
        self, result: PredicateResult
    ) -> _ApplicationContext | RuntimeTransition:
        """Validate identity/version/frontier/temporal context for ``apply``."""

        occurrence_scope = (
            str(result.hypothesis_id),
            result.graph_node_id,
            result.occurrence_id,
        )
        duplicate = self._duplicate_transition(result.result_id, occurrence_scope)
        if duplicate:
            return duplicate
        envelope_error = self._validate_result_envelope(result)
        if envelope_error:
            return self._reject(result.result_id, envelope_error)
        current = self._hypotheses.get(result.hypothesis_id)
        if current is None:
            return self._reject(result.result_id, "unknown hypothesis")

        # Sibling AND results dispatched from one parent must be projected over
        # active binding forks rather than becoming isolated one-sided forks.
        if current.lifecycle == HypothesisLifecycle.FORKED:
            projected = self._apply_to_active_fork_children(result)
            if projected is not None:
                return projected

        result = self._rebase_still_active_result(result, current)
        context_key = (
            result.hypothesis_id,
            result.expected_hypothesis_version,
            result.checkpoint_id,
            result.graph_node_id,
        )
        fork_context = self._fork_contexts.get(context_key)
        if (
            current.version == result.expected_hypothesis_version
            and current.lifecycle == HypothesisLifecycle.ACTIVE
        ):
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
            return self._reject(
                result.result_id, "result node is not part of the active checkpoint"
            )
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
            base, result.graph_node_id, result.event_time_interval
        )
        if not temporal_ok:
            return self._reject(result.result_id, temporal_reason)
        return _ApplicationContext(
            result=result,
            base=base,
            frontier=frontier,
            fork_context=fork_context,
            context_key=context_key,
            occurrence_scope=occurrence_scope,
        )

    def _rebase_still_active_result(
        self, result: PredicateResult, current: Hypothesis
    ) -> PredicateResult:
        """Rebase an in-flight sibling only if its predicate remains active."""

        if (
            current.lifecycle != HypothesisLifecycle.ACTIVE
            or current.version == result.expected_hypothesis_version
        ):
            return result
        active_frontier = self.get_frontier(current.hypothesis_id)
        if (
            active_frontier is None
            or result.graph_node_id not in active_frontier.snapshot.enabled_node_ids
            or self.graph.nodes_by_id[result.graph_node_id].predicate
            != result.semantic_predicate
        ):
            return result
        checkpoint = active_frontier.checkpoint_for_node(result.graph_node_id)
        return result.model_copy(
            update={
                "expected_hypothesis_version": current.version,
                "frontier_id": active_frontier.snapshot.frontier_id,
                "checkpoint_id": checkpoint.checkpoint_id,
            }
        )

    def _resolve_bindings(
        self, context: _ApplicationContext
    ) -> tuple[dict[str, EntityBinding], dict[str, EntityBinding]]:
        """Canonicalize result-local aliases against the target hypothesis."""

        result = context.result
        return self.bindings.canonicalize_delta(
            graph=self.graph,
            hypothesis=context.base,
            node_id=result.graph_node_id,
            introduced=result.binding_delta.introduced,
            validated=result.binding_delta.validated,
            source_id=self._primary_source(result.provenance.source_ids),
            occurrence_id=result.occurrence_id,
        )

    def _apply_to_active_fork_children(
        self, result: PredicateResult
    ) -> RuntimeTransition | None:
        child_ids = {
            child_id
            for key, context in self._fork_contexts.items()
            if key[0] == result.hypothesis_id
            for child_id in context.child_ids_by_key.values()
        }
        transitions: list[RuntimeTransition] = []
        for child_id in sorted(child_ids, key=str):
            child = self._hypotheses.get(child_id)
            frontier = self.get_frontier(child_id)
            if (
                child is None
                or child.lifecycle != HypothesisLifecycle.ACTIVE
                or frontier is None
                or result.graph_node_id not in frontier.snapshot.enabled_node_ids
            ):
                continue
            checkpoint = frontier.checkpoint_for_node(result.graph_node_id)
            projected = result.model_copy(
                update={
                    "result_id": uuid7(),
                    "hypothesis_id": child_id,
                    "expected_hypothesis_version": child.version,
                    "frontier_id": frontier.snapshot.frontier_id,
                    "checkpoint_id": checkpoint.checkpoint_id,
                }
            )
            transition = self.apply(projected)
            if transition.status not in {
                ApplyStatus.DUPLICATE,
                ApplyStatus.STALE,
                ApplyStatus.REJECTED,
            }:
                transitions.append(transition)
        if not transitions:
            return None

        # The projected result IDs are internal concurrency envelopes. Mark
        # the provider's original envelope as consumed as well so redelivery
        # cannot repeat the cross-product.
        self._processed_result_ids.add(result.result_id)
        hypothesis_ids = tuple(
            dict.fromkeys(
                hypothesis_id
                for transition in transitions
                for hypothesis_id in transition.hypothesis_ids
            )
        )
        frontiers = tuple(
            frontier
            for transition in transitions
            for frontier in transition.frontiers
        )
        cancelled_nodes = {
            node_id
            for transition in transitions
            for node_id in transition.cancellation.node_ids
        }
        cancelled_branches = {
            branch_id
            for transition in transitions
            for branch_id in transition.cancellation.branch_ids
        }
        return RuntimeTransition(
            status=(
                ApplyStatus.FORKED
                if any(t.status == ApplyStatus.FORKED for t in transitions)
                else ApplyStatus.APPLIED
            ),
            parent_hypothesis_id=result.hypothesis_id,
            hypothesis_ids=hypothesis_ids,
            frontiers=frontiers,
            cancellation=CancellationSet(
                node_ids=tuple(sorted(cancelled_nodes)),
                branch_ids=tuple(sorted(cancelled_branches)),
                reason="concurrent AND evidence projected across active binding forks",
            ),
            result_id=result.result_id,
        )

    def close_temporal_windows(self, watermarks: WatermarkSnapshot) -> tuple[RuntimeTransition, ...]:
        """Resolve ABSENT nodes only after coverage watermarks close their windows.

        A lack of observations is not evidence of absence until every required
        source has operational coverage beyond the interval (including allowed
        lateness). Successful closure mutates affected hypotheses/frontiers and
        returns their transitions.
        """

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

    def expire_temporal_windows(
        self,
        watermarks: WatermarkSnapshot,
        *,
        completed_source_ids: set[str] | frozenset[str] = frozenset(),
    ) -> tuple[RuntimeTransition, ...]:
        """Expire unresolved windows only after their source replay is complete.

        Absence resolution remains the responsibility of
        :meth:`close_temporal_windows`. This hook is intentionally conservative:
        an end-of-replay marker alone cannot manufacture positive or negative
        semantic evidence, so currently it returns no transitions.
        """
        del watermarks, completed_source_ids
        return ()

    def _apply_in_place(
        self,
        result: PredicateResult,
        *,
        base: Hypothesis,
        old_frontier: DerivedFrontier,
        introduced: dict[str, EntityBinding],
        validated: dict[str, EntityBinding],
    ) -> RuntimeTransition:
        """Apply evidence to a hypothesis that does not require a binding fork."""

        # Work on a deep copy so readers never observe a half-applied result and
        # so `_replace` can update canonical indexes as one logical operation.
        updated = base.model_copy(deep=True)
        updated.version += 1
        updated.updated_at = result.processing_completed_at
        updated.frontier_id = None
        updated.role_bindings.update(introduced)
        updated.role_bindings.update(validated)
        # Request-start is only a provisional anchor.  The first real evidence
        # supplies occurrence identity used for terminal-event deduplication.
        self._anchor_first_observation(updated, result.occurrence_id)
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
        # First propagate the new leaf truth, then resolve authored OR losers,
        # re-check identity constraints, and propagate once more because either
        # operation can change composite/root status.
        self.frontier_deriver.propagate_composites(updated)
        losing_nodes, losing_branches = self._resolve_satisfied_or_branches(updated)
        updated = self._revalidate_identity(updated)
        self.frontier_deriver.propagate_composites(updated)
        new_frontier = self._derive_and_store(updated)
        # Replace semantic state before remembering the delivery.  Both happen
        # under the runtime lock, so a retry sees either the complete update or
        # the prior state, never an intermediate graph.
        self._replace(base, updated)
        self._remember_result(
            result.result_id,
            (str(result.hypothesis_id), result.graph_node_id, result.occurrence_id),
        )

        old_enabled = set(old_frontier.snapshot.enabled_node_ids)
        new_enabled = set(new_frontier.snapshot.enabled_node_ids if new_frontier else ())
        cancelled = (old_enabled - new_enabled) | losing_nodes
        # Physical cancellation is derived from frontier difference rather
        # than guessed from predicate type, keeping semantics authoritative.
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
        """Create or merge a canonical child for newly introduced bindings."""

        context = self._fork_contexts.get(context_key)
        if context is None:
            # Freeze the pre-result parent once.  Every candidate result for
            # this checkpoint is projected from this same semantic baseline.
            context = _ForkContext(base=base.model_copy(deep=True), frontier=frontier)
            self._fork_contexts[context_key] = context
            parent = self._hypotheses[base.hypothesis_id].model_copy(deep=True)
            parent.lifecycle = HypothesisLifecycle.FORKED
            parent.version += 1
            parent.updated_at = result.processing_completed_at
            parent.frontier_id = None
            self._replace(base, parent)

        candidate = context.base.model_copy(deep=True)
        # Children require independent identity/version space; reusing the
        # parent's ID would make optimistic concurrency ambiguous.
        candidate.hypothesis_id = uuid7()
        candidate.version = context.base.version + 1
        candidate.lifecycle = HypothesisLifecycle.ACTIVE
        candidate.updated_at = result.processing_completed_at
        candidate.frontier_id = None
        candidate.role_bindings.update(introduced)
        candidate.role_bindings.update(validated)
        self._anchor_first_observation(candidate, result.occurrence_id)
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
            # Different provider-local observations can canonicalize to the
            # same entity tuple. Merge provenance instead of duplicating work.
            existing = self._hypotheses[existing_id]
            self._merge_evidence_only(
                existing,
                {**introduced, **validated},
                result.result_id,
                result.artifact_ids,
            )
            self._remember_result(
                result.result_id,
                (str(result.hypothesis_id), result.graph_node_id, result.occurrence_id),
            )
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
        self._remember_result(
            result.result_id,
            (str(result.hypothesis_id), result.graph_node_id, result.occurrence_id),
        )
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

    @staticmethod
    def _anchor_first_observation(hypothesis: Hypothesis, occurrence_id: str) -> None:
        if (
            hypothesis.anchor_occurrence_id.startswith("request-start:")
            and not hypothesis.provenance_result_ids
        ):
            object.__setattr__(hypothesis, "anchor_occurrence_id", occurrence_id)

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

    def _duplicate_transition(
        self,
        result_id: UUID,
        occurrence_scope: tuple[str, str, str],
    ) -> RuntimeTransition | None:
        if result_id in self._processed_result_ids:
            return RuntimeTransition(
                status=ApplyStatus.DUPLICATE,
                result_id=result_id,
                reason="result_id was already applied",
            )
        if (
            self.config.suppress_duplicate_occurrences
            and occurrence_scope in self._processed_occurrences
        ):
            return RuntimeTransition(
                status=ApplyStatus.DUPLICATE,
                result_id=result_id,
                reason="occurrence_id was already applied",
            )
        return None

    def _remember_result(
        self,
        result_id: UUID,
        occurrence_scope: tuple[str, str, str],
    ) -> None:
        self._processed_result_ids.add(result_id)
        if self.config.suppress_duplicate_occurrences:
            self._processed_occurrences.add(occurrence_scope)

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
