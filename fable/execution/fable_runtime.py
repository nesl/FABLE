"""Top-level closed-loop FABLE runtime.

This is the small control loop that turns semantic progress and runtime-condition
changes into concrete provider lifecycle actions:

PredicateMatch -> identity -> CE instances -> frontier -> planner ->
ExecutionPlan -> reconciler -> START/KEEP/STOP -> node agents.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from datetime import datetime, timedelta, timezone
from typing import Mapping

from fable.language.event_parser import Event
from fable.planning.physical_planner import ExecutionPlan, PhysicalPlanner
from fable.planning.runtime_state import LinkState, RunningProvider, RuntimeState
from fable.providers.predicate_result import PredicateMatch
from fable.providers.identity import IdentityAssociation
from fable.runtime.ce_instance import CEInstance
from fable.runtime.frontier import ActiveFrontier, derive_predicate_frontier
from fable.runtime.instance_manager import CEInstanceManager

from .command_transport import CommandTransport
from .identity_resolver import IdentityResolver
from .node_agent import CommandResult, NodeStatus
from .plan_reconciler import ReconcileActions, reconcile_plan


@dataclass(frozen=True, slots=True)
class RuntimeUpdate:
    plan: ExecutionPlan
    actions: ReconcileActions
    produced_instances: tuple[CEInstance, ...] = ()
    completed_instances: tuple[CEInstance, ...] = ()


class ExecutionApplyError(RuntimeError):
    pass


class FableRuntime:
    def __init__(
        self,
        event: Event,
        runtime_state: RuntimeState,
        command_transport: CommandTransport,
        *,
        planner: PhysicalPlanner | None = None,
        instance_manager: CEInstanceManager | None = None,
        identity_resolver: IdentityResolver | None = None,
        frontier_mode: str = "fable",
    ) -> None:
        self.event = event
        self.runtime_state = runtime_state
        self.transport = command_transport
        self.planner = planner or PhysicalPlanner()
        self.manager = instance_manager or CEInstanceManager(event)
        self.identity_resolver = identity_resolver or IdentityResolver()
        if frontier_mode not in {"fable", "whole_event"}:
            raise ValueError("frontier_mode must be 'fable' or 'whole_event'")
        self.frontier_mode = frontier_mode
        self.last_plan = ExecutionPlan((), {}, 0.0, 0, 0, 0.0, 1.0)
        self.action_log: list[ReconcileActions] = []
        self._lock = threading.RLock()
        self._predicate_event_watermark: datetime | None = None
        self._buffered_predicates = frozenset({"audio_event"})
        self._predicate_buffer_window = timedelta(seconds=90)
        self._predicate_buffer: list[PredicateMatch] = []

    def current_frontier(self, now: datetime) -> ActiveFrontier:
        return self.manager.current_frontier(now)

    def visualization_state(self) -> dict[str, object]:
        """Return a read-only JSON-ready view of current CE instance progress."""
        from fable.visualization import runtime_visualization_state

        with self._lock:
            return runtime_visualization_state(self)

    def start(self, now: datetime) -> RuntimeUpdate:
        """Plan and activate persistent discovery work."""
        return self.replan(now)

    def handle_predicate_match(self, match: PredicateMatch) -> RuntimeUpdate:
        """Canonicalize one provider result, advance CE candidates, then replan."""
        with self._lock:
            before_completed = len(self.manager.completed_instances())
            canonical = self.identity_resolver.canonicalize_match(match)
            if canonical.predicate in self._buffered_predicates:
                self._buffer_predicate_match(canonical)
                if not self._frontier_accepts(canonical.predicate, canonical.event_time):
                    update = self.replan(canonical.event_time)
                    return RuntimeUpdate(update.plan, update.actions)
            logical_now = canonical.event_time
            if self._predicate_event_watermark is not None:
                logical_now = max(logical_now, self._predicate_event_watermark)
            self._predicate_event_watermark = logical_now
            if len(canonical.source_ids) == 1:
                source_id = canonical.source_ids[0]
                for arg_name in canonical.classes:
                    object_id = canonical.arguments.get(arg_name)
                    if isinstance(object_id, str) and object_id:
                        self.runtime_state.object_sources[object_id] = source_id
            produced = self.manager.handle_match(canonical, now=logical_now)
            produced_rows = list(produced)
            if canonical.predicate not in self._buffered_predicates:
                for buffered in tuple(self._predicate_buffer):
                    if not self._frontier_accepts(buffered.predicate, logical_now):
                        continue
                    replay_now = max(logical_now, buffered.event_time)
                    produced_rows.extend(self.manager.handle_match(buffered, now=replay_now))
                    logical_now = replay_now
            completed = self.manager.completed_instances()[before_completed:]
            update = self.replan(logical_now)
            return RuntimeUpdate(update.plan, update.actions, tuple(produced_rows), completed)

    def _buffer_predicate_match(self, match: PredicateMatch) -> None:
        self._predicate_buffer.append(match)
        newest = max(row.event_time for row in self._predicate_buffer)
        cutoff = newest - self._predicate_buffer_window
        self._predicate_buffer[:] = [
            row for row in self._predicate_buffer if row.event_time >= cutoff
        ]

    def _frontier_accepts(self, predicate: str, now: datetime) -> bool:
        frontier = self.manager.current_frontier(now)
        return any(
            item.predicate == predicate
            for items in frontier.continuation.values()
            for item in items
        ) or any(item.predicate == predicate for item in frontier.discovery)

    def handle_identity_association(
        self, association: IdentityAssociation, *, now: datetime | None = None
    ) -> RuntimeUpdate:
        """Apply one ReID association delivered by a provider worker."""
        when = now or datetime.now(timezone.utc)
        return self.merge_identity(
            association.left_object_id, association.right_object_id, now=when
        )

    def merge_identity(self, left: str, right: str, *, now: datetime) -> RuntimeUpdate:
        """Apply object ReID; CE-occurrence deduplication remains separate."""
        with self._lock:
            self.identity_resolver.merge(left, right)
            self.manager.recanonicalize_identities(self.identity_resolver.canonical)
            new_sources: dict[str, str] = {}
            for object_id, source_id in self.runtime_state.object_sources.items():
                new_sources[self.identity_resolver.canonical(object_id)] = source_id
            self.runtime_state.object_sources = new_sources
            return self.replan(now)

    def apply_identity_associations(
        self, associations: Mapping[str, str], *, now: datetime
    ) -> RuntimeUpdate:
        self.identity_resolver.apply_associations(associations)
        self.manager.recanonicalize_identities(self.identity_resolver.canonical)
        new_sources: dict[str, str] = {}
        for object_id, source_id in self.runtime_state.object_sources.items():
            new_sources[self.identity_resolver.canonical(object_id)] = source_id
        self.runtime_state.object_sources = new_sources
        return self.replan(now)

    def handle_link_state(self, link: LinkState, *, now: datetime) -> RuntimeUpdate:
        self.runtime_state.update_link(link)
        return self.replan(now)

    def handle_link_states(self, links: tuple[LinkState, ...], *, now: datetime) -> RuntimeUpdate:
        """Apply one atomic network-condition update and replan once."""
        with self._lock:
            for link in links:
                self.runtime_state.update_link(link)
            return self.replan(now)

    def handle_node_status(self, status: NodeStatus, *, now: datetime) -> RuntimeUpdate:
        self.runtime_state.update_node(status.as_node_state())
        self.runtime_state.replace_running_for_node(status.node_id, status.running)
        return self.replan(now)

    def refresh_node_status(self, node_id: str, *, now: datetime | None = None) -> NodeStatus:
        status = self.transport.status(node_id)
        self.runtime_state.update_node(status.as_node_state())
        self.runtime_state.replace_running_for_node(status.node_id, status.running)
        return status

    def replan(self, now: datetime) -> RuntimeUpdate:
        frontier = self.manager.current_frontier(now)
        if self.frontier_mode == "whole_event":
            # B0 evaluation policy: keep every authored predicate in this CE
            # active for the whole session.  Semantic matching still uses the
            # ordinary instance manager, so this changes resource use rather
            # than the meaning of a completed event.
            predicates: set[str] = set()

            def visit(expr) -> None:
                if expr.is_predicate:
                    predicates.add(expr.op)
                    return
                for child in expr.children:
                    visit(child)

            visit(self.event.pattern)
            all_items = tuple(
                item
                for predicate in sorted(predicates)
                for item in derive_predicate_frontier(self.event, predicate)
            )
            # A CE can author modality alternatives (for example audio
            # conversation OR visual proximity) while a replay exposes only
            # one modality. B0 means all feasible event work is always on; it
            # must not make a missing optional modality fatal.
            feasible_items = []
            for item in all_items:
                try:
                    self.planner.plan(
                        ActiveFrontier((item,), {}), self.runtime_state, now=now
                    )
                except RuntimeError:
                    continue
                feasible_items.append(item)
            frontier = ActiveFrontier(tuple(feasible_items), {})
        lookahead = tuple(
            item
            for predicate in self._buffered_predicates
            for item in derive_predicate_frontier(self.event, predicate)
        )
        if lookahead and self.frontier_mode == "fable":
            frontier = ActiveFrontier(frontier.discovery + lookahead, frontier.continuation)
        plan = self.planner.plan(frontier, self.runtime_state, now=now)
        actions = reconcile_plan(self.runtime_state.running, plan)
        self._apply(actions)
        self.last_plan = plan
        self.action_log.append(actions)
        return RuntimeUpdate(plan, actions)

    def shutdown(self) -> ReconcileActions:
        """Stop every FABLE-managed provider currently recorded as running."""
        empty = ExecutionPlan((), {}, 0.0, 0, 0, 0.0, 1.0)
        actions = reconcile_plan(self.runtime_state.running, empty)
        self._apply(actions)
        self.last_plan = empty
        self.action_log.append(actions)
        return actions

    def _apply(self, actions: ReconcileActions) -> None:
        """Apply replacement safely: START all replacements before STOP old work."""
        started: list[RunningProvider] = []
        # START is idempotent. Re-sending kept specs refreshes the small set of
        # active semantic demands without restarting a shared worker.
        for spec in actions.keep:
            result = self.transport.start(spec)
            if not result.ok:
                raise ExecutionApplyError(
                    f"failed to update {spec.key.provider_id}@{spec.key.node_id}: {result.message}"
                )
        for spec in actions.start:
            result = self.transport.start(spec)
            if not result.ok:
                # Do not execute STOP actions when replacement startup failed.
                # Replacements started earlier in this loop are already reflected
                # in RuntimeState and can be reused by the next replan.
                raise ExecutionApplyError(
                    f"failed to start {spec.key.provider_id}@{spec.key.node_id}: {result.message}"
                )
            row = RunningProvider(spec.key.provider_id, spec.key.node_id, spec.key.source_ids)
            started.append(row)
            self._add_running(row.provider_id, row.node_id, row.source_ids)

        for key in actions.stop:
            result = self.transport.stop(key)
            if not result.ok:
                raise ExecutionApplyError(
                    f"failed to stop {key.provider_id}@{key.node_id}: {result.message}"
                )
            self.runtime_state.running = tuple(
                row for row in self.runtime_state.running
                if not (
                    row.provider_id == key.provider_id
                    and row.node_id == key.node_id
                    and tuple(sorted(row.source_ids)) == key.source_ids
                )
            )

    def _add_running(self, provider_id: str, node_id: str, source_ids: tuple[str, ...]) -> None:
        expected = (provider_id, node_id, tuple(sorted(source_ids)))
        if any(
            (row.provider_id, row.node_id, tuple(sorted(row.source_ids))) == expected
            for row in self.runtime_state.running
        ):
            return
        self.runtime_state.running = tuple(
            sorted(
                (*self.runtime_state.running, RunningProvider(provider_id, node_id, source_ids)),
                key=lambda row: (row.node_id, row.provider_id, row.source_ids),
            )
        )
