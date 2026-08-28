"""Closed-loop deployed controller for FABLE.

This module composes the semantic, planning, scheduling, and distributed layers.
The public request boundary is an authored complex-event request; physical plan
candidates remain available only as a lower-level debug/transport interface.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
from uuid import UUID

from fable.common.enums import (
    CheckpointKind,
    ExecutionInputKind,
    HypothesisLifecycle,
    NodeAvailability,
)
from fable.common.ids import deterministic_id, uuid7
from fable.common.schemas import (
    Hypothesis,
    NodeHeartbeat,
    PredicateDemand,
    PredicateResult,
    RuntimeLinkUpdate,
    TerminalComplexEvent,
)
from fable.planning import (
    AlternativeBuildConfig,
    ArtifactCatalog,
    BoundedLabelPlanner,
    DemandCompileContext,
    DemandCompiler,
    PhysicalAlternativeGraphBuilder,
    RuntimeDeploymentView,
    default_predicate_registry,
)
from fable.planning.models import PhysicalAlternativeGraph, PrunedAlternative
from fable.planning.provider_registry import ProviderRegistry
from fable.scheduling.adapters import candidate_from_search_result
from fable.scheduling.control import CheckpointController
from fable.scheduling.models import TaskSchedulingPolicy
from fable.semantic import (
    EventRequestCompiler,
    RuntimeTransition,
    SemanticRuntime,
    SemanticRuntimeConfig,
    StructuredEventRequest,
)

from fable.distributed.codec import decode_model, encode_model
from fable.distributed.models import (
    ArtifactAnnouncement,
    EventRequestResponse,
    EventRequestSubmission,
    ExecutionProfile,
    RuntimeMode,
    RuntimeDisturbanceAck,
    RuntimeDisturbanceRequest,
)
from fable.distributed.orchestrator import DistributedOrchestrator
from fable.distributed.topics import (
    disturbance_ack_topic,
    disturbance_request_topic,
    event_request_topic,
    event_response_topic,
    terminal_event_topic,
)
from .planning_policy import (
    ControllerPlanningContext,
    ControllerPlanningPolicy,
    FableControllerPlanningPolicy,
)
from .telemetry import NetworkTelemetrySource

LOGGER = logging.getLogger(__name__)

_SYMBOLIC_DISCOVERY_LOCATIONS = frozenset(
    {
        "chase_gate",
        "convoy_gate",
        "convergence_gate",
        "rendezvous_gate",
        "route_gate",
        "visit_reference",
    }
)

# Identity-gated departure discovery must remain bounded: enough independent
# cameras to avoid a first-source identity lottery, but not an all-deployment
# broadcast on large sites. Four covers the uncalibrated West Point replay
# deployment and stays within the hosted-VLM per-run budget.
_MAX_IDENTITY_DEPARTURE_SOURCES = 4


def _requires_source_discovery_fanout(demand: PredicateDemand) -> bool:
    """Return true when a frontier has no concrete observation source.

    Authored names such as ``convergence_gate`` describe a semantic place, not
    a calibrated physical camera.  Treating one cheap camera realization as
    the location is an unsupported guess and makes equal-cost planning depend
    on heartbeat/enumeration order.  A genuinely bound concrete location does
    not need discovery fan-out.
    """

    if len(demand.eligible_source_ids) <= 1:
        return False
    # An unbound EXITS observation immediately followed by SAME_ENTITY is a
    # candidate-producing frontier. Without a calibrated departure zone, a
    # cheapest-node choice makes identity success depend on node enumeration:
    # the recovered vehicle may leave through another eligible camera. Keep a
    # small physical source pool alive; the semantic identity gate, not source
    # ordering, decides which departure is valid. This function is consumed
    # only by the FABLE planning branch, so fixed/ablation policies are not
    # silently given discovery fan-out.
    if (
        demand.semantic_predicate.predicate_id == "EXITS"
        and bool(demand.unbound_roles)
    ):
        return True
    # A relational predicate with two or more unknown entities has no source
    # anchor either. Choosing one camera by cost/enumeration order is an
    # unsupported guess: discover candidate bindings at every eligible source,
    # then let normal hypothesis/cancellation semantics retain accepted facts.
    if len(demand.unbound_roles) >= 2:
        return True
    if "location" in demand.unbound_roles:
        return True
    return any(
        demand.bound_roles.get(role) in _SYMBOLIC_DISCOVERY_LOCATIONS
        for role in ("location", "reference", "zone")
    )


@dataclass
class _RequestState:
    submission: EventRequestSubmission
    family_id: str
    runtime: SemanticRuntime
    demand_context: DemandCompileContext
    discovery_hypothesis_id: UUID | None = None
    discovery_child_ids: set[UUID] = field(default_factory=set)
    discovery_candidates: list["_DiscoveryCandidate"] = field(default_factory=list)
    # A return observation may introduce a provisional track and expose an
    # exact SAME_ENTITY checkpoint.  Keep the producing demand alive until one
    # provisional child passes that identity gate; otherwise the first
    # time-valid (but visually wrong) tracker fragment permanently consumes
    # the return frontier.
    retained_identity_candidate_demands: dict[UUID, UUID] = field(
        default_factory=dict
    )
    retained_repeated_visit_candidates: dict[
        tuple[str, str], list[tuple[datetime, UUID]]
    ] = field(default_factory=dict)
    # B1 is an authored, whole-event pipeline: downstream providers are live
    # before their semantic nodes reach the active frontier.  Preserve a
    # bounded set of those early observations and re-envelope them only when
    # the authored node becomes active.  Adaptive policies never use this
    # bridge; their providers are dispatched from the current frontier.
    deferred_static_results: list[PredicateResult] = field(default_factory=list)
    terminal_occurrence_times: list[datetime] = field(default_factory=list)
    # Frame-level providers may repeat the same track-stable semantic fact.
    # Keep semantic progression occurrence-based: a recovered/ departing
    # vehicle is one candidate per source and canonical entity, not one fork
    # per frame carrying that track.
    robbery_candidate_observations: set[tuple[str, str, str]] = field(
        default_factory=set
    )
    robbery_candidate_counts: dict[tuple[str, str], int] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _DiscoveryCandidate:
    hypothesis_id: UUID
    partition: str
    identity: str
    admitted_order: int


@dataclass(frozen=True)
class ControllerPlanningEvent:
    """Public, bounded description of one admitted controller realization."""

    request_id: str
    hypothesis_id: UUID
    checkpoint_id: UUID
    policy_id: str
    trigger: str
    semantic_epoch: int
    resource_epoch: int
    selected_alternative_ids: tuple[str, ...]
    selected_chain_ids: tuple[str, ...]
    selected_node_ids: tuple[str, ...]
    activated_provider_keys: tuple[str, ...]
    predicted_completion_ms: int
    predicted_transfer_bytes: int
    reason: str
    commands: tuple[object, ...]


class FableController:
    """Authoritative end-to-end FABLE control loop."""

    def __init__(
        self,
        *,
        orchestrator: DistributedOrchestrator,
        provider_registry: ProviderRegistry,
        deployment_view: RuntimeDeploymentView,
        artifact_catalog: ArtifactCatalog | None = None,
        request_compiler: EventRequestCompiler | None = None,
        execution_profile: ExecutionProfile = ExecutionProfile.DEVELOPMENT,
        network_telemetry: NetworkTelemetrySource | None = None,
        planning_policies: Mapping[str, ControllerPlanningPolicy] | None = None,
        planning_event_sink: Callable[[ControllerPlanningEvent], None] | None = None,
        retrospective_policy_id: str = "R2_FABLE_TYPED_REPLAY",
    ) -> None:
        self.orchestrator = orchestrator
        self.providers = provider_registry
        self.deployment_view = deployment_view
        self.artifacts = artifact_catalog or ArtifactCatalog()
        self.request_compiler = request_compiler or EventRequestCompiler()
        self.execution_profile = execution_profile
        self.network_telemetry = network_telemetry
        self.planning_event_sink = planning_event_sink
        if retrospective_policy_id not in {
            "R0_NO_REPLAY",
            "R1_RAW_REPLAY",
            "R2_FABLE_TYPED_REPLAY",
        }:
            raise ValueError(
                f"unsupported retrospective policy {retrospective_policy_id!r}"
            )
        self.retrospective_policy_id = retrospective_policy_id
        default_policy = FableControllerPlanningPolicy()
        self.planning_policies: dict[str, ControllerPlanningPolicy] = {
            default_policy.policy_id: default_policy
        }
        if planning_policies is not None:
            self.planning_policies.update(
                {key: value for key, value in planning_policies.items()}
            )
        self.checkpoints = CheckpointController(
            lifecycle=orchestrator.lifecycle,
            artifact_catalog=self.artifacts,
        )
        self.requests: dict[str, _RequestState] = {}
        self._handlers_registered = False

    def bind(self) -> None:
        """Attach callbacks/subscriptions before the distributed orchestrator starts."""
        self.orchestrator.on_result = self.handle_result
        self.orchestrator.on_replan_required = self.handle_replan
        self.orchestrator.on_heartbeat = self.handle_heartbeat
        self.orchestrator.on_artifact = self.handle_artifact
        self._capacity_deferred: set[tuple[str, UUID]] = set()
        self._last_capacity_retry = 0.0
        if self._handlers_registered:
            return
        self.orchestrator.transport.subscribe(
            event_request_topic(self.orchestrator.orchestrator_id),
            self._on_event_request,
            qos=1,
        )
        self.orchestrator.transport.subscribe(
            disturbance_request_topic(self.orchestrator.orchestrator_id),
            self._on_disturbance_request,
            qos=1,
        )
        if self.network_telemetry is not None:
            self.network_telemetry.bind(self.orchestrator.transport, self.apply_network_updates)
        self._handlers_registered = True

    def submit_event(self, submission: EventRequestSubmission) -> EventRequestResponse:
        if submission.planning_policy_id not in self.planning_policies:
            return EventRequestResponse(
                request_message_id=submission.message_id,
                request_id=submission.request_id,
                accepted=False,
                reason=(
                    f"unknown planning_policy_id={submission.planning_policy_id}; "
                    f"available={sorted(self.planning_policies)}"
                ),
            )
        if submission.request_id in self.requests:
            state = self.requests[submission.request_id]
            return EventRequestResponse(
                request_message_id=submission.message_id,
                request_id=submission.request_id,
                accepted=True,
                hypothesis_ids=tuple(item.hypothesis_id for item in state.runtime.hypotheses),
                reason="request already active; no duplicate semantic runtime created",
            )

        compilation = self.request_compiler.compile(
            StructuredEventRequest(
                family_id=submission.family_id,
                parameters=submission.parameters,
            )
        )
        runtime = SemanticRuntime(
            compilation.graph,
            config=SemanticRuntimeConfig(
                request_id=submission.request_id,
                hypothesis_horizon_ms=submission.hypothesis_horizon_ms,
                deadline_offset_ms=submission.deadline_offset_ms,
            ),
        )
        context = DemandCompileContext(
            raw_data_must_remain_local=submission.raw_data_must_remain_local,
            allowed_node_ids=submission.allowed_node_ids,
            allowed_regions=submission.allowed_regions,
            maximum_transfer_bytes=submission.maximum_transfer_bytes,
        )
        state = _RequestState(
            submission=submission,
            family_id=compilation.family_id,
            runtime=runtime,
            demand_context=context,
        )
        self.requests[submission.request_id] = state
        self.orchestrator.store.put("tasks", submission.request_id, submission)
        self.orchestrator.store.put("graphs", submission.request_id, compilation.graph)

        transition = runtime.start(
            event_time_window=submission.event_time_window,
            observed_at=submission.submitted_at,
        )
        state.discovery_hypothesis_id = (
            transition.hypothesis_ids[0] if transition.hypothesis_ids else None
        )
        self._persist_runtime(state)
        admitted_plan_ids: list[UUID] = []
        command_ids: list[UUID] = []
        for frontier in transition.frontiers:
            plans, commands = self._plan_frontier(
                state, frontier.snapshot.hypothesis_id, trigger="ADMISSION"
            )
            admitted_plan_ids.extend(plans)
            command_ids.extend(commands)

        return EventRequestResponse(
            request_message_id=submission.message_id,
            request_id=submission.request_id,
            accepted=True,
            hypothesis_ids=transition.hypothesis_ids,
            admitted_plan_ids=tuple(admitted_plan_ids),
            command_message_ids=tuple(command_ids),
            reason=(
                f"compiled family={compilation.family_id} mode={compilation.mode.value}; "
                f"execution_profile={self.execution_profile.value}; "
                f"planning_policy={submission.planning_policy_id}"
            ),
        )

    def handle_result(self, result: PredicateResult) -> None:
        state = self.requests.get(result.request_id)
        if state is None:
            LOGGER.warning("result for unknown request=%s ignored by semantic controller", result.request_id)
            return
        if self._duplicate_robbery_candidate(state, result):
            return
        transition = state.runtime.apply(result)
        if transition.status.value in {"DUPLICATE", "STALE", "REJECTED", "NOOP"}:
            if (
                transition.status.value in {"STALE", "REJECTED"}
                and state.submission.planning_policy_id == "B1_HANDWRITTEN_STATIC"
                and not self._static_result_targets(state, result)
                and (
                    transition.status.value == "STALE"
                    or transition.reason == "result node is not part of the active checkpoint"
                )
            ):
                # A fixed B1 pipeline may observe a later authored predicate
                # before its predecessor advances. Do not silently lose that
                # evidence. The graph's ordinary temporal/binding checks are
                # still applied after the node becomes active.
                replacement_index = next(
                    (
                        index
                        for index, pending in enumerate(state.deferred_static_results)
                        if pending.graph_node_id == result.graph_node_id
                    ),
                    None,
                )
                if replacement_index is None:
                    state.deferred_static_results.append(result)
                else:
                    # B1 owns one authored hypothesis, not a discovery fan-out.
                    # Keep the most recent candidate for each semantic role so
                    # high-rate tracker fragments cannot multiply terminals.
                    state.deferred_static_results[replacement_index] = result
                del state.deferred_static_results[:-512]
                LOGGER.info(
                    "deferred early B1 result request=%s predicate=%s node=%s pending=%s",
                    result.request_id,
                    result.semantic_predicate.predicate_id,
                    result.graph_node_id,
                    len(state.deferred_static_results),
                )
                return
            LOGGER.info("semantic result not applied status=%s reason=%s", transition.status, transition.reason)
            return
        # Persist only state named by the authoritative transition.  The old
        # implementation rewrote every hypothesis and frontier for every
        # incoming observation, including stale and guard-rejected results.
        # Apart from making noisy discovery traffic increasingly expensive,
        # that work delayed still-valid observations until their deadlines.
        # RuntimeTransition is the mutation boundary: APPLIED updates its
        # parent, FORKED updates the parent and creates children, and MERGED
        # names the canonical child whose evidence changed.
        self._persist_transition(state, transition)

        next_demands: list[PredicateDemand] = []
        for frontier in transition.frontiers:
            hypothesis = state.runtime.get_hypothesis(frontier.snapshot.hypothesis_id)
            next_demands.extend(self._compile_frontier_demands(state, hypothesis.hypothesis_id))
        try:
            result_hypothesis = state.runtime.get_hypothesis(result.hypothesis_id)
            lifecycle = result_hypothesis.lifecycle
        except KeyError:
            lifecycle = HypothesisLifecycle.ACTIVE
        discovery_fork = (
            transition.status.value == "FORKED"
            and result.hypothesis_id == state.discovery_hypothesis_id
            and state.submission.max_seed_hypotheses > 1
        )
        if discovery_fork:
            state.discovery_child_ids.update(transition.hypothesis_ids)
            self._maintain_discovery_pool(state, result, transition.hypothesis_ids)
        preserve_discovery = (
            discovery_fork
            and (
                state.submission.seed_admission_strategy == "reference_bounded"
                or len(state.discovery_child_ids)
                < state.submission.max_seed_hypotheses
            )
        )
        preserve_identity_candidates = self._preserve_identity_candidate_frontier(
            state,
            result,
            transition,
            next_demands,
        )
        if preserve_discovery or preserve_identity_candidates:
            # The semantic runtime intentionally retains a fork context for
            # additional binding candidates. Keep the corresponding provider
            # leases as well. Completing/cancelling this checkpoint here would
            # make that runtime capability unreachable after the first camera.
            released: set[UUID] = set()
            LOGGER.info(
                "preserving candidate frontier request=%s hypothesis=%s "
                "discovery=%s identity_gated=%s children=%s limit=%s",
                result.request_id,
                result.hypothesis_id,
                preserve_discovery,
                preserve_identity_candidates,
                len(state.discovery_child_ids),
                state.submission.max_seed_hypotheses,
            )
        else:
            outcome = self.checkpoints.handle_predicate_result(
                result=result,
                transition=transition,
                request_id=result.request_id,
                hypothesis_id=result.hypothesis_id,
                next_demands=next_demands,
                continuation_artifact_ids=result.artifact_ids,
                hypothesis_lifecycle=lifecycle,
            )
            released = set(outcome.completed_lease_ids)
            if outcome.cancellation is not None:
                released.update(outcome.cancellation.released_lease_ids)

        # A successful exact identity result resolves the provisional-return
        # search. Retire the older PASSES/ENTERS demand which was deliberately
        # kept alive so later candidates could fork from its original
        # checkpoint.
        if result.semantic_predicate.predicate_id == "SAME_ENTITY":
            retained_demand_id = state.retained_identity_candidate_demands.pop(
                result.hypothesis_id,
                None,
            )
            if retained_demand_id is not None:
                released.update(
                    self.orchestrator.lifecycle.complete_demand(retained_demand_id)
                )
                for child_id, demand_id in tuple(
                    state.retained_identity_candidate_demands.items()
                ):
                    if demand_id == retained_demand_id:
                        state.retained_identity_candidate_demands.pop(child_id, None)
        for lease_id in sorted(released, key=str):
            managed = self.orchestrator.lifecycle.leases.get(lease_id)
            if managed is not None:
                self.orchestrator.dispatcher.send_cancel(
                    managed,
                    reason="semantic checkpoint resolved",
                )

        # Planning starts only after semantic state and cancellations are authoritative.
        for frontier in transition.frontiers:
            self._plan_frontier(
                state, frontier.snapshot.hypothesis_id, trigger="SEMANTIC_FRONTIER"
            )
        for hypothesis_id in transition.hypothesis_ids:
            hypothesis = state.runtime.get_hypothesis(hypothesis_id)
            if hypothesis.lifecycle == HypothesisLifecycle.COMPLETED:
                self._emit_completed_event(state, hypothesis)
        self._drain_static_results(state)

    @staticmethod
    def _duplicate_robbery_candidate(
        state: _RequestState,
        result: PredicateResult,
    ) -> bool:
        """Suppress repeated frames of one robbery vehicle occurrence.

        This is deliberately scoped to robbery's candidate-producing vehicle
        predicates. Distinct canonical entities and distinct sources remain
        independent hypotheses; only an identical track/source fact is
        idempotent. Other CE families retain their authored repeated-visit
        semantics.
        """

        if state.family_id != "robbery" or result.semantic_predicate.predicate_id not in {
            "VEHICLE_PRESENT_BEFORE",
            "EXITS",
        }:
            return False
        entity_id = next(
            (
                value
                for role, value in result.binding_delta.introduced.items()
                if "vehicle" in role
            ),
            None,
        )
        if not entity_id:
            return False
        source_id = (
            result.provenance.source_ids[0]
            if result.provenance.source_ids
            else result.provenance.node_id
        )
        # Replay invocations mint a fresh replay UUID while preserving the
        # source-local tracker identity: ``source:replay_uuid:track``. The
        # replay UUID identifies transport execution, not a new semantic
        # vehicle occurrence, so omit that middle component for deduplication.
        entity_parts = entity_id.rsplit(":", 2)
        semantic_entity_id = (
            f"{entity_parts[0]}:{entity_parts[2]}"
            if len(entity_parts) == 3
            else entity_id
        )
        key = (
            result.semantic_predicate.predicate_id,
            source_id,
            semantic_entity_id,
        )
        if key in state.robbery_candidate_observations:
            return True
        pool_key = (result.semantic_predicate.predicate_id, source_id)
        # A bounded candidate pool is part of the request contract. Replay-
        # local trackers can fragment one physical vehicle into arbitrarily
        # many IDs, so exact-ID deduplication alone is not a bound. Four
        # source-local representatives preserve multiple vehicles while
        # preventing frame-rate hypothesis growth.
        count = state.robbery_candidate_counts.get(pool_key, 0)
        if count >= 4:
            return True
        state.robbery_candidate_observations.add(key)
        state.robbery_candidate_counts[pool_key] = count + 1
        return False

    @staticmethod
    def _static_result_targets(
        state: _RequestState,
        result: PredicateResult,
    ) -> list[tuple[Hypothesis, object]]:
        """Return active B1 hypotheses whose current frontier accepts a node."""

        targets: list[tuple[Hypothesis, object]] = []
        for hypothesis in state.runtime.active_hypotheses:
            frontier = state.runtime.get_frontier(hypothesis.hypothesis_id)
            if (
                frontier is not None
                and result.graph_node_id in frontier.snapshot.enabled_node_ids
            ):
                targets.append((hypothesis, frontier))
        return targets

    def _drain_static_results(self, state: _RequestState) -> None:
        """Apply early B1 evidence once its authored frontier becomes active."""

        if (
            state.submission.planning_policy_id != "B1_HANDWRITTEN_STATIC"
            or not state.deferred_static_results
        ):
            return
        for pending in tuple(state.deferred_static_results):
            targets = self._static_result_targets(state, pending)
            if not targets:
                continue
            state.deferred_static_results.remove(pending)
            for hypothesis, frontier in targets:
                checkpoint = frontier.checkpoint_for_node(pending.graph_node_id)
                projected = pending.model_copy(
                    update={
                        "result_id": uuid7(),
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "expected_hypothesis_version": hypothesis.version,
                        "frontier_id": frontier.snapshot.frontier_id,
                        "checkpoint_id": checkpoint.checkpoint_id,
                    }
                )
                LOGGER.info(
                    "applying deferred B1 result request=%s predicate=%s node=%s hypothesis=%s",
                    pending.request_id,
                    pending.semantic_predicate.predicate_id,
                    pending.graph_node_id,
                    hypothesis.hypothesis_id,
                )
                self.handle_result(projected)

    def _preserve_identity_candidate_frontier(
        self,
        state: _RequestState,
        result: PredicateResult,
        transition: RuntimeTransition,
        next_demands: list[PredicateDemand],
    ) -> bool:
        """Retain a binding frontier while its provisional identity is tested.

        SemanticRuntime already retains the immutable fork context and can
        create another child from a later observation.  The controller must
        retain the physical producer lease as well or that capability becomes
        unreachable after the first candidate.  Scope this to immediate,
        exact SAME_ENTITY gates so unrelated completed checkpoints retain their
        normal cancellation behavior.
        """

        if transition.status.value != "FORKED" or not result.binding_delta.introduced:
            return False
        if not next_demands or any(
            demand.semantic_predicate.predicate_id != "SAME_ENTITY"
            or demand.unbound_roles
            for demand in next_demands
        ):
            return False
        repeated_visit_binding = next(
            (
                (role, entity_id)
                for role, entity_id in result.binding_delta.introduced.items()
                if role.startswith("visit_vehicle_")
            ),
            None,
        )
        robbery_departure_binding = next(
            (
                (role, entity_id)
                for role, entity_id in result.binding_delta.introduced.items()
                if role == "departing_vehicle"
            ),
            None,
        )
        # Both repeated visits and robbery departures are candidate-producing
        # checkpoints: the first time-valid tracker fragment is not guaranteed
        # to be the entity that satisfies the following exact SAME_ENTITY gate.
        # Robbery deliberately does *not* join the repeated-visit rolling pool
        # below.  Its producer remains open only until an identity result wins,
        # at which point handle_result completes the retained producer demand.
        # This preserves later EXITS candidates without exposing robbery to the
        # stalking-specific eviction/cancellation policy.
        if repeated_visit_binding is None and robbery_departure_binding is None:
            return False
        if repeated_visit_binding is not None:
            role, entity_id = repeated_visit_binding
            # Canonical local IDs are ``source:replay:track``. Partitioning by
            # the source keeps independent camera hypotheses alive.
            parts = entity_id.rsplit(":", 2)
            source_id = parts[0] if len(parts) == 3 else entity_id
            pool_key = (role, source_id)
            candidates = state.retained_repeated_visit_candidates.setdefault(
                pool_key,
                [],
            )
            candidate_time = result.event_time_interval.start
            nearby_candidates = sum(
                abs((candidate_time - existing_time).total_seconds()) < 10.0
                for existing_time, _ in candidates
            )
            redundant = nearby_candidates >= 3
            if redundant:
                reason = "duplicate repeated-visit tracker burst"
                for child_id in transition.hypothesis_ids:
                    if not state.runtime.invalidate_hypothesis(child_id):
                        continue
                    invalidated = state.runtime.get_hypothesis(child_id)
                    self.orchestrator.store.put(
                        "hypotheses",
                        str(invalidated.hypothesis_id),
                        invalidated,
                    )
                    LOGGER.info(
                        "discarded repeated-visit identity candidate "
                        "request=%s role=%s source=%s hypothesis=%s reason=%s",
                        result.request_id,
                        role,
                        source_id,
                        child_id,
                        reason,
                    )
                return True
            # Bound provisional identity work by the request's global seed
            # budget, divided fairly across eligible cameras. The previous
            # hard-coded 12-per-role-per-camera pool admitted 48 simultaneous
            # ReID branches in a four-camera deployment even when the global
            # seed budget was 20. Valid results then arrived stale.
            source_count = max(1, len(state.submission.allowed_node_ids))
            per_source_limit = max(
                2,
                (
                    state.submission.max_seed_hypotheses
                    + source_count
                    - 1
                )
                // source_count,
            )
            if len(candidates) >= per_source_limit:
                # This is a rolling pool, not a first-N pool. A correct return
                # can appear late in a long view. Retire the oldest unresolved
                # candidate and all of its physical leases before admitting
                # the new event-time-separated representative.
                _, evicted_child_id = candidates.pop(0)
                if state.runtime.invalidate_hypothesis(evicted_child_id):
                    invalidated = state.runtime.get_hypothesis(evicted_child_id)
                    self.orchestrator.store.put(
                        "hypotheses",
                        str(invalidated.hypothesis_id),
                        invalidated,
                    )
                evicted_demand_ids = {
                    managed.lease.demand_id
                    for managed in self.orchestrator.lifecycle.leases.values()
                    if managed.hypothesis_id == evicted_child_id
                }
                for demand_id in evicted_demand_ids:
                    # The identity worker retains semantic demand state after
                    # its short physical lease can already be RELEASED. Send
                    # cancellation at the demand boundary before consulting
                    # active leases so stale crop/ReID work is always retired.
                    self.orchestrator.dispatcher.send_identity_demand_cancel(
                        request_id=result.request_id,
                        demand_id=demand_id,
                        reason="rolling repeated-visit candidate eviction",
                    )
                    for lease_id in self.orchestrator.lifecycle.cancel_demand(demand_id):
                        managed = self.orchestrator.lifecycle.leases.get(lease_id)
                        if managed is not None:
                            self.orchestrator.dispatcher.send_cancel(
                                managed,
                                reason="rolling repeated-visit candidate eviction",
                            )
                state.retained_identity_candidate_demands.pop(
                    evicted_child_id,
                    None,
                )
                LOGGER.info(
                    "evicted repeated-visit identity candidate request=%s "
                    "role=%s source=%s hypothesis=%s limit=%s",
                    result.request_id,
                    role,
                    source_id,
                    evicted_child_id,
                    per_source_limit,
                )
            candidates.extend(
                (candidate_time, child_id)
                for child_id in transition.hypothesis_ids
            )
        for child_id in transition.hypothesis_ids:
            state.retained_identity_candidate_demands[child_id] = result.demand_id
        return True

    def handle_replan(self, node_id: str, demand_ids: tuple[UUID, ...], reason: str) -> None:
        # Node failure callbacks arrive after Phase 5 marked its instances failed.
        # Network-triggered replans use a synthetic node_id and only update links.
        hard_node_failure = node_id in self.deployment_view.base.nodes
        if hard_node_failure:
            self.deployment_view.set_node_availability(node_id, False)
        hypothesis_keys: set[tuple[str, UUID]] = set()
        demand_keys: dict[UUID, set[tuple[str, UUID]]] = {}
        for demand_id in demand_ids:
            static_request_ids = {
                managed.request_id
                for managed in self.orchestrator.lifecycle.leases.values()
                if managed.lease.demand_id == demand_id
                and (
                    (state := self.requests.get(managed.request_id)) is not None
                    and state.submission.planning_policy_id
                    == "B1_HANDWRITTEN_STATIC"
                )
            }
            if static_request_ids:
                LOGGER.info(
                    "B1 static pipeline ignores resource replan requests=%s "
                    "demand=%s reason=%s",
                    tuple(sorted(static_request_ids)),
                    demand_id,
                    reason,
                )
                continue
            keys: set[tuple[str, UUID]] = set()
            for managed in self.orchestrator.lifecycle.leases.values():
                if managed.lease.demand_id == demand_id:
                    keys.add((managed.request_id, managed.hypothesis_id))
            hypothesis_keys.update(keys)
            demand_keys[demand_id] = keys
            # A genuinely failed execution node cannot finish its invocation,
            # so retire it before planning elsewhere.  Soft resource and link
            # changes are different: cancelling first creates an evidence gap
            # while the replacement plan is being admitted.  Keep that valid
            # demand alive until a replacement is admitted below.
            if hard_node_failure:
                self._cancel_replanned_demand(demand_id, reason)
        replanned_keys: set[tuple[str, UUID]] = set()
        for request_id, hypothesis_id in sorted(hypothesis_keys, key=lambda item: (item[0], str(item[1]))):
            state = self.requests.get(request_id)
            if state is None:
                continue
            try:
                hypothesis = state.runtime.get_hypothesis(hypothesis_id)
            except KeyError:
                continue
            if hypothesis.lifecycle != HypothesisLifecycle.ACTIVE:
                continue
            LOGGER.info(
                "replanning request=%s hypothesis=%s reason=%s",
                request_id,
                hypothesis_id,
                reason,
            )
            admitted_plan_ids, _command_ids = self._plan_frontier(
                state, hypothesis_id, trigger=f"RESOURCE_EPOCH:{reason}"
            )
            if admitted_plan_ids:
                replanned_keys.add((request_id, hypothesis_id))

        if not hard_node_failure:
            for demand_id, keys in demand_keys.items():
                if keys and keys.issubset(replanned_keys):
                    self._cancel_replanned_demand(demand_id, reason)
                else:
                    LOGGER.info(
                        "retaining in-flight demand=%s during soft replan; "
                        "replacement was not admitted reason=%s",
                        demand_id,
                        reason,
                    )

    def _cancel_replanned_demand(self, demand_id: UUID, reason: str) -> None:
        released_lease_ids = self.orchestrator.lifecycle.cancel_demand(demand_id)
        for lease_id in released_lease_ids:
            managed = self.orchestrator.lifecycle.leases.get(lease_id)
            if managed is not None:
                self.orchestrator.dispatcher.send_cancel(
                    managed,
                    reason=f"replanning: {reason}",
                )

    def handle_heartbeat(self, heartbeat: NodeHeartbeat) -> None:
        try:
            self.deployment_view.record_heartbeat(heartbeat)
        except KeyError:
            LOGGER.debug("heartbeat node=%s not present in planning deployment", heartbeat.node_id)
            return
        if not self._capacity_deferred or heartbeat.availability != NodeAvailability.AVAILABLE:
            return
        now = time.monotonic()
        if now - self._last_capacity_retry < 2.0:
            return
        self._last_capacity_retry = now
        for request_id, hypothesis_id in tuple(sorted(
            self._capacity_deferred, key=lambda item: (item[0], str(item[1]))
        )):
            state = self.requests.get(request_id)
            if state is None:
                self._capacity_deferred.discard((request_id, hypothesis_id))
                continue
            admitted, _commands = self._plan_frontier(
                state, hypothesis_id, trigger="CAPACITY_RETRY"
            )
            if admitted:
                self._capacity_deferred.discard((request_id, hypothesis_id))

    def handle_artifact(self, announcement: ArtifactAnnouncement) -> None:
        self.artifacts.register(announcement.artifact, replace=True)

    def apply_network_updates(
        self,
        updates: tuple[RuntimeLinkUpdate, ...],
        reason: str = "runtime network state changed",
    ) -> None:
        valid: list[RuntimeLinkUpdate] = []
        for update in updates:
            try:
                self.deployment_view.base.node(update.source_node_id)
                self.deployment_view.base.node(update.target_node_id)
                valid.append(update)
            except KeyError:
                LOGGER.debug(
                    "runtime network link %s->%s is outside deployment",
                    update.source_node_id,
                    update.target_node_id,
                )
        changed = self.deployment_view.update_links(valid) if valid else False
        if changed:
            demand_ids = tuple(
                sorted(
                    {lease.lease.demand_id for lease in self.orchestrator.lifecycle.active_leases},
                    key=str,
                )
            )
            if demand_ids:
                self.handle_replan("network", demand_ids, reason)

    def apply_disturbance(
        self, request: RuntimeDisturbanceRequest
    ) -> RuntimeDisturbanceAck:
        """Apply an acknowledged E4/runtime disturbance to authoritative state."""

        active = tuple(self.orchestrator.lifecycle.active_leases)
        affected_demand_ids = tuple(
            sorted({item.lease.demand_id for item in active}, key=str)
        )
        replanned_request_ids = tuple(
            sorted({item.request_id for item in active})
        )
        previous = self.deployment_view.resource_epoch
        try:
            changed, previous_epoch, resource_epoch = self.deployment_view.apply_updates(
                node_updates=request.node_updates,
                link_updates=request.link_updates,
            )
        except Exception as exc:
            return RuntimeDisturbanceAck(
                request_message_id=request.message_id,
                disturbance_id=request.disturbance_id,
                accepted=False,
                previous_resource_epoch=previous,
                resource_epoch=self.deployment_view.resource_epoch,
                reason=str(exc),
            )
        if changed and affected_demand_ids:
            self.handle_replan(
                "runtime-disturbance",
                affected_demand_ids,
                request.reason or f"disturbance {request.disturbance_id}",
            )
        return RuntimeDisturbanceAck(
            request_message_id=request.message_id,
            disturbance_id=request.disturbance_id,
            accepted=True,
            changed=changed,
            previous_resource_epoch=previous_epoch,
            resource_epoch=resource_epoch,
            affected_demand_ids=affected_demand_ids,
            replanned_request_ids=(replanned_request_ids if changed else ()),
            reason=(
                request.reason
                if changed
                else "disturbance accepted; runtime deployment state was unchanged"
            ),
        )

    def _plan_frontier(
        self,
        state: _RequestState,
        hypothesis_id: UUID,
        *,
        trigger: str = "SEMANTIC_FRONTIER",
    ) -> tuple[tuple[UUID, ...], tuple[UUID, ...]]:
        hypothesis = state.runtime.get_hypothesis(hypothesis_id)
        if hypothesis.lifecycle != HypothesisLifecycle.ACTIVE:
            return (), ()
        demands = self._compile_frontier_demands(state, hypothesis_id)
        demands = self._apply_retrospective_demand_policy(demands)
        if not demands:
            return (), ()
        policy = self.planning_policies[state.submission.planning_policy_id]
        expand_admission = getattr(policy, "expand_admission_demands", None)
        if trigger == "ADMISSION" and expand_admission is not None:
            expanded = expand_admission(
                semantic_graph=state.runtime.graph,
                hypothesis=hypothesis,
                demand_context=state.demand_context,
                deployment=self.deployment_view.snapshot(),
                trace_id=state.submission.trace_id,
                placement_id=state.submission.baseline_placement_id,
                active_demands=demands,
            )
            if expanded is not None:
                demands = expanded
        constrain_demands = getattr(policy, "constrain_frontier_demands", None)
        if constrain_demands is not None:
            demands = constrain_demands(
                trace_id=state.submission.trace_id,
                placement_id=state.submission.baseline_placement_id,
                demands=demands,
            )
        frontier = state.runtime.get_frontier(hypothesis_id)
        if frontier is None:
            # A concurrent provider result may complete the hypothesis after
            # demand compilation but before checkpoint grouping.
            return (), ()
        by_checkpoint: dict[tuple[UUID, UUID | None], list[PredicateDemand]] = defaultdict(list)
        for demand in demands:
            try:
                checkpoint = frontier.checkpoint_for_node(demand.graph_node_id)
            except KeyError:
                # Static whole-event admission deliberately includes future
                # semantic stages. Their providers start now, but their
                # observations remain subject to normal frontier validation.
                checkpoint = None
            # OR children are independently executable alternatives. Treating
            # the full OR checkpoint as a sequential planning batch makes one
            # unavailable branch invalidate every otherwise viable watcher.
            source_discovery_partition = (
                hypothesis_id == state.discovery_hypothesis_id
                and len(demand.eligible_source_ids) == 1
            )
            branch_key = (
                demand.demand_id
                if (
                    (
                        checkpoint is not None
                        and checkpoint.kind == CheckpointKind.OR_RESOLUTION
                    )
                    or source_discovery_partition
                )
                else None
            )
            by_checkpoint[(demand.checkpoint_id, branch_key)].append(demand)

        admitted: list[UUID] = []
        commands: list[UUID] = []
        deployment = self.deployment_view.snapshot()
        for planning_key in sorted(by_checkpoint, key=str):
            checkpoint_id, _branch_id = planning_key
            checkpoint_demands = tuple(by_checkpoint[planning_key])
            raw_retrospective_control = (
                self.retrospective_policy_id == "R1_RAW_REPLAY"
                and any(
                    demand.retrospective_context
                    for demand in checkpoint_demands
                )
            )
            graph = PhysicalAlternativeGraphBuilder(
                provider_registry=self.providers,
                artifact_catalog=self.artifacts,
                deployment=deployment,
                # REAL execution may span workers when every intermediate
                # artifact has an explicit, typed transfer contract.  The
                # fail-closed validation in _filter_runtime_realizations()
                # remains the authority; forcing colocation here would prune
                # valid sensor-crop -> site-ReID chains before that validator
                # can inspect their broker scopes and topics.
                config=AlternativeBuildConfig(
                    # R1 has no artifact-transfer/router adaptation: its raw
                    # recording pipeline is a fixed source-local control.
                    # Enumerate that executable realization directly instead
                    # of filling the bounded candidate set with cross-worker
                    # combinations which the real executor must later reject.
                    require_internal_step_colocation=raw_retrospective_control,
                ),
                placement_eligible=self._runtime_placement_eligible,
            ).build(checkpoint_demands)
            graph = self._filter_runtime_realizations(graph)
            if raw_retrospective_control:
                graph = self._retain_raw_retrospective_realizations(graph)
            if not graph.alternatives:
                LOGGER.warning(
                    "frontier alternatives exhausted request=%s hypothesis=%s "
                    "checkpoint=%s predicates=%s pruning_counts=%s pruning_samples=%s",
                    state.submission.request_id,
                    hypothesis_id,
                    checkpoint_id,
                    tuple(
                        demand.semantic_predicate.predicate_id
                        for demand in checkpoint_demands
                    ),
                    dict(Counter(item.code for item in graph.pruned)),
                    tuple(
                        (item.chain_id, item.code, item.reason)
                        for item in graph.pruned[:12]
                    ),
                )
            decision = policy.select(
                ControllerPlanningContext(
                    request_id=state.submission.request_id,
                    trace_id=state.submission.trace_id,
                    placement_id=state.submission.baseline_placement_id,
                    family_id=state.family_id,
                    hypothesis_id=hypothesis_id,
                    semantic_epoch=hypothesis.version,
                    resource_epoch=self.deployment_view.resource_epoch,
                    checkpoint_id=checkpoint_id,
                    hypothesis=hypothesis,
                    semantic_graph=state.runtime.graph,
                    demand_context=state.demand_context,
                    frontier_demands=checkpoint_demands,
                    frontier_graph=graph,
                    deployment=deployment,
                    provider_registry=self.providers,
                    artifact_catalog=self.artifacts,
                    runtime_provider_keys=frozenset(
                        (runtime.node_id, runtime.provider_id)
                        for runtime in self.orchestrator.runtime_resolver.runtimes
                        if not (
                            self.execution_profile == ExecutionProfile.REAL
                            and runtime.mode == RuntimeMode.REFERENCE
                        )
                    ),
                )
            )
            if decision.allowed_alternative_ids is not None:
                allowed_ids = set(decision.allowed_alternative_ids)
                graph = graph.model_copy(
                    update={
                        "alternatives": tuple(
                            item for item in graph.alternatives
                            if item.alternative_id in allowed_ids
                        )
                    }
                )
            LOGGER.info(
                "planning policy request=%s checkpoint=%s policy=%s alternatives=%s reason=%s",
                state.submission.request_id,
                checkpoint_id,
                decision.policy_id,
                len(graph.alternatives),
                decision.reason,
            )
            planner = BoundedLabelPlanner(
                provider_registry=self.providers,
                artifact_catalog=self.artifacts,
                deployment=deployment,
            )
            planning_graphs = (graph,)
            discovery_demand = checkpoint_demands[0] if len(checkpoint_demands) == 1 else None
            if (
                decision.policy_id == "FABLE"
                and discovery_demand is not None
                and _requires_source_discovery_fanout(discovery_demand)
            ):
                # With no bound location or spatial prior, selecting one cheap
                # sensor is not evidence discovery: it is an unsupported guess.
                # Launch the best realization at each eligible physical source;
                # normal OR/cancellation semantics stop the siblings after the
                # first accepted binding.
                alternatives_by_node: dict[str, list] = defaultdict(list)
                for alternative in graph.alternatives:
                    if alternative.step_placements:
                        alternatives_by_node[
                            alternative.step_placements[0].node_id
                        ].append(alternative)
                if len(alternatives_by_node) > 1:
                    node_items = sorted(alternatives_by_node.items())
                    if (
                        discovery_demand.semantic_predicate.predicate_id == "EXITS"
                        and discovery_demand.unbound_roles
                    ):
                        node_items = node_items[:_MAX_IDENTITY_DEPARTURE_SOURCES]
                    planning_graphs = tuple(
                        graph.model_copy(update={"alternatives": tuple(items)})
                        for _node_id, items in node_items
                    )
            capacity_deferred = False
            checkpoint_admitted = False
            for planning_graph in planning_graphs:
                remaining_graph = planning_graph
                while remaining_graph.alternatives:
                    LOGGER.info(
                        "planning search start request=%s checkpoint=%s alternatives=%s",
                        state.submission.request_id,
                        checkpoint_id,
                        len(remaining_graph.alternatives),
                    )
                    result = planner.search(remaining_graph, checkpoint_demands)
                    LOGGER.info(
                        "planning search complete request=%s checkpoint=%s selected=%s",
                        state.submission.request_id,
                        checkpoint_id,
                        result.selected is not None,
                    )
                    if result.selected is None or result.execution_plan is None:
                        LOGGER.warning(
                            "no feasible plan request=%s hypothesis=%s checkpoint=%s "
                            "phase3_failures=%s search_boundaries=%s",
                            state.submission.request_id,
                            hypothesis_id,
                            checkpoint_id,
                            result.trace.phase3_pruning,
                            result.trace.boundaries,
                        )
                        break
                    candidate = candidate_from_search_result(
                        result,
                        remaining_graph,
                        checkpoint_demands,
                        task_policy=TaskSchedulingPolicy(request_id=state.submission.request_id),
                    )
                    LOGGER.info(
                        "candidate submission start request=%s checkpoint=%s candidate=%s",
                        state.submission.request_id,
                        checkpoint_id,
                        candidate.candidate_id,
                    )
                    batch, emitted_commands = self.orchestrator.submit_candidates((candidate,))
                    LOGGER.info(
                        "candidate submission complete request=%s checkpoint=%s candidate=%s",
                        state.submission.request_id,
                        checkpoint_id,
                        candidate.candidate_id,
                    )
                    for record in batch.records:
                        LOGGER.info(
                            "admission decision request=%s checkpoint=%s candidate=%s "
                            "decision=%s reason=%s plan=%s",
                            state.submission.request_id,
                            checkpoint_id,
                            record.candidate_id,
                            record.decision.value,
                            record.reason,
                            record.plan_id,
                        )
                        if record.decision.value == "DEFERRED":
                            capacity_deferred = True
                    admitted.extend(batch.admitted_plan_ids)
                    commands.extend(command.message_id for command in emitted_commands)
                    if batch.admitted_plan_ids:
                        self._emit_planning_event(
                            state=state,
                            hypothesis_id=hypothesis_id,
                            checkpoint_id=checkpoint_id,
                            policy_id=decision.policy_id,
                            trigger=trigger,
                            selected=result.selected.selected_alternative_ids,
                            graph=remaining_graph,
                            reason=decision.reason,
                            commands=tuple(emitted_commands),
                        )
                        checkpoint_admitted = True
                        break

                    # Planning feasibility is evaluated against the deployment
                    # snapshot, while admission also accounts for leases already
                    # active in this cell. If the best realization loses that race,
                    # retry the remaining physical alternatives instead of silently
                    # abandoning an otherwise feasible semantic demand.
                    rejected_ids = set(result.selected.selected_alternative_ids)
                    remaining_graph = remaining_graph.model_copy(
                        update={
                            "alternatives": tuple(
                                alternative
                                for alternative in remaining_graph.alternatives
                                if alternative.alternative_id not in rejected_ids
                            )
                        }
                    )
            if capacity_deferred and not checkpoint_admitted:
                self._capacity_deferred.add((state.submission.request_id, hypothesis_id))
            elif checkpoint_admitted:
                self._capacity_deferred.discard((state.submission.request_id, hypothesis_id))
        return tuple(admitted), tuple(commands)

    def _emit_planning_event(
        self,
        *,
        state: _RequestState,
        hypothesis_id: UUID,
        checkpoint_id: UUID,
        policy_id: str,
        trigger: str,
        selected: tuple[str, ...],
        graph: PhysicalAlternativeGraph,
        reason: str,
        commands: tuple[object, ...],
    ) -> None:
        if self.planning_event_sink is None:
            return
        by_id = {item.alternative_id: item for item in graph.alternatives}
        alternatives = tuple(by_id[item] for item in selected if item in by_id)
        event = ControllerPlanningEvent(
            request_id=state.submission.request_id,
            hypothesis_id=hypothesis_id,
            checkpoint_id=checkpoint_id,
            policy_id=policy_id,
            trigger=trigger,
            semantic_epoch=state.runtime.get_hypothesis(hypothesis_id).version,
            resource_epoch=self.deployment_view.resource_epoch,
            selected_alternative_ids=selected,
            selected_chain_ids=tuple(item.chain_id for item in alternatives),
            selected_node_ids=tuple(sorted({
                step.node_id for item in alternatives for step in item.step_placements
            })),
            activated_provider_keys=tuple(sorted({
                f"{step.provider_id}@{step.node_id}"
                for item in alternatives for step in item.step_placements
            })),
            predicted_completion_ms=max(
                (item.estimated_completion_ms for item in alternatives), default=0
            ),
            predicted_transfer_bytes=sum(
                item.estimated_transfer_bytes for item in alternatives
            ),
            reason=reason,
            commands=commands,
        )
        try:
            self.planning_event_sink(event)
        except Exception:
            LOGGER.exception(
                "planning telemetry sink failed request=%s checkpoint=%s",
                state.submission.request_id,
                checkpoint_id,
            )

    def _compile_frontier_demands(
        self,
        state: _RequestState,
        hypothesis_id: UUID,
    ) -> tuple[PredicateDemand, ...]:
        hypothesis = state.runtime.get_hypothesis(hypothesis_id)
        frontier = state.runtime.get_frontier(hypothesis_id)
        if frontier is None or hypothesis.lifecycle != HypothesisLifecycle.ACTIVE:
            return ()
        compiler = DemandCompiler(
            predicate_registry=default_predicate_registry(),
            deployment=self.deployment_view.snapshot(),
        )
        demands = compiler.compile_frontier(
            graph=state.runtime.graph,
            hypothesis=hypothesis,
            frontier=frontier,
            context=state.demand_context,
        )
        if (
            hypothesis_id == state.discovery_hypothesis_id
            and len(demands) == 1
            and _requires_source_discovery_fanout(demands[0])
        ):
            demand = demands[0]
            deployment = self.deployment_view.snapshot()
            partitioned: list[PredicateDemand] = []
            for source_id in demand.eligible_source_ids:
                source = deployment.sources.get(source_id)
                if source is None:
                    continue
                constraints = demand.hard_constraints
                if constraints.raw_data_must_remain_local:
                    constraints = constraints.model_copy(
                        update={"allowed_node_ids": (source.node_id,)}
                    )
                payload = demand.model_dump(mode="python")
                payload.update(
                    {
                        "demand_id": uuid7(),
                        "eligible_source_ids": (source_id,),
                        "source_preferences": tuple(
                            item
                            for item in demand.source_preferences
                            if item.source_id == source_id
                        ),
                        # Source discovery partitions evidence acquisition by
                        # camera.  It constrains execution to that camera only
                        # when raw locality is required; otherwise trusted-edge
                        # offload must remain a physical planning alternative.
                        "hard_constraints": constraints,
                        "sharing_key": None,
                    }
                )
                partitioned.append(PredicateDemand.model_validate(payload))
            if partitioned:
                return tuple(partitioned)
        return demands

    def _maintain_discovery_pool(
        self,
        state: _RequestState,
        result: PredicateResult,
        child_ids: tuple[UUID, ...],
    ) -> None:
        """Keep deployed multi-seed discovery bounded and camera-fair."""

        if state.submission.seed_admission_strategy != "reference_bounded":
            return
        introduced = result.binding_delta.introduced
        partition = str(
            introduced.get("reference")
            or introduced.get("location")
            or result.provenance.node_id
        )
        identity = "|".join(
            f"{role}={value}"
            for role, value in sorted(introduced.items())
            if role not in {"reference", "location", "zone"}
        )
        active = []
        for candidate in state.discovery_candidates:
            try:
                hypothesis = state.runtime.get_hypothesis(candidate.hypothesis_id)
            except KeyError:
                continue
            if hypothesis.lifecycle == HypothesisLifecycle.ACTIVE:
                active.append(candidate)
        state.discovery_candidates = active

        for child_id in child_ids:
            duplicate = next(
                (
                    item
                    for item in state.discovery_candidates
                    if item.partition == partition
                    and identity
                    and item.identity == identity
                ),
                None,
            )
            if duplicate is not None:
                self._evict_discovery_candidate(
                    state, child_id, reason="duplicate camera-local discovery identity"
                )
                continue
            state.discovery_candidates.append(
                _DiscoveryCandidate(
                    hypothesis_id=child_id,
                    partition=partition,
                    identity=identity,
                    admitted_order=len(state.discovery_candidates),
                )
            )

        source_count = max(1, len(state.submission.allowed_node_ids))
        per_partition_limit = max(
            1,
            (state.submission.max_seed_hypotheses + source_count - 1)
            // source_count,
        )
        while True:
            counts: dict[str, int] = defaultdict(int)
            for item in state.discovery_candidates:
                counts[item.partition] += 1
            overfull = next(
                (
                    name
                    for name, count in sorted(counts.items())
                    if count > per_partition_limit
                ),
                None,
            )
            if overfull is None and len(state.discovery_candidates) <= state.submission.max_seed_hypotheses:
                break
            eligible = [
                item
                for item in state.discovery_candidates
                if overfull is None or item.partition == overfull
            ]
            victim = min(eligible, key=lambda item: self._discovery_eviction_key(state, item))
            self._evict_discovery_candidate(
                state,
                victim.hypothesis_id,
                reason="rolling camera discovery pool replacement",
            )

    @staticmethod
    def _discovery_eviction_key(
        state: _RequestState,
        candidate: _DiscoveryCandidate,
    ) -> tuple[int, int]:
        """Rank the least useful discovery candidate for eviction.

        Semantic progress is authoritative.  On equal progress, retain the
        older causal seed and evict the newest tracker fragment.  Long-lived
        events such as repeated visits and robbery need the first observation
        to remain available while later evidence arrives; evicting the oldest
        equal-progress seed continuously reset those graphs before identity
        association could advance them.

        This does not make the pool unbounded: a later candidate that advances
        farther than an older false seed still wins on the primary progress
        key, while equal, unproven fragments remain bounded by the configured
        pool size.
        """
        hypothesis = state.runtime.get_hypothesis(candidate.hypothesis_id)
        progress = sum(
            node.status.value == "SATISFIED"
            for node in hypothesis.node_states.values()
        )
        return progress, -candidate.admitted_order

    def _evict_discovery_candidate(
        self,
        state: _RequestState,
        hypothesis_id: UUID,
        *,
        reason: str,
    ) -> None:
        if not state.runtime.invalidate_hypothesis(hypothesis_id):
            return
        # Invalidation happens after the result transition has been persisted,
        # so write this one changed record explicitly.  Otherwise a controller
        # restart can resurrect a candidate that the rolling discovery pool
        # already evicted and recreate the same evidence flood.
        invalidated = state.runtime.get_hypothesis(hypothesis_id)
        self.orchestrator.store.put(
            "hypotheses", str(invalidated.hypothesis_id), invalidated
        )
        state.discovery_child_ids.discard(hypothesis_id)
        state.discovery_candidates = [
            item
            for item in state.discovery_candidates
            if item.hypothesis_id != hypothesis_id
        ]
        demand_ids = {
            managed.lease.demand_id
            for managed in self.orchestrator.lifecycle.active_leases
            if managed.request_id == state.submission.request_id
            and managed.hypothesis_id == hypothesis_id
        }
        for demand_id in demand_ids:
            leases = tuple(
                managed
                for managed in self.orchestrator.lifecycle.active_leases
                if managed.lease.demand_id == demand_id
            )
            for managed in leases:
                self.orchestrator.dispatcher.send_cancel(managed, reason=reason)
            self.orchestrator.lifecycle.cancel_demand(demand_id)
        LOGGER.info(
            "evicted discovery candidate request=%s hypothesis=%s reason=%s",
            state.submission.request_id,
            hypothesis_id,
            reason,
        )

    def _filter_runtime_realizations(self, graph: PhysicalAlternativeGraph) -> PhysicalAlternativeGraph:
        allowed = []
        pruned = list(graph.pruned)
        for alternative in graph.alternatives:
            reason = ""
            for placement in alternative.step_placements:
                if not self.orchestrator.runtime_resolver.has(
                    placement.node_id, placement.provider_id
                ):
                    reason = (
                        f"no deployed runtime for {placement.provider_id} on {placement.node_id}"
                    )
                    break
                runtime = self.orchestrator.runtime_resolver.resolve(
                    node_id=placement.node_id,
                    provider_id=placement.provider_id,
                )
                if self.execution_profile == ExecutionProfile.REAL and runtime.mode == RuntimeMode.REFERENCE:
                    reason = (
                        f"REFERENCE runtime {placement.provider_id}@{placement.node_id} is forbidden "
                        "by FABLE_EXECUTION_PROFILE=real"
                    )
                    break
            prune_code = "RUNTIME_UNAVAILABLE"
            if not reason and self.execution_profile == ExecutionProfile.REAL:
                # The deployed executor can faithfully hand intermediate data
                # between logical steps only when the steps live inside the same
                # physical worker. Cross-worker transfer/remote-reference is still
                # modeled by the planner and exercised in PLUMBING mode, but REAL
                # runs reject it until a concrete artifact-transfer backend is
                # configured. This keeps experimental claims aligned with what the
                # executor actually performs.
                placement_by_step = {item.step_id: item for item in alternative.step_placements}
                worker_by_step = {
                    item.step_id: self.orchestrator.runtime_resolver.worker_key(
                        item.node_id, item.provider_id
                    )
                    for item in alternative.step_placements
                }
                for transfer in alternative.transfers:
                    source_step_id = transfer.source_ref.split(".", 1)[0]
                    if source_step_id not in placement_by_step:
                        continue  # live/deployment/retained external input
                    target_step_id = transfer.target_step_id
                    if target_step_id not in worker_by_step:
                        continue
                    if worker_by_step[source_step_id] != worker_by_step[target_step_id]:
                        source_placement = placement_by_step[source_step_id]
                        target_placement = placement_by_step[target_step_id]
                        if self.orchestrator.runtime_resolver.supports_artifact_topic_transfer(
                            source_node_id=source_placement.node_id,
                            source_provider_id=source_placement.provider_id,
                            target_node_id=target_placement.node_id,
                            target_provider_id=target_placement.provider_id,
                            data_type=transfer.data_type,
                        ):
                            continue
                        reason = (
                            "cross-worker intermediate dataflow "
                            f"{source_step_id}({source_placement.provider_id}@"
                            f"{source_placement.node_id})->{target_step_id}("
                            f"{target_placement.provider_id}@{target_placement.node_id}) "
                            f"type={transfer.data_type} is not executable in "
                            "FABLE_EXECUTION_PROFILE=real without a transfer backend"
                        )
                        prune_code = "UNSUPPORTED_EXECUTION_PATH"
                        break
            if not reason:
                allowed.append(alternative)
                continue
            pruned.append(
                PrunedAlternative(
                    candidate_id=deterministic_id(
                        "runtime_pruned",
                        {"alternative_id": alternative.alternative_id, "reason": reason},
                    ),
                    demand_id=alternative.demand_id,
                    chain_id=alternative.chain_id,
                    code=prune_code,
                    reason=reason,
                )
            )
        return graph.model_copy(update={"alternatives": tuple(allowed), "pruned": tuple(pruned)})

    def _apply_retrospective_demand_policy(
        self, demands: tuple[PredicateDemand, ...]
    ) -> tuple[PredicateDemand, ...]:
        if self.retrospective_policy_id != "R0_NO_REPLAY":
            return demands
        return tuple(
            demand for demand in demands if not demand.retrospective_context
        )

    def _retain_raw_retrospective_realizations(
        self, graph: PhysicalAlternativeGraph
    ) -> PhysicalAlternativeGraph:
        """Keep only realizations backed by retained raw sensor input.

        R1 is an explicit raw-replay control, not a second adaptive policy. It
        may use deployment metadata, but it cannot satisfy historical work
        from a compact derived continuation artifact.
        """

        retained = []
        pruned = list(graph.pruned)
        for alternative in graph.alternatives:
            has_retained_raw = any(
                item.kind
                in {
                    ExecutionInputKind.RETAINED_ARTIFACT,
                    ExecutionInputKind.LIVE_SOURCE,
                }
                and self.providers.data_type(item.data_type).kind == "raw_sensor"
                for item in alternative.external_inputs
            )
            if has_retained_raw:
                retained.append(alternative)
                continue
            pruned.append(
                PrunedAlternative(
                    candidate_id=deterministic_id(
                        "retrospective_policy_pruned",
                        {
                            "alternative_id": alternative.alternative_id,
                            "policy": self.retrospective_policy_id,
                        },
                    ),
                    demand_id=alternative.demand_id,
                    chain_id=alternative.chain_id,
                    code="RETROSPECTIVE_POLICY",
                    reason="R1_RAW_REPLAY requires a retained raw sensor input",
                )
            )
        return graph.model_copy(
            update={"alternatives": tuple(retained), "pruned": tuple(pruned)}
        )

    def _runtime_placement_eligible(self, node_id: str, provider_id: str) -> bool:
        """Reject non-executable placements before bounded enumeration.

        Post-enumeration filtering remains as a fail-closed validation boundary,
        but cannot be the first runtime check: otherwise unavailable placements
        can exhaust the bounded candidate budget and starve valid realizations.
        """
        resolver = self.orchestrator.runtime_resolver
        if not resolver.has(node_id, provider_id):
            return False
        runtime = resolver.resolve(node_id=node_id, provider_id=provider_id)
        return not (
            self.execution_profile == ExecutionProfile.REAL
            and runtime.mode == RuntimeMode.REFERENCE
        )

    def _persist_runtime(self, state: _RequestState) -> None:
        for hypothesis in state.runtime.hypotheses:
            self.orchestrator.store.put("hypotheses", str(hypothesis.hypothesis_id), hypothesis)
            frontier = state.runtime.get_frontier(hypothesis.hypothesis_id)
            if frontier is not None:
                self.orchestrator.store.put("frontiers", str(frontier.snapshot.frontier_id), frontier)

    def _persist_transition(
        self,
        state: _RequestState,
        transition: RuntimeTransition,
    ) -> None:
        """Persist exactly the hypotheses mutated by one semantic transition.

        Result IDs and occurrence IDs used for process-local idempotency remain
        owned by ``SemanticRuntime``.  Durable predicate-result receipt happens
        before this controller callback, so skipping a semantic snapshot for a
        no-op transition does not weaken result durability or acknowledgments.
        """

        hypothesis_ids = set(transition.hypothesis_ids)
        if transition.parent_hypothesis_id is not None:
            hypothesis_ids.add(transition.parent_hypothesis_id)
        for hypothesis_id in sorted(hypothesis_ids, key=str):
            try:
                hypothesis = state.runtime.get_hypothesis(hypothesis_id)
            except KeyError:
                # A transition may refer to a parent retired by a merge.  Its
                # canonical child is still present in ``hypothesis_ids``.
                continue
            self.orchestrator.store.put(
                "hypotheses", str(hypothesis.hypothesis_id), hypothesis
            )
            frontier = state.runtime.get_frontier(hypothesis.hypothesis_id)
            if frontier is not None:
                self.orchestrator.store.put(
                    "frontiers", str(frontier.snapshot.frontier_id), frontier
                )

    def _emit_completed_event(self, state: _RequestState, hypothesis: Hypothesis) -> None:
        event = TerminalComplexEvent(
            request_id=hypothesis.request_id,
            family_id=state.family_id,
            hypothesis_id=hypothesis.hypothesis_id,
            graph_hash=hypothesis.graph_hash,
            bindings={
                role: binding.canonical_entity_id
                for role, binding in sorted(hypothesis.role_bindings.items())
            },
            event_time_window=hypothesis.event_time_window,
            provenance_result_ids=hypothesis.provenance_result_ids,
        )
        event_key = self._terminal_event_key(state, hypothesis)
        occurrence_time = self._terminal_evidence_time(hypothesis)
        rearm_ms = self._terminal_rearm_interval_ms(state)
        if rearm_ms is not None and any(
            abs((occurrence_time - previous).total_seconds())
            <= rearm_ms / 1000.0
            for previous in state.terminal_occurrence_times
        ):
            LOGGER.info(
                "suppressed terminal during scene-clear rearm request=%s "
                "hypothesis=%s occurrence_time=%s rearm_ms=%s",
                event.request_id,
                hypothesis.hypothesis_id,
                occurrence_time.isoformat(),
                rearm_ms,
            )
            return
        if not self.orchestrator.emit_complex_event_once(
            event_key, event.model_dump(mode="json", exclude_none=True)
        ):
            return
        state.terminal_occurrence_times.append(occurrence_time)
        self.orchestrator.store.put("terminal_events", str(event.message_id), event)
        self.orchestrator.transport.publish(
            terminal_event_topic(event.request_id),
            encode_model(event),
            qos=1,
            retain=False,
        )
        if state.submission.planning_policy_id == "B1_HANDWRITTEN_STATIC":
            # Future-stage B1 watches are intentionally admitted before their
            # semantic frontier exists, so ordinary checkpoint cleanup cannot
            # name all of them. Completion is the request-wide lifecycle
            # boundary for this fixed pipeline: retire every remaining lease
            # without changing any other baseline's frontier behavior.
            for managed in tuple(self.orchestrator.lifecycle.active_leases):
                if managed.request_id != event.request_id:
                    continue
                self.orchestrator.lifecycle.cancel_demand(
                    managed.lease.demand_id
                )
                self.orchestrator.dispatcher.send_cancel(
                    managed,
                    reason="B1 whole-event pipeline completed",
                )

    @staticmethod
    def _terminal_evidence_time(hypothesis: Hypothesis) -> datetime:
        intervals = [
            interval
            for node_state in hypothesis.node_states.values()
            for interval in node_state.event_time_intervals
        ]
        return max(
            (interval.end for interval in intervals),
            default=hypothesis.event_time_window.end,
        )

    @staticmethod
    def _terminal_rearm_interval_ms(state: _RequestState) -> int | None:
        graph = state.runtime.graph.graph
        root = next(node for node in graph.nodes if node.node_id == graph.root_node_id)
        policy = root.annotations.get("post_completion_policy")
        if not isinstance(policy, Mapping) or policy.get("mode") != "scene_clear_rearm":
            return None
        clear_interval_ms = policy.get("clear_interval_ms")
        if not isinstance(clear_interval_ms, int) or clear_interval_ms <= 0:
            LOGGER.warning(
                "invalid scene-clear rearm policy graph=%s policy=%r",
                graph.graph_id,
                policy,
            )
            return None
        return clear_interval_ms

    @staticmethod
    def _terminal_event_key(state: _RequestState, hypothesis: Hypothesis) -> str:
        """Return the semantic occurrence key used for terminal deduplication.

        Hypothesis identity is intentionally more specific than terminal-event
        identity: it includes every role binding.  Set-valued events such as a
        convoy can therefore have several completed follower hypotheses for a
        single leader occurrence.  An authored root may project terminal
        identity onto the roles which actually distinguish event occurrences.
        """

        graph = state.runtime.graph.graph
        root = next(node for node in graph.nodes if node.node_id == graph.root_node_id)
        configured_roles = root.annotations.get("terminal_event_identity_roles")
        if not isinstance(configured_roles, (list, tuple)) or not configured_roles:
            return hypothesis.canonical_key or str(hypothesis.hypothesis_id)

        role_keys: list[str] = []
        for role in configured_roles:
            if not isinstance(role, str) or role not in hypothesis.role_bindings:
                LOGGER.warning(
                    "invalid terminal_event_identity_roles=%r for graph=%s; "
                    "falling back to hypothesis identity",
                    configured_roles,
                    graph.graph_id,
                )
                return hypothesis.canonical_key or str(hypothesis.hypothesis_id)
            binding = hypothesis.role_bindings[role]
            role_keys.append(
                "|".join(
                    (
                        role,
                        binding.canonical_entity_id,
                        binding.established_by_occurrence_id or "",
                    )
                )
            )
        return deterministic_id(
            "terminal_event",
            {
                "request_id": hypothesis.request_id,
                "graph_hash": hypothesis.graph_hash,
                "identity_roles": role_keys,
            },
        )

    def _on_disturbance_request(self, _topic: str, payload: bytes) -> None:
        try:
            request = decode_model(payload, RuntimeDisturbanceRequest)
            previous = self.orchestrator.store.get(
                "runtime_disturbance_acks", str(request.message_id), RuntimeDisturbanceAck
            )
            if previous is None:
                ack = self.apply_disturbance(request)
                self.orchestrator.store.put(
                    "runtime_disturbance_acks", str(request.message_id), ack
                )
            else:
                ack = previous
            self.orchestrator.transport.publish(
                disturbance_ack_topic(request.submitter_id),
                encode_model(ack),
                qos=1,
                retain=False,
            )
        except Exception:
            LOGGER.exception("invalid FABLE runtime disturbance request")

    def _on_event_request(self, _topic: str, payload: bytes) -> None:
        try:
            request = decode_model(payload, EventRequestSubmission)
            previous = self.orchestrator.store.get(
                "event_request_responses", str(request.message_id), EventRequestResponse
            )
            if previous is not None:
                response = previous
            else:
                try:
                    response = self.submit_event(request)
                except Exception as exc:
                    LOGGER.exception("event request failed request_id=%s", request.request_id)
                    response = EventRequestResponse(
                        request_message_id=request.message_id,
                        request_id=request.request_id,
                        accepted=False,
                        reason=str(exc),
                    )
                self.orchestrator.store.put(
                    "event_request_responses", str(request.message_id), response
                )
            self.orchestrator.transport.publish(
                event_response_topic(request.submitter_id),
                encode_model(response),
                qos=1,
                retain=False,
            )
        except Exception:
            LOGGER.exception("invalid FABLE event request")
