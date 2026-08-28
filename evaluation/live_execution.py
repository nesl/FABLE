"""Authoritative semantic-result progression for live evaluation requests."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from collections import Counter, deque
import logging
import os
from typing import Callable
from uuid import UUID

from evaluation.baselines.models import BaselinePlanningCase
from evaluation.concurrent_admission import joint_batch_case
from evaluation.live_orchestration import LivePlanningBridge, LivePlanningResult
from evaluation.live_records import planning_records
from evaluation.orchestration import PlanningTrigger
from evaluation.observation_buffer import EarlyObservationBuffer
from evaluation.planning_cases import executable_runtime_graph, scope_demands_to_nodes
from evaluation.schemas import BaselineId
from evaluation.schemas.records import EvaluationRecord
from fable.common.enums import HypothesisLifecycle, TruthValue
from fable.common.schemas import (
    BindingDelta,
    NodeHeartbeat,
    PredicateDemand,
    PredicateResult,
    ResultProvenance,
)
from fable.common.time import EventTimeInterval, SourceWatermark, WatermarkSnapshot, utc_now
from fable.distributed.config import ProviderRuntimeResolver
from fable.planning import BoundedLabelPlanner, DemandCompileContext, DemandCompiler
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import ActiveProviderInstance, PhysicalAlternativeGraph
from fable.planning.alternative_graph import PhysicalAlternativeGraphBuilder
from fable.scheduling.control import CheckpointController
from fable.scheduling.models import TaskSchedulingPolicy
from fable.semantic.models import RuntimeTransition
from fable.semantic.runtime import SemanticRuntime
from fable.spatial.models import SpatialFilterMode, SpatialObservation


LOGGER = logging.getLogger(__name__)


def transition_changes_frontier(status: object) -> bool:
    """Whether a semantic application requires successor planning."""

    value = getattr(status, "value", status)
    return str(value) in {"APPLIED", "FORKED"}


@dataclass
class LiveRequestState:
    run_id: str
    trace_id: str
    event_family: str
    runtime: SemanticRuntime
    whole_event_demands: tuple
    whole_event_graph: PhysicalAlternativeGraph
    task_policy: TaskSchedulingPolicy
    placement_id: str = ""
    replay_supported_sensor_ids: tuple[str, ...] = ()
    allowed_execution_node_ids: tuple[str, ...] = ()
    allow_raw_to_trusted_site_edge: bool = False
    coverage_node_id: str | None = None
    active_spatial_deployment_id: str | None = None
    spatial_maximum_observation_groups: int = 1
    source_watermarks: dict[str, SourceWatermark] = field(default_factory=dict)
    completed_source_ids: set[str] = field(default_factory=set)
    resource_epoch: int = 0
    semantic_epoch: int = 0
    early_observations: EarlyObservationBuffer | None = None
    pending_observation_batch: deque[PredicateResult] = field(default_factory=deque)
    baseline_id: BaselineId = BaselineId.FABLE
    evaluation_record_sink: Callable[[EvaluationRecord], None] | None = None
    checkpoint_snapshot_sink: Callable[[BaselinePlanningCase], None] | None = None
    planning_cases: dict[str, BaselinePlanningCase] = field(default_factory=dict)


@dataclass(frozen=True)
class LiveProgression:
    transition: RuntimeTransition
    planning: tuple[LivePlanningResult, ...] = ()
    terminal_lifecycles: dict[str, str] | None = None
    detections: tuple[dict, ...] = ()
    semantic_epoch: int | None = None


class AuthoritativeLiveExecution:
    """Own semantic runtimes and dispatch each newly-derived frontier.

    The distributed transport remains responsible for durable delivery.  This
    object is the result callback installed by the orchestrator service; it is
    deliberately the only component allowed to mutate live hypotheses.
    """

    def __init__(
        self,
        *,
        demand_compiler: DemandCompiler,
        graph_builder: PhysicalAlternativeGraphBuilder,
        deployment: DeploymentGraph,
        bridge: LivePlanningBridge,
        checkpoint_controller: CheckpointController,
        runtime_resolver: ProviderRuntimeResolver | None = None,
    ) -> None:
        self.demand_compiler = demand_compiler
        self.graph_builder = graph_builder
        self.deployment = deployment
        self.bridge = bridge
        self.checkpoints = checkpoint_controller
        self.runtime_resolver = runtime_resolver
        self._requests: dict[str, LiveRequestState] = {}

    def register(self, state: LiveRequestState) -> None:
        request_id = state.runtime.config.request_id
        if request_id in self._requests:
            raise ValueError(f"live request already registered: {request_id}")
        if state.task_policy.request_id != request_id:
            raise ValueError("task policy and semantic runtime request IDs differ")
        self._requests[request_id] = state

    def request_state(self, request_id: str) -> LiveRequestState:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise ValueError(
                f"no authoritative live semantic runtime for {request_id}"
            ) from exc

    def _refresh_process_reuse_view(self) -> PhysicalAlternativeGraphBuilder:
        """Expose only truly demand-agnostic live processes to planning."""

        active = tuple(
            ActiveProviderInstance(
                provider_instance_id=item.provider_instance_id,
                provider_id=item.share_key.provider_id,
                node_id=item.share_key.node_id,
                configuration_hash=item.share_key.configuration_hash,
                source_ids=item.share_key.source_signature,
                output_data_types=item.share_key.output_data_types,
            )
            for item in self.checkpoints.lifecycle.active_instances
            if item.share_key.demand_agnostic_process
        )
        self.graph_builder = PhysicalAlternativeGraphBuilder(
            provider_registry=self.graph_builder.providers,
            artifact_catalog=self.graph_builder.artifacts,
            deployment=self.deployment,
            config=self.graph_builder.config,
            active_providers=active,
        )
        return self.graph_builder

    def update_network_deployment(self, deployment: DeploymentGraph) -> None:
        """Replace planner-visible link costs while preserving provider state."""

        self.deployment = deployment
        self.graph_builder = PhysicalAlternativeGraphBuilder(
            provider_registry=self.graph_builder.providers,
            artifact_catalog=self.graph_builder.artifacts,
            deployment=deployment,
            config=self.graph_builder.config,
            active_providers=self.graph_builder.active_providers,
        )
        self.bridge.deployment = deployment
        # Preserve the coordinator's admission decision so it can validate the
        # resource-epoch transition, but replace the planner used by adaptive
        # policies. Greedy policies have no planner attribute.
        policy = self.bridge.coordinator.policy
        if hasattr(policy, "planner"):
            policy.planner = BoundedLabelPlanner(
                provider_registry=self.graph_builder.providers,
                artifact_catalog=self.graph_builder.artifacts,
                deployment=deployment,
            )

    def _plan_and_record(
        self,
        state: LiveRequestState,
        case: BaselinePlanningCase,
        *,
        trigger: PlanningTrigger,
    ) -> LivePlanningResult:
        if state.checkpoint_snapshot_sink is not None:
            state.checkpoint_snapshot_sink(case)
        result = self.bridge.plan_and_dispatch(
            case,
            trigger=trigger,
            task_policy=state.task_policy,
        )
        if (
            trigger == PlanningTrigger.RESOURCE_EPOCH
            and not result.decision.selected_alternative_ids
            and not result.commands
            and "VALIDATED_LINK_STATE" in case.replan_trigger
            and case.replan_trigger.endswith(":FAIL")
        ):
            # No independent source/placement can satisfy this obligation
            # while the validated sensor link is down. Make the deliberate
            # wait state auditable instead of reporting a mysterious generic
            # infeasibility; RESTORE will issue bounded recovery work.
            case = replace(
                case,
                replan_trigger=(
                    f"{case.replan_trigger}:WAIT_FOR_SOURCE_RECOVERY"
                ),
            )
            result = replace(
                result,
                decision=result.decision.model_copy(
                    update={
                        "reason": (
                            result.decision.reason
                            + " WAIT_FOR_SOURCE_RECOVERY: no independent "
                            "source and executable placement remain while the "
                            "validated sensor link is unavailable."
                        )
                    }
                ),
            )
        if state.evaluation_record_sink is not None:
            for record in planning_records(
                case=case,
                baseline_id=state.baseline_id,
                planning=result,
            ):
                state.evaluation_record_sink(record)
        return result

    def add_seed(self, seed) -> LiveProgression:
        """Fork an active request from an additional typed seed observation."""

        state = self.request_state(seed.request_id)
        transition = state.runtime.seed(seed)
        planning: list[LivePlanningResult] = []
        state.semantic_epoch += 1
        for hypothesis_id in transition.hypothesis_ids:
            hypothesis = state.runtime.get_hypothesis(hypothesis_id)
            frontier = state.runtime.get_frontier(hypothesis_id)
            if frontier is None:
                continue
            allowed_execution_nodes = set(state.allowed_execution_node_ids)
            eligible_sources = tuple(
                sorted(
                    source_id
                    for source_id, source in self.deployment.sources.items()
                    if (
                        not allowed_execution_nodes
                        or source.node_id in allowed_execution_nodes
                    )
                )
            )
            demands = self.demand_compiler.compile_frontier(
                graph=state.runtime.graph,
                hypothesis=hypothesis,
                frontier=frontier,
                context=DemandCompileContext(
                    raw_data_must_remain_local=(
                        not state.allow_raw_to_trusted_site_edge
                    ),
                    eligible_source_ids_by_node={
                        graph_node_id: eligible_sources
                        for graph_node_id in frontier.snapshot.enabled_node_ids
                    }
                ),
            )
            demands = scope_demands_to_nodes(
                demands,
                state.allowed_execution_node_ids,
            )
            known = {item.demand_id for item in state.whole_event_demands}
            state.whole_event_demands = (
                *state.whole_event_demands,
                *(item for item in demands if item.demand_id not in known),
            )
            frontier_graph = self.graph_builder.build(
                demands,
                now=seed.observed_at,
            )
            state.whole_event_graph = self.graph_builder.build(
                state.whole_event_demands,
                now=seed.observed_at,
            )
            if self.runtime_resolver is not None:
                frontier_graph = executable_runtime_graph(
                    frontier_graph,
                    runtime_resolver=self.runtime_resolver,
                    allow_reference_runtimes=False,
                    allowed_node_ids=state.allowed_execution_node_ids,
                )
                state.whole_event_graph = executable_runtime_graph(
                    state.whole_event_graph,
                    runtime_resolver=self.runtime_resolver,
                    allow_reference_runtimes=False,
                    allowed_node_ids=state.allowed_execution_node_ids,
                )
            case = BaselinePlanningCase(
                run_id=state.run_id,
                trace_id=state.trace_id,
                request_id=seed.request_id,
                event_family=state.event_family,
                placement_id=state.placement_id,
                frontier_demands=demands,
                all_task_demands=state.whole_event_demands,
                frontier_graph=frontier_graph,
                whole_event_graph=state.whole_event_graph,
                now=seed.observed_at,
                replay_supported_sensor_ids=state.replay_supported_sensor_ids,
                resource_epoch=state.resource_epoch,
                semantic_epoch=state.semantic_epoch,
                replan_trigger="SEMANTIC_FRONTIER",
            )
            state.planning_cases[str(hypothesis_id)] = case
            planning.append(
                self._plan_and_record(
                    state,
                    case,
                    trigger=PlanningTrigger.SEMANTIC_FRONTIER,
                )
            )
        return LiveProgression(
            transition=transition,
            planning=tuple(planning),
            semantic_epoch=state.semantic_epoch,
        )

    def evict_unprogressed_seed(
        self,
        request_id: str,
        hypothesis_id: UUID,
        *,
        seed_event_time,
        minimum_progress_gap_ms: int,
        force: bool = False,
    ) -> bool:
        """Remove one unchanged seed hypothesis and its demand leases."""

        state = self.request_state(request_id)
        if not state.runtime.invalidate_unprogressed_hypothesis(
            hypothesis_id,
            seed_event_time=seed_event_time,
            minimum_progress_gap_ms=minimum_progress_gap_ms,
            force=force,
        ):
            return False
        removed = tuple(
            demand
            for demand in state.whole_event_demands
            if demand.hypothesis_id == hypothesis_id
        )
        removed_ids = {item.demand_id for item in removed}
        leases = tuple(
            managed
            for managed in self.checkpoints.lifecycle.active_leases
            if managed.lease.demand_id in removed_ids
        )
        if leases:
            self.bridge.dispatcher.cancel_leases(
                leases,
                reason="rolling seed pool evicted unchanged hypothesis",
            )
        for demand_id in removed_ids:
            self.checkpoints.lifecycle.complete_demand(demand_id, now=utc_now())
        state.whole_event_demands = tuple(
            demand
            for demand in state.whole_event_demands
            if demand.hypothesis_id != hypothesis_id
        )
        state.planning_cases.pop(str(hypothesis_id), None)
        state.whole_event_graph = self.graph_builder.build(
            state.whole_event_demands,
            now=utc_now(),
        )
        return True

    def handle_result(self, result: PredicateResult) -> LiveProgression:
        state = self.request_state(result.request_id)
        if state.early_observations is None:
            first = self._apply_result(result)
            accumulated_planning = list(first.planning)
            accumulated_detections = list(first.detections)
            last = first
            # Dynamic-frontier policies deliberately do not buffer arbitrary
            # early sensor observations, but deterministic semantic closure is
            # independent of that policy. Drain SAME_ENTITY(x, x) after every
            # frontier-changing result so FABLE cannot strand a grounded
            # identity checkpoint merely because it has no early buffer.
            if not first.terminal_lifecycles:
                for _ in range(64):
                    derived = self._pop_active_observation(
                        state,
                        now=result.processing_completed_at,
                    )
                    if derived is None:
                        break
                    last = self._apply_result(derived)
                    accumulated_planning.extend(last.planning)
                    accumulated_detections.extend(last.detections)
                    if last.terminal_lifecycles:
                        break
            return LiveProgression(
                transition=last.transition,
                planning=tuple(accumulated_planning),
                terminal_lifecycles=(
                    last.terminal_lifecycles or first.terminal_lifecycles
                ),
                detections=tuple(accumulated_detections),
                semantic_epoch=state.semantic_epoch,
            )

        state.early_observations.add(
            result,
            now=result.processing_completed_at,
        )
        accumulated_planning: list[LivePlanningResult] = []
        accumulated_detections: list[dict] = []
        last: LiveProgression | None = None
        meaningful: LiveProgression | None = None
        # One early observation can unlock another already-buffered graph
        # stage. Drain to a fixed point, but retain a hard safety bound.
        for _ in range(64):
            matched = self._pop_active_observation(state, now=result.processing_completed_at)
            if matched is None:
                break
            last = self._apply_result(matched)
            accumulated_planning.extend(last.planning)
            accumulated_detections.extend(last.detections)
            derived_identity = (
                matched.provenance.provider_id == "identity_reflexivity"
            )
            if derived_identity:
                LOGGER.info(
                    "derived identity request=%s hypothesis=%s status=%s reason=%s "
                    "event_interval=%s bindings=%s",
                    matched.request_id,
                    matched.hypothesis_id,
                    last.transition.status,
                    last.transition.reason,
                    matched.event_time_interval,
                    matched.binding_delta.validated,
                )
            if (
                transition_changes_frontier(last.transition.status)
                or last.terminal_lifecycles
                or last.detections
                # Do not hide a failed orchestrator-derived transition behind
                # the external observation that activated it. This diagnostic
                # is essential because no provider MQTT result exists for a
                # reflexive identity fact.
                or (
                    derived_identity
                    and str(getattr(last.transition.status, "value", last.transition.status))
                    not in {"APPLIED", "FORKED", "MERGED", "NOOP"}
                )
            ):
                meaningful = last
            if last.terminal_lifecycles:
                break
        if last is None:
            rejection_counts: dict[str, int] = {}
            for hypothesis in state.runtime.active_hypotheses:
                frontier = state.runtime.get_frontier(hypothesis.hypothesis_id)
                if frontier is None:
                    continue
                demands = self.demand_compiler.compile_frontier(
                    graph=state.runtime.graph,
                    hypothesis=hypothesis,
                    frontier=frontier,
                    context=DemandCompileContext(
                        raw_data_must_remain_local=(
                            not state.allow_raw_to_trusted_site_edge
                        ),
                        eligible_source_ids_by_node={
                            node_id: tuple(
                                sorted(
                                    source_id
                                    for source_id, source in self.deployment.sources.items()
                                    if not state.allowed_execution_node_ids
                                    or source.node_id in set(state.allowed_execution_node_ids)
                                )
                            )
                            for node_id in frontier.snapshot.enabled_node_ids
                        },
                    ),
                )
                for demand in scope_demands_to_nodes(
                    demands, state.allowed_execution_node_ids
                ):
                    for reason, count in state.early_observations.rejection_counts(
                        demand,
                        source_aliases=self._source_aliases(),
                    ).items():
                        rejection_counts[reason] = rejection_counts.get(reason, 0) + count
            diagnostic = ", ".join(
                f"{reason}={count}" for reason, count in sorted(rejection_counts.items())
            ) or "no-active-demand"
            return LiveProgression(
                transition=RuntimeTransition(
                    status="NOOP",
                    result_id=result.result_id,
                    reason=(
                        "observation retained until its semantic graph node "
                        "and grounded bindings become active; mismatch_counts: "
                        f"{diagnostic}"
                    ),
                ),
                semantic_epoch=state.semantic_epoch,
            )
        reported = meaningful or last
        return LiveProgression(
            transition=reported.transition,
            planning=tuple(accumulated_planning),
            terminal_lifecycles=(
                last.terminal_lifecycles or reported.terminal_lifecycles
            ),
            detections=tuple(accumulated_detections),
            semantic_epoch=state.semantic_epoch,
        )

    def _pop_active_observation(
        self,
        state: LiveRequestState,
        *,
        now: datetime,
    ) -> PredicateResult | None:
        if (
            state.pending_observation_batch
            and state.baseline_id != BaselineId.B0_PRODUCE_ALL
        ):
            return state.pending_observation_batch.popleft()

        active_demands: list[PredicateDemand] = []
        # A physical observation can match more than one rolling hypothesis.
        # Consume it for the most-progressed hypothesis first.  UUID ordering is
        # not semantic ordering and previously allowed a newly seeded
        # hypothesis waiting for visit 2 to steal visit 3 from an older chain.
        # That failure only appeared during multi-camera fan-out because the
        # extra seeds changed UUID/order pressure; single-camera replay passed.
        active_hypotheses = sorted(
            state.runtime.active_hypotheses,
            key=self._hypothesis_observation_priority,
            reverse=True,
        )
        for hypothesis in active_hypotheses:
            frontier = state.runtime.get_frontier(hypothesis.hypothesis_id)
            if frontier is None:
                continue
            allowed_nodes = set(state.allowed_execution_node_ids)
            eligible_sources = tuple(
                sorted(
                    source_id
                    for source_id, source in self.deployment.sources.items()
                    if not allowed_nodes or source.node_id in allowed_nodes
                )
            )
            demands = self.demand_compiler.compile_frontier(
                graph=state.runtime.graph,
                hypothesis=hypothesis,
                frontier=frontier,
                context=DemandCompileContext(
                    raw_data_must_remain_local=(
                        not state.allow_raw_to_trusted_site_edge
                    ),
                    eligible_source_ids_by_node={
                        node_id: eligible_sources
                        for node_id in frontier.snapshot.enabled_node_ids
                    }
                ),
            )
            demands = scope_demands_to_nodes(
                demands,
                state.allowed_execution_node_ids,
            )
            active_demands.extend(demands)

        # Derived deterministic facts must be drained globally before sensor
        # observations. With rolling seed hypotheses, an older PASSES frontier
        # can otherwise keep matching/re-buffering duplicate observations and
        # starve a later hypothesis already waiting at SAME_ENTITY(x, x).
        for demand in active_demands:
            reflexive = self._reflexive_identity_result(demand, now=now)
            if reflexive is not None:
                return reflexive

        # B0 deliberately broadcasts the CE-specific provider union and can
        # queue hundreds of simultaneously valid sensor projections.  Those
        # physical observations must not starve a deterministic identity
        # checkpoint which is already active. Other policies retain their
        # existing FIFO behavior.
        if state.pending_observation_batch:
            return state.pending_observation_batch.popleft()

        if state.early_observations is None:
            return None

        for demand in active_demands:
                matches = state.early_observations.match_for_demand(
                    demand,
                    now=now,
                    source_aliases=self._source_aliases(),
                )
                if matches:
                    # Preserve the exact envelope that may create a binding
                    # fork. Recompiling this demand later produces a distinct
                    # checkpoint UUID, which cannot address the runtime's
                    # retained fork context and makes every late sibling stale.
                    if all(
                        item.demand_id != demand.demand_id
                        for item in state.whole_event_demands
                    ):
                        state.whole_event_demands = (
                            *state.whole_event_demands,
                            demand,
                        )
                    state.pending_observation_batch.extend(matches[1:])
                    return matches[0]

        # A late-binding predicate can produce candidates over several sensor
        # callbacks rather than in one MQTT batch.  Once the first candidate
        # forks its parent, that predicate is no longer in any child frontier,
        # but the semantic runtime deliberately retains the original fork
        # envelope so later candidates can become sibling hypotheses.  Keep
        # projecting observations through only those retired demands whose
        # parent is actually FORKED; arbitrary stale demands remain closed.
        for demand in reversed(state.whole_event_demands):
            try:
                parent = state.runtime.get_hypothesis(demand.hypothesis_id)
            except KeyError:
                continue
            if parent.lifecycle.value != "FORKED":
                continue
            matches = state.early_observations.match_for_demand(
                demand,
                now=now,
                source_aliases=self._source_aliases(),
            )
            if matches:
                state.pending_observation_batch.extend(matches[1:])
                return matches[0]
        return None

    @staticmethod
    def _hypothesis_observation_priority(hypothesis) -> tuple[int, int, float]:
        """Prefer established evidence chains over newer rolling seeds."""

        satisfied = sum(
            state.status.value == "SATISFIED"
            for state in hypothesis.node_states.values()
        )
        occurrences = sum(
            len(state.occurrence_ids)
            for state in hypothesis.node_states.values()
            if state.status.value == "SATISFIED"
        )
        # Earlier creation wins the final tie when two chains have equal graph
        # progress. ``reverse=True`` is used by the caller.
        return satisfied, occurrences, -hypothesis.created_at.timestamp()

    @staticmethod
    def _reflexive_identity_result(
        demand: PredicateDemand,
        *,
        now: datetime,
    ) -> PredicateResult | None:
        """Resolve ``SAME_ENTITY(x, x)`` without deploying an identity model.

        Scoped local track identifiers are opaque values.  Equal values are
        nevertheless the same entity by reflexivity; invoking ReID for this
        case is both unnecessary and unsafe because an unavailable provider
        chain can turn a logically certain result into a false negative.
        Distinct identifiers still follow the normal descriptor/ReID/VLM
        path, so this does not lower any cross-camera association threshold.
        """

        if demand.semantic_predicate.predicate_id != "SAME_ENTITY":
            return None
        left = demand.bound_roles.get("left")
        right = demand.bound_roles.get("right")
        if not left or left != right:
            return None
        # BindingDelta is part of the semantic-runtime contract: its keys are
        # graph variables, not provider-facing predicate role names.  SAME_ENTITY
        # commonly maps ``left``/``right`` onto variables such as ``vehicle`` and
        # ``visit_vehicle_2``.  Returning the predicate labels here causes the
        # runtime to reject a logically certain reflexive result as referencing
        # unknown variables, after which orchestration needlessly deploys ReID.
        role_variables = {
            role.role_name: role.variable for role in demand.semantic_predicate.roles
        }
        validated = {
            role_variables[role_name]: entity_id
            for role_name, entity_id in (("left", left), ("right", right))
            if role_name in role_variables
        }
        if not validated:
            return None
        observed_at = utc_now()
        # The demand interval is the scheduler's *eligible execution window*,
        # not the duration of this derived identity fact.  Carrying its full
        # horizon into semantic state makes SAME_ENTITY appear active until
        # the request deadline and incorrectly blocks the next visit's
        # minimum-delay guard. Reflexivity is an instantaneous state fact at
        # the opening of the enabled identity checkpoint.
        identity_event_time = demand.event_time_interval.model_copy(
            update={"end": demand.event_time_interval.start}
        )
        return PredicateResult(
            occurrence_id=f"reflexive-identity:{demand.demand_id}",
            demand_id=demand.demand_id,
            request_id=demand.request_id,
            graph_hash=demand.graph_hash,
            hypothesis_id=demand.hypothesis_id,
            expected_hypothesis_version=demand.hypothesis_version,
            frontier_id=demand.frontier_id,
            checkpoint_id=demand.checkpoint_id,
            graph_node_id=demand.graph_node_id,
            semantic_predicate=demand.semantic_predicate,
            truth=TruthValue.TRUE,
            confidence=1.0,
            event_time_interval=identity_event_time,
            binding_delta=BindingDelta(validated=validated),
            provenance=ResultProvenance(
                provider_id="identity_reflexivity",
                provider_contract_version=1,
                node_id="orchestrator",
                # PredicateResult provenance is required to name a source so
                # the binding manager can canonicalize validation deltas. This
                # is a derived logical fact rather than a sensor observation;
                # give it an explicit orchestrator-owned source instead of an
                # empty tuple (which the semantic runtime rejects).
                source_ids=("orchestrator:identity_reflexivity",),
            ),
            processing_started_at=observed_at,
            processing_completed_at=observed_at,
        )

    def _source_aliases(self) -> dict[str, tuple[str, ...]]:
        """Map runtime-scoped source names to deployment source IDs.

        Provider artifacts retain concrete node namespaces such as
        ``dvpg_gq_orin_13:camera`` while requests use stable deployment IDs
        such as ``orin13_camera``. Keep this translation at the orchestration
        trust boundary and permit only aliases represented by the configured
        deployment graph.
        """

        aliases: dict[str, set[str]] = {}
        for source_id, source in self.deployment.sources.items():
            runtime_aliases = {source.node_id}
            if "vision" in source.modalities:
                runtime_aliases.add(f"{source.node_id}:camera")
            if "audio" in source.modalities:
                runtime_aliases.update(
                    {
                        f"{source.node_id}:audio",
                        f"{source.node_id}:microphone",
                    }
                )
            for alias in runtime_aliases:
                aliases.setdefault(alias, set()).add(source_id)
        return {
            alias: tuple(sorted(source_ids))
            for alias, source_ids in aliases.items()
        }

    def _apply_result(self, result: PredicateResult) -> LiveProgression:
        try:
            state = self._requests[result.request_id]
        except KeyError as exc:
            raise ValueError(
                f"no authoritative live semantic runtime for {result.request_id}"
            ) from exc

        try:
            lifecycle_before = state.runtime.get_hypothesis(
                result.hypothesis_id
            ).lifecycle.value
        except KeyError:
            lifecycle_before = "MISSING"
        LOGGER.info(
            "predicate_result_boundary boundary=SEMANTIC_APPLY_START "
            "result_id=%s occurrence_id=%s predicate_id=%s request_id=%s "
            "hypothesis_id=%s expected_version=%s lifecycle_before=%s "
            "demand_id=%s frontier_id=%s checkpoint_id=%s graph_node_id=%s "
            "event_time_start=%s event_time_end=%s introduced=%s validated=%s",
            result.result_id,
            result.occurrence_id,
            result.semantic_predicate.predicate_id,
            result.request_id,
            result.hypothesis_id,
            result.expected_hypothesis_version,
            lifecycle_before,
            result.demand_id,
            result.frontier_id,
            result.checkpoint_id,
            result.graph_node_id,
            result.event_time_interval.start.isoformat(),
            result.event_time_interval.end.isoformat(),
            dict(result.binding_delta.introduced),
            dict(result.binding_delta.validated),
        )
        transition = state.runtime.apply(result)
        lifecycle_after = {}
        for hypothesis_id in transition.hypothesis_ids:
            try:
                lifecycle_after[str(hypothesis_id)] = (
                    state.runtime.get_hypothesis(hypothesis_id).lifecycle.value
                )
            except KeyError:
                lifecycle_after[str(hypothesis_id)] = "MISSING"
        LOGGER.info(
            "predicate_result_boundary boundary=SEMANTIC_APPLY_FINISH "
            "result_id=%s status=%s reason=%s semantic_epoch_before=%s "
            "result_hypothesis_lifecycle_before=%s output_hypotheses=%s "
            "output_lifecycles=%s",
            result.result_id,
            transition.status.value,
            transition.reason,
            state.semantic_epoch,
            lifecycle_before,
            tuple(str(item) for item in transition.hypothesis_ids),
            lifecycle_after,
        )
        accepted_transition = transition.status.value in {
            "APPLIED",
            "FORKED",
            "MERGED",
            "NOOP",
        }
        # MERGED is an evidence-only outcome: the runtime resolved a
        # provider-local observation to an already-existing canonical
        # hypothesis without changing its semantic frontier. Replanning that
        # same frontier creates a fresh demand/lease set for every progressive
        # interval sample. In live rendezvous replay this amplified one
        # proximity interval into thousands of callbacks and starved the
        # subsequent bound EXITS result. NOOP likewise has no new frontier.
        frontier_changed = transition_changes_frontier(transition.status)
        # B0 starts the CE-specific provider union before every role is
        # grounded.  Several all-node results can therefore fork the same
        # parent faster than their successor contracts are instantiated.  A
        # later duplicate is then correctly MERGED into an existing child,
        # but treating MERGED as unconditionally evidence-only can strand
        # that child with no physical plan.  Recover only B0 children which
        # have never received a planning case; ordinary duplicate merges and
        # every other policy remain no-ops.
        unplanned_merged_hypothesis_ids = (
            tuple(
                hypothesis_id
                for hypothesis_id in transition.hypothesis_ids
                if str(hypothesis_id) not in state.planning_cases
                and state.runtime.get_frontier(hypothesis_id) is not None
            )
            if (
                state.baseline_id == BaselineId.B0_PRODUCE_ALL
                and str(getattr(transition.status, "value", transition.status))
                == "MERGED"
            )
            else ()
        )
        # Capture predecessor leases so the dispatcher can stop node-local
        # processes after lifecycle reconciliation. The completed demand must
        # leave the capacity ledger before successor planning: otherwise a
        # frontier transition is evaluated against resources belonging to a
        # demand that has already semantically completed.
        superseded_leases = (
            tuple(
                managed
                for managed in self.checkpoints.lifecycle.active_leases
                if managed.lease.demand_id == result.demand_id
            )
            if accepted_transition
            else ()
        )
        planning: list[LivePlanningResult] = []
        next_demands = []
        if frontier_changed:
            state.semantic_epoch += 1
        hypothesis_ids = (
            transition.hypothesis_ids
            if frontier_changed
            else unplanned_merged_hypothesis_ids
        )
        # A duplicate result has already advanced (or been merged into) its
        # authoritative hypothesis. Replanning its current frontier creates a
        # fresh demand/lease set for every MQTT redelivery and can amplify a
        # handful of event-time observations into thousands of executions.
        # Restart reconciliation restores durable plans separately, so a live
        # duplicate must be an idempotent no-op here.
        for hypothesis_id in hypothesis_ids:
            hypothesis = state.runtime.get_hypothesis(hypothesis_id)
            frontier = state.runtime.get_frontier(hypothesis_id)
            demands = ()
            if frontier is not None:
                allowed_execution_nodes = set(state.allowed_execution_node_ids)
                all_source_ids = tuple(
                    sorted(
                        source_id
                        for source_id, source in self.deployment.sources.items()
                        if (
                            not allowed_execution_nodes
                            or source.node_id in allowed_execution_nodes
                        )
                    )
                )
                coverage_source_ids = tuple(
                    sorted(
                        source_id
                        for source_id, source in self.deployment.sources.items()
                        if source.node_id == state.coverage_node_id
                    )
                )
                eligible_source_ids_by_node = {}
                for node_id in frontier.snapshot.enabled_node_ids:
                    node = state.runtime.graph.nodes_by_id[node_id]
                    same_scene_required = (
                        node.predicate is not None
                        and node.predicate.predicate_id == "MOVING"
                        and bool(coverage_source_ids)
                    )
                    eligible_source_ids_by_node[node_id] = (
                        coverage_source_ids if same_scene_required else all_source_ids
                    )
                spatial_observation_by_node = {}
                if (
                    state.active_spatial_deployment_id
                    and self.demand_compiler.spatial_model is not None
                    and result.provenance.source_ids
                ):
                    source_id = result.provenance.source_ids[0]
                    sensor_id = (
                        self.demand_compiler.spatial_bindings.sensor_for_source(
                            source_id,
                            state.active_spatial_deployment_id,
                        )
                        or source_id
                    )
                    spatial_observation_by_node = {
                        node_id: SpatialObservation(
                            current_sensor_id=sensor_id,
                            active_deployment_id=(
                                state.active_spatial_deployment_id
                            ),
                            maximum_observation_groups=(
                                state.spatial_maximum_observation_groups
                            ),
                            filter_mode=SpatialFilterMode.PREFER,
                        )
                        for node_id in frontier.snapshot.enabled_node_ids
                    }
                demands = self.demand_compiler.compile_frontier(
                    graph=state.runtime.graph,
                    hypothesis=hypothesis,
                    frontier=frontier,
                context=DemandCompileContext(
                    raw_data_must_remain_local=(
                        not state.allow_raw_to_trusted_site_edge
                    ),
                        eligible_source_ids_by_node=eligible_source_ids_by_node,
                        spatial_observation_by_node=spatial_observation_by_node,
                    ),
                )
                demands = scope_demands_to_nodes(
                    demands,
                    state.allowed_execution_node_ids,
                )
                next_demands.extend(demands)
                known = {item.demand_id for item in state.whole_event_demands}
                state.whole_event_demands = (
                    *state.whole_event_demands,
                    *(item for item in demands if item.demand_id not in known),
                )
                observed_at = result.processing_completed_at
                self.checkpoints.handle_predicate_result(
                    result=result,
                    transition=transition,
                    request_id=result.request_id,
                    hypothesis_id=hypothesis_id,
                    next_demands=demands,
                    continuation_artifact_ids=result.artifact_ids,
                    hypothesis_lifecycle=hypothesis.lifecycle,
                    now=observed_at,
                    immediate_completed_demand=True,
                )
                graph_builder = self._refresh_process_reuse_view()
                frontier_graph = graph_builder.build(demands, now=observed_at)
                if not frontier_graph.alternatives:
                    LOGGER.warning(
                        "frontier graph has no alternatives request=%s predicates=%s "
                        "active_processes=%s pruned=%s",
                        result.request_id,
                        tuple(
                            demand.semantic_predicate.predicate_id
                            for demand in demands
                        ),
                        tuple(
                            (item.provider_id, item.node_id)
                            for item in graph_builder.active_providers
                        ),
                        {
                            "counts": dict(
                                Counter(item.code for item in frontier_graph.pruned)
                            ),
                            "samples": tuple(
                                (item.chain_id, item.code, item.reason)
                                for item in frontier_graph.pruned[:8]
                            ),
                        },
                    )
                state.whole_event_graph = graph_builder.build(
                    state.whole_event_demands,
                    now=observed_at,
                )
                if self.runtime_resolver is not None:
                    frontier_graph = executable_runtime_graph(
                        frontier_graph,
                        runtime_resolver=self.runtime_resolver,
                        allow_reference_runtimes=False,
                        allowed_node_ids=state.allowed_execution_node_ids,
                    )
                    state.whole_event_graph = executable_runtime_graph(
                        state.whole_event_graph,
                        runtime_resolver=self.runtime_resolver,
                        allow_reference_runtimes=False,
                        allowed_node_ids=state.allowed_execution_node_ids,
                    )
                case = BaselinePlanningCase(
                    run_id=state.run_id,
                    trace_id=state.trace_id,
                    request_id=result.request_id,
                    event_family=state.event_family,
                    placement_id=state.placement_id,
                    frontier_demands=demands,
                    all_task_demands=state.whole_event_demands,
                    frontier_graph=frontier_graph,
                    whole_event_graph=state.whole_event_graph,
                    now=observed_at,
                    replay_supported_sensor_ids=state.replay_supported_sensor_ids,
                    resource_epoch=state.resource_epoch,
                    semantic_epoch=state.semantic_epoch,
                    replan_trigger="SEMANTIC_FRONTIER",
                )
                state.planning_cases[str(hypothesis_id)] = case
                planning.append(
                    self._plan_and_record(
                        state,
                        case,
                        trigger=PlanningTrigger.SEMANTIC_FRONTIER,
                    )
                )
            else:
                self.checkpoints.handle_predicate_result(
                    result=result,
                    transition=transition,
                    request_id=result.request_id,
                    hypothesis_id=hypothesis_id,
                    next_demands=demands,
                    continuation_artifact_ids=result.artifact_ids,
                    hypothesis_lifecycle=hypothesis.lifecycle,
                    now=result.processing_completed_at,
                    immediate_completed_demand=True,
                )

        if accepted_transition and not hypothesis_ids:
            self.checkpoints.lifecycle.complete_demand(
                result.demand_id,
                now=result.processing_completed_at,
            )
        if superseded_leases:
            # Successor admission has now attached any compatible process
            # leases. Removing predecessor leases therefore stops only truly
            # idle providers on the node agent.
            self.bridge.dispatcher.cancel_leases(
                superseded_leases,
                reason=(
                    "semantic frontier advanced after transactional successor "
                    f"handoff at version {result.expected_hypothesis_version}"
                ),
            )

        terminal_lifecycles = None
        detections: list[dict] = []
        if transition.hypothesis_ids:
            transitioned = tuple(
                state.runtime.get_hypothesis(item)
                for item in transition.hypothesis_ids
            )
            completed = any(
                item.lifecycle == HypothesisLifecycle.COMPLETED
                for item in transitioned
            )
            exhausted = (
                all(
                    item.lifecycle != HypothesisLifecycle.ACTIVE
                    for item in transitioned
                )
                and not state.runtime.active_hypotheses
            )
            if completed or exhausted:
                terminal_lifecycles = {
                    str(item): state.runtime.get_hypothesis(item).lifecycle.value
                    for item in transition.hypothesis_ids
                }
                for item in transition.hypothesis_ids:
                    hypothesis = state.runtime.get_hypothesis(item)
                    if hypothesis.lifecycle != HypothesisLifecycle.COMPLETED:
                        continue
                    intervals = tuple(
                        interval
                        for node_state in hypothesis.node_states.values()
                        for interval in node_state.event_time_intervals
                    )
                    if not intervals:
                        continue
                    detections.append(
                        {
                            "hypothesis_id": str(item),
                            "event_family": state.event_family,
                            "event_start_time": min(
                                interval.start for interval in intervals
                            ),
                            "event_end_time": max(
                                interval.end for interval in intervals
                            ),
                            "emitted_at": result.processing_completed_at,
                            "bindings": {
                                role: binding.canonical_entity_id
                                for role, binding in hypothesis.role_bindings.items()
                            },
                        }
                    )
                self.bridge.coordinator.forget(result.request_id)
                self._requests.pop(result.request_id, None)
        return LiveProgression(
            transition=transition,
            planning=tuple(planning),
            terminal_lifecycles=terminal_lifecycles,
            detections=tuple(detections),
            semantic_epoch=state.semantic_epoch,
        )

    def prepare_resource_epoch_cases(
        self,
        request_id: str,
        *,
        observed_at: datetime,
        reason: str,
        recovery_intervals: dict[str, EventTimeInterval] | None = None,
        recovery_demand_ids: set[str] | None = None,
    ) -> tuple[BaselinePlanningCase, ...]:
        """Build immutable adaptive scopes after a validated resource change."""

        state = self.request_state(request_id)
        if state.baseline_id not in {
            BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
            BaselineId.B4_GREEDY_FRONTIER,
            BaselineId.FABLE,
            BaselineId.FABLE_NO_SHARING,
        }:
            return ()
        state.resource_epoch += 1

        def recovery_scoped(
            demands: tuple[PredicateDemand, ...],
        ) -> tuple[PredicateDemand, ...]:
            if not recovery_intervals:
                return demands
            recovered: list[PredicateDemand] = []
            for demand in demands:
                if (
                    recovery_demand_ids is not None
                    and str(demand.demand_id) not in recovery_demand_ids
                ):
                    continue
                matching = tuple(
                    source_id
                    for source_id in demand.eligible_source_ids
                    if source_id in recovery_intervals
                    and source_id.endswith("_camera")
                )
                if not matching:
                    continue
                start = max(
                    demand.event_time_interval.start,
                    min(recovery_intervals[item].start for item in matching),
                )
                end = min(
                    demand.event_time_interval.end,
                    max(recovery_intervals[item].end for item in matching),
                )
                if end <= start:
                    continue
                context = dict(demand.retrospective_context or {})
                recovery_request_id = (
                    f"{request_id}:outage-recovery:{state.resource_epoch}:"
                    f"{matching[0]}:{start.isoformat()}:{end.isoformat()}"
                )
                context.update(
                    {
                        "activation_lookback_ms": max(
                            1, int((end - start).total_seconds() * 1000)
                        ),
                        "outage_recovery": True,
                        "recovery_request_id": recovery_request_id,
                        "active_replay_stream": True,
                        "source_local_catchup": True,
                        "catch_up_and_follow": True,
                        "outage_gap_start": start.isoformat(),
                        "outage_gap_end": end.isoformat(),
                        # Durable typed evidence is consumed continuously by
                        # the semantic runtime. There is not yet a compact
                        # evidence-replay adapter for this source, so a
                        # remaining exact coverage gap advances to bounded raw
                        # replay. Recording the stage makes that fallback
                        # auditable and leaves room for a compact adapter.
                        "recovery_policy_stage": "RAW_FALLBACK",
                        "durable_typed_evidence_checked": True,
                        "compact_evidence_replay_available": False,
                    }
                )
                recovered.append(
                    PredicateDemand.model_validate(
                        {
                            **demand.model_dump(mode="python"),
                            "eligible_source_ids": matching,
                            "event_time_interval": EventTimeInterval(
                                start=start, end=end
                            ),
                            "retrospective_context": context,
                            # This key is derived from the semantic interval,
                            # sources, and bindings. Recovery changes the first
                            # two, so force schema validation to derive a new
                            # key instead of carrying the live demand's key.
                            "sharing_key": None,
                        }
                    )
                )
            return tuple(recovered)

        updated_cases: list[BaselinePlanningCase] = []
        for hypothesis in state.runtime.active_hypotheses:
            key = str(hypothesis.hypothesis_id)
            case = state.planning_cases.get(key)
            if case is None:
                continue
            frontier_demands = recovery_scoped(case.frontier_demands)
            all_task_demands = recovery_scoped(case.all_task_demands)
            if recovery_intervals and not frontier_demands:
                continue
            if recovery_intervals and not all_task_demands:
                all_task_demands = frontier_demands
            frontier_graph = self.graph_builder.build(
                frontier_demands,
                now=observed_at,
            )
            whole_event_graph = self.graph_builder.build(
                all_task_demands,
                now=observed_at,
            )
            if self.runtime_resolver is not None:
                frontier_graph = executable_runtime_graph(
                    frontier_graph,
                    runtime_resolver=self.runtime_resolver,
                    allow_reference_runtimes=False,
                    allowed_node_ids=state.allowed_execution_node_ids,
                )
                whole_event_graph = executable_runtime_graph(
                    whole_event_graph,
                    runtime_resolver=self.runtime_resolver,
                    allow_reference_runtimes=False,
                    allowed_node_ids=state.allowed_execution_node_ids,
                )
            updated = replace(
                case,
                now=observed_at,
                frontier_graph=frontier_graph,
                whole_event_graph=whole_event_graph,
                frontier_demands=frontier_demands,
                all_task_demands=all_task_demands,
                resource_epoch=state.resource_epoch,
                replan_trigger=f"RESOURCE_EPOCH:{reason}",
            )
            # Recovery is a temporary execution scope, not a new semantic
            # frontier. Persisting its source/window-restricted demands as the
            # canonical case discarded already accumulated observations (most
            # visibly the first/second visits of three-visit stalking).
            if not recovery_intervals:
                state.whole_event_graph = whole_event_graph
                state.planning_cases[key] = updated
            updated_cases.append(updated)

        return tuple(updated_cases)

    def dispatch_prepared_resource_epoch(
        self,
        request_id: str,
        cases: tuple[BaselinePlanningCase, ...],
        *,
        reason: str,
        allow_joint_hypotheses: bool = True,
    ) -> tuple[LivePlanningResult, ...]:
        """Dispatch already-snapshotted resource scopes for one request."""

        state = self.request_state(request_id)
        planning: list[LivePlanningResult] = []
        # E2's joint-planning mechanism is meaningful only if simultaneous
        # live hypotheses reach the policy in one immutable checkpoint batch.
        # Keep this experiment-gated until the physical campaign validates
        # result routing and provider lifecycle behavior. Other policies retain
        # their authored independent/sequential semantics.
        joint_live_replanning = (
            state.baseline_id == BaselineId.FABLE
            and os.environ.get("FABLE_JOINT_RESOURCE_EPOCH_PLANNING", "0") == "1"
            and allow_joint_hypotheses
            and len(cases) > 1
        )
        if joint_live_replanning:
            joint = joint_batch_case(
                cases,
                run_id=state.run_id,
                request_id=request_id,
            )
            joint = replace(
                joint,
                resource_epoch=state.resource_epoch,
                replan_trigger=f"RESOURCE_EPOCH:{reason}:JOINT_HYPOTHESES",
            )
            planning.append(
                self._plan_and_record(
                    state, joint, trigger=PlanningTrigger.RESOURCE_EPOCH
                )
            )
        else:
            for updated in cases:
                planning.append(
                    self._plan_and_record(
                        state,
                        updated,
                        trigger=PlanningTrigger.RESOURCE_EPOCH,
                    )
                )
        return tuple(planning)

    def handle_resource_epoch(
        self,
        request_id: str,
        *,
        observed_at: datetime,
        reason: str,
        recovery_intervals: dict[str, EventTimeInterval] | None = None,
        recovery_demand_ids: set[str] | None = None,
    ) -> tuple[LivePlanningResult, ...]:
        """Replan active adaptive scopes after a validated resource change."""

        cases = self.prepare_resource_epoch_cases(
            request_id,
            observed_at=observed_at,
            reason=reason,
            recovery_intervals=recovery_intervals,
            recovery_demand_ids=recovery_demand_ids,
        )
        return self.dispatch_prepared_resource_epoch(
            request_id,
            cases,
            reason=reason,
        )

    def handle_heartbeat(
        self,
        heartbeat: NodeHeartbeat,
    ) -> tuple[LiveProgression, ...]:
        """Advance coverage-aware absence windows from live source progress."""
        progressions: list[LiveProgression] = []
        for request_id, state in tuple(self._requests.items()):
            observed_sources = {
                source_id: SourceWatermark(
                    source_id=source_id,
                    event_time=source.latest_event_time,
                    observed_at=heartbeat.sent_at,
                    sequence=source.latest_sequence,
                    operational_coverage=source.operational_coverage,
                )
                for source_id, source in heartbeat.sources.items()
            }
            # Authored graphs use ``camera_mobile`` as a logical coverage role.
            # Bind it only to the camera on the node that supplied the seed;
            # this prevents an unrelated camera from proving scene absence.
            if state.coverage_node_id == heartbeat.node_id:
                configured_camera_ids = {
                    source_id
                    for source_id, source in self.deployment.sources.items()
                    if source.node_id == state.coverage_node_id
                    and "vision" in source.modalities
                }
                camera_sources = [
                    source
                    for source_id, source in observed_sources.items()
                    if source_id in configured_camera_ids
                ]
                if camera_sources:
                    camera = max(camera_sources, key=lambda item: item.event_time)
                    observed_sources["camera_mobile"] = camera.model_copy(
                        update={"source_id": "camera_mobile"}
                    )
                    physical_camera = heartbeat.sources.get(camera.source_id)
                    if physical_camera is not None and physical_camera.replay_complete:
                        state.completed_source_ids.add("camera_mobile")
            for source_id, source in heartbeat.sources.items():
                if source.replay_complete:
                    state.completed_source_ids.add(source_id)
            for source_id, observed in observed_sources.items():
                prior = state.source_watermarks.get(source_id)
                if prior is None or (
                    observed.event_time,
                    observed.sequence or 0,
                ) >= (
                    prior.event_time,
                    prior.sequence or 0,
                ):
                    state.source_watermarks[source_id] = observed
            if not state.source_watermarks:
                continue
            snapshot = WatermarkSnapshot(
                generated_at=heartbeat.sent_at,
                sources=dict(state.source_watermarks),
            )
            transitions = state.runtime.close_temporal_windows(
                snapshot
            )
            transitions = (
                *transitions,
                *state.runtime.expire_temporal_windows(
                    snapshot,
                    completed_source_ids=state.completed_source_ids,
                ),
            )
            for transition in transitions:
                terminal_lifecycles: dict[str, str] | None = None
                detections: list[dict] = []
                transitioned = tuple(
                    state.runtime.get_hypothesis(item)
                    for item in transition.hypothesis_ids
                )
                completed = any(
                    item.lifecycle == HypothesisLifecycle.COMPLETED
                    for item in transitioned
                )
                exhausted = (
                    bool(transitioned)
                    and all(
                        item.lifecycle != HypothesisLifecycle.ACTIVE
                        for item in transitioned
                    )
                    and not state.runtime.active_hypotheses
                )
                if completed or exhausted:
                    terminal_lifecycles = {
                        str(item): state.runtime.get_hypothesis(item).lifecycle.value
                        for item in transition.hypothesis_ids
                    }
                    for item in transition.hypothesis_ids:
                        hypothesis = state.runtime.get_hypothesis(item)
                        if hypothesis.lifecycle != HypothesisLifecycle.COMPLETED:
                            continue
                        intervals = tuple(
                            interval
                            for node_state in hypothesis.node_states.values()
                            for interval in node_state.event_time_intervals
                        )
                        if not intervals:
                            continue
                        detections.append(
                            {
                                "hypothesis_id": str(item),
                                "event_family": state.event_family,
                                "event_start_time": min(
                                    interval.start for interval in intervals
                                ),
                                "event_end_time": max(
                                    interval.end for interval in intervals
                                ),
                                "emitted_at": heartbeat.sent_at,
                                "bindings": {
                                    role: binding.canonical_entity_id
                                    for role, binding in hypothesis.role_bindings.items()
                                },
                            }
                        )
                    self.bridge.coordinator.forget(request_id)
                    self._requests.pop(request_id, None)
                progressions.append(
                    LiveProgression(
                        transition=transition,
                        terminal_lifecycles=terminal_lifecycles,
                        detections=tuple(detections),
                    )
                )
        return tuple(progressions)

    def has_request(self, request_id: str) -> bool:
        return request_id in self._requests

    def cancel(self, request_id: str) -> bool:
        state = self._requests.pop(request_id, None)
        if state is None:
            return False
        self.bridge.coordinator.forget(request_id)
        return True
