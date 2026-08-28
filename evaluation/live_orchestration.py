"""Bridge controlled planning decisions into the shared live dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import threading
import time
from typing import Any, Protocol

from evaluation.baselines.models import BaselineDecision, BaselinePlanningCase
from evaluation.orchestration import ControlledPlanningCoordinator, PlanningTrigger
from evaluation.schemas import BaselineId
from fable.common.ids import deterministic_id
from fable.distributed.models import ProviderRuntimeSpec
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import (
    ExternalInputKind,
    PhysicalAlternative,
    PhysicalAlternativeGraph,
)
from fable.planning.provider_registry import ProviderRegistry
from fable.scheduling.adapters import candidate_from_alternatives
from fable.scheduling.models import PlanCandidate, TaskSchedulingPolicy


LOGGER = logging.getLogger(__name__)

B1_BASELINES = frozenset(
    {BaselineId.B1_STATIC_WHOLE_EVENT, BaselineId.B1_HANDWRITTEN_STATIC}
)


class CandidateDispatcher(Protocol):
    def submit_candidates(
        self,
        candidates: tuple[PlanCandidate, ...],
        *,
        runtime_overrides: dict[str, ProviderRuntimeSpec] | None = None,
        now: datetime | None = None,
        allow_capacity_overcommit: bool = False,
    ) -> tuple[Any, tuple[Any, ...]]: ...


@dataclass(frozen=True)
class LivePlanningResult:
    decision: BaselineDecision
    candidates: tuple[PlanCandidate, ...]
    admission_batches: tuple[Any, ...]
    commands: tuple[Any, ...]


class LivePlanningBridge:
    """Plan at the policy boundary, then use the existing scheduler/dispatcher."""

    def __init__(
        self,
        *,
        coordinator: ControlledPlanningCoordinator,
        provider_registry: ProviderRegistry,
        dispatcher: CandidateDispatcher,
        deployment: DeploymentGraph | None = None,
        fanout_predicate_ids: frozenset[str] = frozenset(),
        fanout_node_ids: frozenset[str] = frozenset(),
        fanout_batch_size: int = 0,
        fanout_batch_interval_seconds: float = 0.0,
        staged_fanout_predicate_ids: frozenset[str] = frozenset(
            {"VEHICLE_PRESENT_BEFORE"}
        ),
    ) -> None:
        self.coordinator = coordinator
        self.provider_registry = provider_registry
        self.dispatcher = dispatcher
        self.deployment = deployment
        self.fanout_predicate_ids = fanout_predicate_ids
        self.fanout_node_ids = fanout_node_ids
        self.fanout_batch_size = max(0, int(fanout_batch_size))
        self.fanout_batch_interval_seconds = max(
            0.0, float(fanout_batch_interval_seconds)
        )
        self.staged_fanout_predicate_ids = staged_fanout_predicate_ids
        self._fanout_generation_by_request: dict[str, int] = {}
        self._late_bound_demand_ids_by_request: dict[str, set[object]] = {}
        self._fanout_lock = threading.Lock()

    def plan_and_dispatch(
        self,
        case: BaselinePlanningCase,
        *,
        trigger: PlanningTrigger,
        task_policy: TaskSchedulingPolicy,
        runtime_overrides: dict[str, ProviderRuntimeSpec] | None = None,
    ) -> LivePlanningResult:
        if task_policy.request_id != case.request_id:
            raise ValueError("task policy and planning case request IDs differ")
        policy = self.coordinator.policy
        direct_b0_late_bound = (
            trigger == PlanningTrigger.SEMANTIC_FRONTIER
            and self.coordinator.baseline_id == BaselineId.B0_PRODUCE_ALL
            and hasattr(policy, "plan_late_bound")
        )
        if direct_b0_late_bound:
            # B0's CE-specific provider union is already frozen at admission.
            # On a semantic fork, instantiate only the newly grounded
            # contract. Re-running the strict whole-event policy first asks it
            # to validate stale structural demands (whose runtime chain alias
            # can differ from the authored registry) and can abort before the
            # bound successor, such as EXITS(vehicle), is ever dispatched.
            already_instantiated = frozenset(
                self._late_bound_demand_ids_by_request.get(case.request_id, set())
            )
            decision = policy.plan_late_bound(  # type: ignore[attr-defined]
                case,
                excluded_demand_ids=already_instantiated,
            )
        else:
            decision = self.coordinator.decide(case, trigger=trigger)
        if (
            trigger == PlanningTrigger.SEMANTIC_FRONTIER
            and decision.baseline_id
            in {
                BaselineId.B0_PRODUCE_ALL,
                BaselineId.B1_STATIC_WHOLE_EVENT,
                BaselineId.B1_HANDWRITTEN_STATIC,
                BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
            }
            and hasattr(policy, "plan_late_bound")
        ):
            already_instantiated = frozenset(
                self._late_bound_demand_ids_by_request.get(case.request_id, set())
            )
            late_bound = (
                decision
                if direct_b0_late_bound
                else policy.plan_late_bound(  # type: ignore[attr-defined]
                    case,
                    excluded_demand_ids=already_instantiated,
                )
            )
            if late_bound.selected_alternative_ids:
                decision = late_bound
                graph_demands = {
                    item.alternative_id: item.demand_id
                    for item in case.frontier_graph.alternatives
                }
                self._late_bound_demand_ids_by_request.setdefault(
                    case.request_id, set()
                ).update(
                    graph_demands[item]
                    for item in decision.selected_alternative_ids
                    if item in graph_demands
                )
        graph = self._decision_graph(case, decision)
        alternatives = self._selected(graph, decision)
        selected_chains_by_demand: dict[object, frozenset[str]] | None = None
        authored_b1_node_ids: frozenset[str] | None = None
        if decision.baseline_id in {
            BaselineId.B0_PRODUCE_ALL,
            BaselineId.B1_STATIC_WHOLE_EVENT,
            BaselineId.B1_HANDWRITTEN_STATIC,
            BaselineId.B2_FRONTIER_FIXED_REALIZATION,
            BaselineId.B2_STATIC_WHOLE_EVENT,
            BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
        }:
            # Coverage expansion is execution scaffolding, not a second
            # planner. Restrict it to the realization selected by the
            # controlled baseline. FABLE deliberately retains its existing
            # unrestricted, frontier-aware coverage behavior.
            mutable_chains: dict[object, set[str]] = {}
            for alternative in alternatives:
                mutable_chains.setdefault(alternative.demand_id, set()).add(
                    alternative.chain_id
                )
            selected_chains_by_demand = {
                demand_id: frozenset(chain_ids)
                for demand_id, chain_ids in mutable_chains.items()
            }
            if decision.baseline_id in B1_BASELINES:
                placement = getattr(
                    getattr(policy, "registry", None),
                    "get_trace_placement",
                    lambda _trace_id: None,
                )(case.trace_id)
                authored_b1_node_ids = (
                    frozenset(placement.allowed_node_ids)
                    if placement is not None
                    else frozenset(
                        step.node_id
                        for alternative in alternatives
                        for step in alternative.step_placements
                    )
                )
        # B1's defining contract is a fixed, manually authored placement.
        # Post-selection coverage expansion is adaptive/broadcast behavior and
        # must never be applied to B1, even when a legacy placement allowlist
        # contains several nodes.  B0 deliberately retains that expansion.
        fanout = (
            ()
            if decision.baseline_id in B1_BASELINES
            else self._fanout_alternatives(
                case,
                graph,
                allowed_chain_ids_by_demand=selected_chains_by_demand,
                allowed_node_ids=authored_b1_node_ids,
            )
        )
        if fanout:
            fanout_demand_ids = {item.demand_id for item in fanout}
            if decision.baseline_id in {
                BaselineId.B0_PRODUCE_ALL,
                BaselineId.B0_ALWAYS_ON,
            }:
                # B0_PRODUCE_ALL is the CE-authored provider set broadcast to
                # all eligible sensors; B0_ALWAYS_ON retains the historical
                # all-realization semantics. Coverage therefore adds node
                # realizations without changing the selected provider chains.
                alternatives = tuple(
                    {
                        item.alternative_id: item
                        for item in (*alternatives, *fanout)
                    }.values()
                )
            else:
                alternatives = tuple(
                    item
                    for item in alternatives
                    if item.demand_id not in fanout_demand_ids
                ) + fanout
            decision = self._decision_with_executed_alternatives(
                decision,
                alternatives,
            )
        if not alternatives:
            return LivePlanningResult(decision, (), (), ())

        if fanout:
            # Each semantic demand's camera coverage is one atomic plan.  Do
            # not submit its per-camera realizations as independent fallback
            # candidates: the scheduler correctly de-duplicates candidates
            # for the same obligation, which previously meant that only the
            # first few cameras actually received commands even though the
            # recorded decision claimed full fan-out.  Separate demands still
            # get separate candidates so one demand cannot suppress another.
            alternatives_by_demand: dict[object, list[PhysicalAlternative]] = {}
            for alternative in alternatives:
                alternatives_by_demand.setdefault(
                    alternative.demand_id, []
                ).append(alternative)
            candidates = tuple(
                candidate_from_alternatives(
                    tuple(group),
                    (self._demand(case, group[0]),),
                    provider_registry=self.provider_registry,
                    task_policy=task_policy,
                    predicted_completion_ms=decision.predicted_completion_ms,
                    allow_replicated_demand=len(group) > 1,
                )
                for _, group in sorted(
                    alternatives_by_demand.items(), key=lambda item: str(item[0])
                )
            )
            submissions = tuple((candidate,) for candidate in candidates)
        elif decision.baseline_id in B1_BASELINES:
            # A scheduling candidate may only contain one checkpoint.  B1's
            # fixed whole-event pipeline can span several semantic
            # checkpoints, so admit each checkpoint slice independently.
            # This executes every authored obligation exactly once; it does
            # not add node/source alternatives or provide failover.
            alternatives_by_checkpoint: dict[object, list[PhysicalAlternative]] = {}
            for alternative in alternatives:
                alternatives_by_checkpoint.setdefault(
                    alternative.checkpoint_id, []
                ).append(alternative)
            candidates = tuple(
                candidate_from_alternatives(
                    tuple(group),
                    tuple(self._demand(case, item) for item in group),
                    provider_registry=self.provider_registry,
                    task_policy=task_policy,
                    predicted_completion_ms=decision.predicted_completion_ms,
                )
                for _, group in sorted(
                    alternatives_by_checkpoint.items(), key=lambda item: str(item[0])
                )
            )
            submissions = tuple((candidate,) for candidate in candidates)
        elif decision.baseline_id in {
            BaselineId.B0_PRODUCE_ALL,
            BaselineId.B0_ALWAYS_ON,
        }:
            # These legacy/all-node controls admit each realization in
            # its own batch so the scheduler's normal fallback de-duplication
            # does not collapse alternatives for the same demand.
            candidates = tuple(
                candidate_from_alternatives(
                    (alternative,),
                    (self._demand(case, alternative),),
                    provider_registry=self.provider_registry,
                    task_policy=task_policy,
                )
                for alternative in alternatives
            )
            submissions = tuple((candidate,) for candidate in candidates)
        else:
            demands = tuple(
                self._demand(case, alternative) for alternative in alternatives
            )
            candidates = (
                candidate_from_alternatives(
                    alternatives,
                    demands,
                    provider_registry=self.provider_registry,
                    task_policy=task_policy,
                    predicted_completion_ms=decision.predicted_completion_ms,
                ),
            )
            submissions = (candidates,)

        immediate_submissions = submissions
        deferred_submissions: tuple[tuple[PlanCandidate, ...], ...] = ()
        staged_fanout = any(
            self._demand(case, alternative).semantic_predicate.predicate_id
            in self.staged_fanout_predicate_ids
            for alternative in fanout
        )
        if (
            staged_fanout
            and self.fanout_batch_size > 0
            and self.fanout_batch_interval_seconds > 0
            and len(submissions) > self.fanout_batch_size
        ):
            immediate_submissions = submissions[: self.fanout_batch_size]
            deferred_submissions = submissions[self.fanout_batch_size :]

        batches: list[Any] = []
        commands: list[Any] = []
        for submission in immediate_submissions:
            batch, emitted = self.dispatcher.submit_candidates(
                submission,
                runtime_overrides=runtime_overrides,
                now=case.now,
                allow_capacity_overcommit=(
                    decision.baseline_id
                    in {
                        BaselineId.B0_PRODUCE_ALL,
                        BaselineId.B1_STATIC_WHOLE_EVENT,
                        BaselineId.B1_HANDWRITTEN_STATIC,
                    }
                ),
            )
            batches.append(batch)
            commands.extend(emitted)
        if decision.baseline_id in B1_BASELINES:
            rejected = []
            for batch in batches:
                for record in getattr(batch, "records", ()):
                    if str(getattr(record, "decision", "")) not in {
                        "ADMITTED",
                        "AdmissionDecision.ADMITTED",
                    }:
                        rejected.append(
                            f"{record.candidate_id}:"
                            f"{getattr(record.decision, 'value', record.decision)}:"
                            f"{record.reason}"
                        )
            if rejected:
                raise RuntimeError(
                    "B1 authored-plan execution conformance failed; selected "
                    "candidate was not admitted: " + "; ".join(rejected)
                )
        if deferred_submissions:
            self._schedule_fanout_escalation(
                request_id=case.request_id,
                submissions=deferred_submissions,
                runtime_overrides=runtime_overrides,
                now=case.now,
            )
        return LivePlanningResult(
            decision=decision,
            candidates=candidates,
            admission_batches=tuple(batches),
            commands=tuple(commands),
        )

    @staticmethod
    def _decision_with_executed_alternatives(
        decision: BaselineDecision,
        alternatives: tuple[PhysicalAlternative, ...],
    ) -> BaselineDecision:
        """Project post-selection coverage expansion into the public decision."""

        nodes = tuple(sorted({
            step.node_id for item in alternatives for step in item.step_placements
        }))
        sources = tuple(sorted({
            external.source_id
            for item in alternatives
            for external in item.external_inputs
            if external.source_id is not None
        }))
        providers = tuple(sorted({
            f"{step.provider_id}@{step.node_id}"
            for item in alternatives
            for step in item.step_placements
        }))
        return decision.model_copy(
            update={
                "selected_alternative_ids": tuple(
                    item.alternative_id for item in alternatives
                ),
                "selected_chain_ids": tuple(sorted({
                    item.chain_id for item in alternatives
                })),
                "selected_node_ids": nodes,
                "selected_source_ids": sources,
                "activated_provider_keys": providers,
                "continuation_types": tuple(sorted({
                    value
                    for item in alternatives
                    for value in item.continuation_output_types
                })),
                "predicted_completion_ms": max(
                    (item.estimated_completion_ms for item in alternatives),
                    default=None,
                ),
                "predicted_transfer_bytes": sum(
                    item.estimated_transfer_bytes for item in alternatives
                ),
                "predicted_compute_ms": sum(
                    sum(step.execution_ms for step in item.step_placements)
                    for item in alternatives
                ),
                "reason": decision.reason + (
                    " Executed coverage expansion is represented as one "
                    "atomically admitted multi-node plan."
                ),
            }
        )

    def _schedule_fanout_escalation(
        self,
        *,
        request_id: str,
        submissions: tuple[tuple[PlanCandidate, ...], ...],
        runtime_overrides: dict[str, ProviderRuntimeSpec] | None,
        now: datetime,
    ) -> None:
        """Dispatch remaining sensor realizations in bounded timed batches.

        The initial batch starts synchronously.  Remaining cameras are always
        reached, but heavy detector activation is staggered so an uncalibrated
        all-camera fallback cannot stampede a shared GPU.  A newer replan
        supersedes an older escalation schedule for the same request.
        """

        with self._fanout_lock:
            generation = self._fanout_generation_by_request.get(request_id, 0) + 1
            self._fanout_generation_by_request[request_id] = generation

        def dispatch_remaining() -> None:
            batch_size = self.fanout_batch_size
            for offset in range(0, len(submissions), batch_size):
                time.sleep(self.fanout_batch_interval_seconds)
                with self._fanout_lock:
                    if self._fanout_generation_by_request.get(request_id) != generation:
                        return
                for submission in submissions[offset : offset + batch_size]:
                    try:
                        self.dispatcher.submit_candidates(
                            submission,
                            runtime_overrides=runtime_overrides,
                            now=now,
                        )
                    except Exception:
                        LOGGER.exception(
                            "deferred fan-out dispatch failed request=%s", request_id
                        )

        thread = threading.Thread(
            target=dispatch_remaining,
            name=f"fable-fanout-{request_id}",
            daemon=True,
        )
        thread.start()

    def _fanout_alternatives(
        self,
        case: BaselinePlanningCase,
        graph: PhysicalAlternativeGraph,
        *,
        allowed_chain_ids_by_demand: dict[object, frozenset[str]] | None = None,
        allowed_node_ids: frozenset[str] | None = None,
    ) -> tuple[PhysicalAlternative, ...]:
        """Choose one local realization per eligible node for fan-out demands.

        This intentionally does not use spatial penalties: campaigns with
        unknown or dynamic sensor placement must search every selected node.
        """

        demand_by_id = {
            item.demand_id: item
            for item in (*case.frontier_demands, *case.all_task_demands)
        }

        def target_nodes(demand) -> frozenset[str]:
            """Return the bounded coverage set for one sensor-local demand.

            A convergence predicate needs a shared camera view, but device
            class is not evidence that a view is useful. In particular, some
            archive intervals begin after the event while a fixed camera has
            complete live evidence. Keep every scenario-selected node here;
            authored source preferences are applied below when they exist.
            """

            # Once a semantic hypothesis has bound a concrete camera
            # reference, its successor is no longer an unscoped discovery
            # demand. Keep execution on that camera; otherwise N camera-local
            # hypotheses each fan out to N cameras, exhaust capacity, and the
            # later valid hypotheses are admitted with no provider commands.
            reference = str(demand.bound_roles.get("reference") or "")
            prefix = "camera_fov:"
            if reference.startswith(prefix):
                node_id = reference[len(prefix) :]
                if (
                    (not self.fanout_node_ids or node_id in self.fanout_node_ids)
                    and (allowed_node_ids is None or node_id in allowed_node_ids)
                ):
                    return frozenset({node_id})
            return (
                self.fanout_node_ids
                if allowed_node_ids is None
                else self.fanout_node_ids & allowed_node_ids
            )
        live_demand_ids = {
            alternative.demand_id
            for alternative in graph.alternatives
            if any(
                item.kind == ExternalInputKind.LIVE_SOURCE
                for item in alternative.external_inputs
            )
        }
        by_demand_and_node: dict[
            tuple[object, str], list[PhysicalAlternative]
        ] = {}
        for alternative in graph.alternatives:
            demand = demand_by_id.get(alternative.demand_id)
            if (
                demand is None
                or demand.semantic_predicate.predicate_id
                not in self.fanout_predicate_ids
            ):
                continue
            if allowed_chain_ids_by_demand is not None and (
                alternative.chain_id
                not in allowed_chain_ids_by_demand.get(
                    alternative.demand_id, frozenset()
                )
            ):
                continue
            if not alternative.step_placements:
                continue
            result_node_id = alternative.step_placements[-1].node_id
            live_source_nodes = {
                str(item.node_id)
                for item in alternative.external_inputs
                if item.kind == ExternalInputKind.LIVE_SOURCE
                and item.node_id is not None
            }
            if alternative.demand_id in live_demand_ids and not live_source_nodes:
                # During live replay, retained artifacts for the same demand
                # must not displace the node-local live realization merely
                # because their estimated startup cost is smaller.
                continue
            # Sensor-local predicate output is published on the source node's
            # node-scoped MQTT topic. A remotely placed final evaluator would
            # subscribe to a different topic and can never consume that
            # evidence, even if the physical graph considers the transfer
            # feasible. Only admit source/evaluator-colocated realizations and
            # key fan-out by the sensing node, not merely by the last step.
            execution_nodes = (
                {result_node_id}
                if not live_source_nodes
                else {
                    node_id
                    for node_id in live_source_nodes
                    if node_id == result_node_id
                }
            )
            for node_id in execution_nodes:
                scoped_nodes = target_nodes(demand)
                if scoped_nodes and node_id not in scoped_nodes:
                    continue
                by_demand_and_node.setdefault(
                    (alternative.demand_id, node_id), []
                ).append(alternative)
        selected = []
        for key in sorted(by_demand_and_node, key=lambda item: (str(item[0]), item[1])):
            selected.append(
                min(
                    by_demand_and_node[key],
                    key=lambda item: (
                        item.estimated_completion_ms,
                        item.estimated_transfer_bytes,
                        -item.minimum_quality_score,
                        item.alternative_id,
                    ),
                )
            )
        covered_nodes = {
            alternative.step_placements[-1].node_id
            for alternative in selected
            if alternative.step_placements
        }
        # The bounded physical graph is intentionally not exhaustive. For a
        # single-live-input, fully local chain, materialize any missing selected
        # sensor nodes from a valid local template. This is safer and cheaper
        # than making request-derived alternative ordering decide which camera
        # receives the demand.
        local_live_templates = [
            item
            for item in selected
            if len(
                [
                    external
                    for external in item.external_inputs
                    if external.kind == ExternalInputKind.LIVE_SOURCE
                ]
            )
            == 1
            and item.step_placements
            and all(
                external.kind
                in {
                    ExternalInputKind.LIVE_SOURCE,
                    ExternalInputKind.OMITTED_OPTIONAL,
                }
                for external in item.external_inputs
            )
        ]
        if (
            self.deployment is not None
            and local_live_templates
            and not any(
                demand_by_id[item.demand_id].source_preferences
                for item in local_live_templates
            )
        ):
            template = min(
                local_live_templates,
                key=lambda item: (
                    item.estimated_completion_ms,
                    item.alternative_id,
                ),
            )
            template_input = next(
                item
                for item in template.external_inputs
                if item.kind == ExternalInputKind.LIVE_SOURCE
            )
            demand = demand_by_id[template.demand_id]
            eligible_sources = set(demand.eligible_source_ids)
            synthesized: list[PhysicalAlternative] = []
            for node_id in sorted(target_nodes(demand)):
                sources = sorted(
                    (
                        source
                        for source in self.deployment.sources.values()
                        if source.available
                        and source.node_id == node_id
                        and template_input.data_type in source.live_data_types
                        and (
                            not eligible_sources
                            or source.source_id in eligible_sources
                        )
                    ),
                    key=lambda source: source.source_id,
                )
                if not sources:
                    continue
                source = sources[0]
                synthesized.append(
                    template.model_copy(
                        update={
                            "alternative_id": deterministic_id(
                                "source_local_fanout_alt",
                                {
                                    "chain": template.chain_id,
                                    "graph_node": demand.graph_node_id,
                                    "node": node_id,
                                },
                                length=32,
                            ),
                            "external_inputs": tuple(
                                item.model_copy(
                                    update={
                                        "node_id": node_id,
                                        "source_id": source.source_id,
                                    }
                                )
                                if item.kind == ExternalInputKind.LIVE_SOURCE
                                else item
                                for item in template.external_inputs
                            ),
                            "step_placements": tuple(
                                step.model_copy(
                                    update={
                                        "node_id": node_id,
                                        "node_class": self.deployment.node(
                                            node_id
                                        ).node_class,
                                        "reused_provider_instance_id": None,
                                    }
                                )
                                for step in template.step_placements
                            ),
                            "transfers": (),
                            "estimated_transfer_bytes": 0,
                            "spatial_preference_penalty": 0,
                            "spatial_preference_reason": (
                                "explicit source-local selected-node fan-out"
                            ),
                        }
                    )
                )
            if synthesized:
                selected = [
                    item
                    for item in selected
                    if item.demand_id != template.demand_id
                ]
                selected.extend(synthesized)
                covered_nodes.update(
                    item.step_placements[-1].node_id
                    for item in synthesized
                )
        # The general alternative builder may spend its per-chain enumeration
        # budget on several realizations of the first nodes. A direct,
        # input-free retrospective matcher is safely relocatable, so complete
        # the selected-node fan-out explicitly instead of inheriting that
        # enumeration-order truncation.
        templates = [
            item
            for item in selected
            if not item.external_inputs
            and not item.transfers
            and item.step_placements
            and all(
                step.provider_id == "historical_vehicle_interval_matcher"
                for step in item.step_placements
            )
        ]
        if templates:
            template = min(
                templates,
                key=lambda item: (
                    item.estimated_completion_ms,
                    item.alternative_id,
                ),
            )
            demand = demand_by_id[template.demand_id]
            for node_id in sorted(target_nodes(demand) - covered_nodes):
                selected.append(
                    template.model_copy(
                        update={
                            "alternative_id": deterministic_id(
                                "fanout_alt",
                                {
                                    "template": template.alternative_id,
                                    "demand": template.demand_id,
                                    "node": node_id,
                                },
                                length=32,
                            ),
                            "step_placements": tuple(
                                step.model_copy(update={"node_id": node_id})
                                for step in template.step_placements
                            ),
                            "spatial_preference_penalty": 0,
                            "spatial_preference_reason": (
                                "topology-free selected-node fan-out"
                            ),
                        }
                    )
                )
        # Preserve authored spatial preferences when available.  For an
        # uncalibrated deployment every penalty is equal and the stable node
        # order supplies deterministic, reproducible escalation.
        spatially_scoped: list[PhysicalAlternative] = []
        for demand_id in sorted(
            {item.demand_id for item in selected}, key=str
        ):
            group = [item for item in selected if item.demand_id == demand_id]
            demand = demand_by_id[demand_id]
            if demand.source_preferences and group:
                best_penalty = min(item.spatial_preference_penalty for item in group)
                group = [
                    item for item in group
                    if item.spatial_preference_penalty == best_penalty
                ]
            spatially_scoped.extend(group)
        return tuple(
            sorted(
                spatially_scoped,
                key=lambda item: (
                    item.spatial_preference_penalty,
                    item.estimated_completion_ms,
                    item.estimated_transfer_bytes,
                    item.step_placements[-1].node_id
                    if item.step_placements
                    else "",
                    item.alternative_id,
                ),
            )
        )

    @staticmethod
    def _decision_graph(
        case: BaselinePlanningCase, decision: BaselineDecision
    ) -> PhysicalAlternativeGraph:
        if decision.planning_scope == "STATIC_LATE_BOUND_INSTANTIATION":
            return case.frontier_graph
        if decision.baseline_id in {
            BaselineId.B0_PRODUCE_ALL,
            BaselineId.B1_STATIC_WHOLE_EVENT,
            BaselineId.B0_ALWAYS_ON,
            BaselineId.B1_HANDWRITTEN_STATIC,
            BaselineId.B2_STATIC_WHOLE_EVENT,
            BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
        }:
            return case.whole_event_graph
        return case.frontier_graph

    @staticmethod
    def _selected(
        graph: PhysicalAlternativeGraph, decision: BaselineDecision
    ) -> tuple[PhysicalAlternative, ...]:
        by_id = {item.alternative_id: item for item in graph.alternatives}
        try:
            return tuple(by_id[item] for item in decision.selected_alternative_ids)
        except KeyError as exc:
            raise ValueError(
                f"policy selected an alternative outside its planning graph: {exc}"
            ) from exc

    @staticmethod
    def _demand(case: BaselinePlanningCase, alternative: PhysicalAlternative):
        demands = {
            item.demand_id: item
            for item in (*case.frontier_demands, *case.all_task_demands)
        }
        try:
            return demands[alternative.demand_id]
        except KeyError as exc:
            raise ValueError(
                "selected alternative has no submitted demand: "
                f"{alternative.demand_id}; submitted="
                f"{sorted(str(item) for item in demands)}"
            ) from exc
