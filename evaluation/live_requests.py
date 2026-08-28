"""Typed admission of authored complex-event requests into live execution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Callable, ClassVar, Literal

from pydantic import Field, model_validator
from pydantic import UUID7
from uuid import UUID

from evaluation.baselines.factory import build_baseline_policy
from evaluation.baselines.models import BaselinePlanningCase
from evaluation.baselines.static_registry import (
    StaticPipelineRegistry,
    static_pipeline_registry_path,
)
from evaluation.live_execution import AuthoritativeLiveExecution, LiveRequestState
from evaluation.live_orchestration import LivePlanningBridge, LivePlanningResult
from evaluation.live_records import planning_records
from evaluation.concurrent_admission import joint_batch_case
from evaluation.e2_snapshots import export_checkpoint_snapshot
from evaluation.schemas.records import EvaluationRecord
from evaluation.orchestration import ControlledPlanningCoordinator, PlanningTrigger
from evaluation.observation_buffer import EarlyObservationBuffer
from evaluation.planning_cases import executable_runtime_graph, scope_demands_to_nodes
from evaluation.task_universe import TaskDemandUniverseBuilder
from evaluation.provider_coverage import validate_live_provider_coverage
from evaluation.schemas import BaselineId
from fable.common.base import FableModel, JSONValue
from fable.common.ids import uuid7
from fable.common.enums import TruthValue
from fable.common.schemas import (
    BindingDelta,
    Hypothesis,
    PredicateDemand,
    PredicateResult,
    ResultProvenance,
)
from fable.common.time import DeadlineSpec, EventTimeInterval, utc_now
from fable.distributed.config import ProviderRuntimeResolver
from fable.planning import (
    BoundedLabelPlanner,
    DemandCompileContext,
    DemandCompiler,
    PhysicalAlternativeGraphBuilder,
)
from fable.planning.deployment import DeploymentGraph
from fable.planning.provider_registry import ProviderRegistry
from fable.scheduling.control import CheckpointController
from fable.scheduling.lifecycle import ProviderLifecycleManager
from fable.scheduling.models import TaskSchedulingPolicy


SENSOR_LOCAL_FANOUT_PREDICATES = frozenset(
    {
        "AUDIO_EVENT",
        "VEHICLE_PRESENT_BEFORE",
        "DEPARTURE_OR_ESCAPE",
        "PERSON_PROXIMITY",
        "PERSON_PRESENT",
        "CONVERSATION",
        "TRANSFER",
        "FOLLOWS",
        # A completed traversal is a source-local observation. Watching only
        # the cheapest camera makes the first camera to be planned—not the
        # camera that sees the arrival—determine every downstream binding.
        "PASSES",
        # Image-space convergence is sensor-local.  In deployments without a
        # calibrated spatial prior, selecting one tied camera can silently
        # miss the shared view; the live bridge bounds this fan-out to the
        # scenario-selected observation cameras.
        "DISTANCE_LT",
    }
)
B1_BASELINES = frozenset(
    {BaselineId.B1_STATIC_WHOLE_EVENT, BaselineId.B1_HANDWRITTEN_STATIC}
)
AUTHORED_ALWAYS_ON_BASELINES = frozenset(
    {BaselineId.B0_PRODUCE_ALL, *B1_BASELINES}
)


def _node_id_aliases(node_id: str) -> frozenset[str]:
    """Return the canonical and replay-facing spellings of a node id.

    Recorded provider envelopes historically used ``orin11`` while the
    deployment graph uses ``dvpg_gq_orin_11``.  Treating those as different
    nodes silently empties seed watches before semantic admission.
    """

    aliases = {node_id}
    short = node_id.removeprefix("dvpg_gq_")
    aliases.add(short)
    if short.startswith("orin_"):
        aliases.add("orin" + short.removeprefix("orin_"))
    return frozenset(aliases)


def _source_for_camera_reference(
    reference: str, deployment: DeploymentGraph
) -> tuple[str, ...]:
    """Resolve ``camera_fov:<node>`` to concrete deployment source ids."""

    prefix = "camera_fov:"
    if not reference.startswith(prefix):
        return ()
    referenced_node = reference[len(prefix) :]
    return tuple(
        sorted(
            source_id
            for source_id, source in deployment.sources.items()
            if "vision" in source.modalities
            and referenced_node in _node_id_aliases(source.node_id)
        )
    )


def _partition_multi_seed_acquisition_demand(
    demand: PredicateDemand,
    *,
    eligible_source_ids: tuple[str, ...],
    deployment: DeploymentGraph,
) -> tuple[PredicateDemand, ...]:
    """Give every seed camera an independently completable watch demand.

    A scheduling demand is completed when one provider result satisfies it.
    Reusing one demand id for a fan-out watch therefore cancels every camera
    lease after the first camera reports a seed.  Multi-hypothesis discovery
    needs one demand per concrete source so completion on one camera cannot
    silence the remaining cameras.
    """

    partitioned: list[PredicateDemand] = []
    for source_id in eligible_source_ids:
        source = deployment.sources.get(source_id)
        if source is None:
            continue
        # Partitioning is a source-coverage boundary, not necessarily an
        # execution-placement boundary.  Pin execution to the source only
        # when the demand's raw-locality policy requires it.  Otherwise the
        # selected source's raw stream may be transferred to an authorized
        # trusted edge runtime (and can migrate there after device capacity
        # disappears).
        constraints = demand.hard_constraints
        if demand.hard_constraints.raw_data_must_remain_local:
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
                "hard_constraints": constraints,
                # The source partition is already the sharing boundary.
                "sharing_key": None,
            }
        )
        partitioned.append(PredicateDemand.model_validate(payload))
    return tuple(partitioned)


def _authored_fanout_nodes(
    request: "LiveComplexEventRequest", candidate_node_ids: set[str] | frozenset[str]
) -> frozenset[str]:
    """Limit B1 coverage expansion to its backward-sliced static placement."""

    candidates = frozenset(candidate_node_ids)
    if request.baseline_id not in B1_BASELINES:
        return candidates
    placement = StaticPipelineRegistry.load(static_pipeline_registry_path()).get_placement(
        request.baseline_placement_id, trace_id=request.trace_id
    )
    if placement is None:
        return candidates
    selected = candidates & frozenset(placement.allowed_node_ids)
    return selected or candidates


def _authored_execution_sources(
    request: "LiveComplexEventRequest", candidate_source_ids: tuple[str, ...]
) -> tuple[str, ...]:
    """Project B1 demand compilation onto its frozen authored sources.

    Restricting only execution nodes is insufficient: a physical alternative
    can execute on an authored node while consuming a live source from another
    camera.  The B1 policy then correctly rejects every such alternative and a
    valid seed appears to expire.  B1 must compile from the same fixed source
    slice as its calibrated placement; other baselines retain their existing
    source universe.
    """

    if request.baseline_id not in B1_BASELINES:
        return candidate_source_ids
    placement = StaticPipelineRegistry.load(static_pipeline_registry_path()).get_placement(
        request.baseline_placement_id, trace_id=request.trace_id
    )
    if placement is None or not placement.allowed_source_ids:
        return candidate_source_ids
    allowed = set(placement.allowed_source_ids)
    selected = tuple(
        source_id for source_id in candidate_source_ids if source_id in allowed
    )
    return selected or candidate_source_ids


def _authored_execution_nodes(
    request: "LiveComplexEventRequest", candidate_node_ids: tuple[str, ...]
) -> tuple[str, ...]:
    """Return B1's exact calibrated node slice for physical compilation."""

    if request.baseline_id not in B1_BASELINES:
        return candidate_node_ids
    placement = StaticPipelineRegistry.load(static_pipeline_registry_path()).get_placement(
        request.baseline_placement_id, trace_id=request.trace_id
    )
    if placement is None or not placement.allowed_node_ids:
        return candidate_node_ids
    allowed = set(placement.allowed_node_ids)
    selected = tuple(node_id for node_id in candidate_node_ids if node_id in allowed)
    return selected or candidate_node_ids
from fable.semantic import EventRequestCompiler, SemanticRuntime, SemanticRuntimeConfig
from fable.semantic.models import SeedPredicateResult


class ObservedSeed(FableModel):
    """Provider-observed seed facts; executable graph fields are server-derived."""

    graph_node_key: str = Field(min_length=1)
    occurrence_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    event_time_interval: EventTimeInterval
    introduced_bindings: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0, le=1)


class LiveComplexEventRequest(FableModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.live_complex_event_request.v1"
    schema_version: Literal["fable.live_complex_event_request.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    request_id: str = Field(min_length=1)
    submitter_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    replay_id: str | None = None
    family_id: str = Field(min_length=1)
    # Stable CE-variant key used only by authored static baselines. It lets one
    # backward-sliced exemplar placement be reused by every trace of the same
    # scenario type without conflating variants which share a semantic family.
    baseline_placement_id: str = ""
    parameters: dict[str, JSONValue] = Field(default_factory=dict)
    baseline_id: BaselineId = BaselineId.FABLE
    seed: ObservedSeed | None = None
    seed_graph_node_key: str | None = None
    allowed_seed_source_ids: tuple[str, ...] = ()
    allowed_seed_node_ids: tuple[str, ...] = ()
    # Optional semantic seed filter, distinct from acquisition coverage. B0
    # uses this to execute providers on every sensor while assembling events
    # from the same authored seed references as B1.
    semantic_seed_source_ids: tuple[str, ...] = ()
    semantic_seed_node_ids: tuple[str, ...] = ()
    allowed_execution_node_ids: tuple[str, ...] = ()
    # Explicit experiment-scoped authorization for sending raw sensor input to
    # the trusted site edge. It remains false for normal accuracy runs and
    # never authorizes raw transfer to cloud nodes.
    allow_raw_to_trusted_site_edge: bool = False
    reject_seed_before_registration: bool = False
    seed_timeout_seconds: int = Field(default=120, ge=1, le=3600)
    max_seed_hypotheses: int = Field(default=1, ge=1, le=32)
    seed_admission_strategy: Literal[
        "first_distinct",
        "reference_diverse",
        "reference_bounded",
    ] = "first_distinct"
    allowed_seed_event_time_interval: EventTimeInterval | None = None
    hypothesis_horizon_ms: int = Field(default=300_000, ge=1)
    deadline_offset_ms: int = Field(default=300_000, ge=1)

    @model_validator(mode="after")
    def _seed_or_watch(self):
        if self.seed is None and not self.seed_graph_node_key:
            raise ValueError("request requires seed or seed_graph_node_key")
        if self.seed is not None and self.seed_graph_node_key is not None:
            raise ValueError("request cannot contain both seed and seed_graph_node_key")
        if self.allow_raw_to_trusted_site_edge:
            non_site_nodes = {
                node_id
                for node_id in self.allowed_execution_node_ids
                if not (
                    node_id == "x86server"
                    or node_id.startswith("dvpg_gq_orin_")
                    or node_id.startswith("mobile_archive_")
                )
            }
            if non_site_nodes:
                raise ValueError(
                    "raw transfer authorization is limited to sensors and x86server"
                )
        return self


class LiveComplexEventResponse(FableModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.live_complex_event_response.v1"
    schema_version: Literal["fable.live_complex_event_response.v1"] = SCHEMA_VERSION
    request_message_id: UUID7
    request_id: str
    accepted: bool
    status: Literal["WATCHING", "ADMITTED", "REJECTED"] = "REJECTED"
    hypothesis_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    command_count: int = 0
    seed_occurrence_id: str | None = None
    seed_action: Literal[
        "WATCH_REGISTERED",
        "INITIAL_ADMISSION",
        "ADDITIONAL_HYPOTHESIS",
        "DUPLICATE_IGNORED",
        "REJECTED",
    ] | None = None
    active_seed_hypothesis_count: int = 0
    reason: str = ""


class LiveComplexEventDetection(FableModel):
    """A completed live hypothesis suitable for ground-truth matching."""

    hypothesis_id: str = Field(min_length=1)
    event_family: str = Field(min_length=1)
    event_start_time: datetime
    event_end_time: datetime
    emitted_at: datetime
    bindings: dict[str, str] = Field(default_factory=dict)


class LiveComplexEventProgress(FableModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.live_complex_event_progress.v1"
    schema_version: Literal["fable.live_complex_event_progress.v1"] = SCHEMA_VERSION
    request_id: str
    transition_status: str
    transition_reason: str = ""
    result_id: str | None = None
    hypothesis_ids: tuple[str, ...] = ()
    dispatched_frontiers: int = 0
    terminal: bool = False
    terminal_lifecycles: dict[str, str] = Field(default_factory=dict)
    detections: tuple[LiveComplexEventDetection, ...] = ()
    semantic_epoch: int | None = Field(default=None, ge=0)


class LiveComplexEventCancelRequest(FableModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.live_complex_event_cancel_request.v1"
    schema_version: Literal["fable.live_complex_event_cancel_request.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    request_id: str = Field(min_length=1)
    submitter_id: str = Field(min_length=1)
    reason: str = Field(default="client requested cancellation", min_length=1)


class LiveComplexEventCancelResponse(FableModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.live_complex_event_cancel_response.v1"
    schema_version: Literal["fable.live_complex_event_cancel_response.v1"] = SCHEMA_VERSION
    request_message_id: UUID7
    request_id: str
    status: Literal["CANCELLED", "NOT_FOUND"]
    cancelled_pending_watch: bool = False
    cancelled_active_execution: bool = False
    cancelled_demand_count: int = 0
    released_lease_count: int = 0
    cancel_command_count: int = 0
    reason: str = ""


@dataclass(frozen=True)
class LiveRequestAdmission:
    response: LiveComplexEventResponse
    planning: LivePlanningResult | None = None


@dataclass
class _SeedAcquisitionPlan:
    request: LiveComplexEventRequest
    demands: tuple[PredicateDemand, ...]
    case: BaselinePlanningCase
    bridge: LivePlanningBridge


class LiveRequestManager:
    """Compile authored requests, validate observations, and dispatch frontier one."""

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry,
        deployment: DeploymentGraph,
        runtime_resolver: ProviderRuntimeResolver,
        graph_builder: PhysicalAlternativeGraphBuilder,
        demand_compiler: DemandCompiler,
        planner: BoundedLabelPlanner,
        dispatcher,
        lifecycle: ProviderLifecycleManager,
        checkpoint_controller: CheckpointController,
        evaluation_record_sink: Callable[[EvaluationRecord], None] | None = None,
        checkpoint_snapshot_dir: str | Path | None = None,
        fanout_batch_size: int = 0,
        fanout_batch_interval_seconds: float = 0.0,
    ) -> None:
        self.providers = provider_registry
        self.deployment = deployment
        self.runtimes = runtime_resolver
        self.graph_builder = graph_builder
        self.demand_compiler = demand_compiler
        self.planner = planner
        self.dispatcher = dispatcher
        self.lifecycle = lifecycle
        self.checkpoints = checkpoint_controller
        self.evaluation_record_sink = evaluation_record_sink
        self.checkpoint_snapshot_dir = (
            Path(checkpoint_snapshot_dir) if checkpoint_snapshot_dir else None
        )
        self.fanout_batch_size = fanout_batch_size
        self.fanout_batch_interval_seconds = fanout_batch_interval_seconds
        self._executions: dict[str, AuthoritativeLiveExecution] = {}
        # Demand IDs are versioned as semantic/resource epochs advance. Keep
        # the request lineage for IDs observed on active leases so recovery of
        # a second node is not invalidated by the first node's replan.
        self._demand_request_history: dict[str, str] = {}
        # Recovery notifications can arrive through both the validated
        # resource-epoch path and the later node-session heartbeat path. MQTT
        # redelivery can repeat either notification.  Keep exact recovery
        # scopes idempotent without collapsing distinct hypotheses/demands.
        self._completed_recovery_scopes: dict[str, set[tuple[object, ...]]] = {}
        self._seed_acquisitions: dict[str, _SeedAcquisitionPlan] = {}

    def _capture_checkpoint_snapshot(self, case: BaselinePlanningCase) -> None:
        if self.checkpoint_snapshot_dir is None or not case.frontier_demands:
            return
        checkpoint_id = case.frontier_demands[0].checkpoint_id
        target = (
            self.checkpoint_snapshot_dir
            / case.run_id
            / f"s{case.semantic_epoch:04d}-r{case.resource_epoch:04d}-{checkpoint_id}.json"
        )
        export_checkpoint_snapshot(
            case,
            target,
            capture_kind="typed_runtime_export",
            deployment_artifacts=self.graph_builder.artifacts.artifacts,
        )

    def acquire_seed(
        self,
        request: LiveComplexEventRequest,
        *,
        recovery_node_id: str | None = None,
    ) -> LivePlanningResult:
        """Plan and lease the authored seed predicate before it is observed.

        Seed watches must not depend on ambient always-on analytics.  This
        provisional hypothesis exists only to compile the initial predicate
        demand; the authoritative semantic hypothesis is still created from
        the first validated provider observation in :meth:`admit`.
        """

        assert request.seed_graph_node_key is not None
        compiled = EventRequestCompiler().compile(
            {"family_id": request.family_id, "parameters": request.parameters}
        )
        runtime = SemanticRuntime(
            compiled.graph,
            config=SemanticRuntimeConfig(request_id=request.request_id),
        )
        node = runtime.graph.nodes_by_key[request.seed_graph_node_key]
        now = utc_now()
        seed_event_window = (
            request.allowed_seed_event_time_interval
            or EventTimeInterval(
                start=now,
                end=now + timedelta(seconds=request.seed_timeout_seconds),
            )
        )
        provisional = Hypothesis(
            request_id=request.request_id,
            graph_id=runtime.graph.graph.graph_id,
            graph_hash=runtime.graph.graph.graph_hash,
            graph_version=runtime.graph.graph.graph_version,
            anchor_occurrence_id=f"seed-watch:{request.request_id}",
            event_time_window=seed_event_window,
            deadline=DeadlineSpec(
                latest_useful_completion=now
                + timedelta(seconds=request.seed_timeout_seconds)
            ),
            created_at=now,
            updated_at=now,
        )
        runtime.frontier_deriver.initialize_node_states(provisional)
        frontier = runtime.frontier_deriver.derive(provisional)
        if frontier is None or node.node_id not in frontier.snapshot.enabled_node_ids:
            raise ValueError("authored seed node is not initially executable")
        eligible_sources = tuple(
            sorted(
                source_id
                for source_id, source in self.deployment.sources.items()
                if (
                    not request.allowed_seed_source_ids
                    or source_id in request.allowed_seed_source_ids
                )
                and (
                    not request.allowed_seed_node_ids
                    or bool(
                        _node_id_aliases(source.node_id)
                        & set(request.allowed_seed_node_ids)
                    )
                )
                and (recovery_node_id is None or source.node_id == recovery_node_id)
            )
        )
        eligible_sources = _authored_execution_sources(request, eligible_sources)
        demand = self.demand_compiler.compile_node(
            graph=runtime.graph,
            hypothesis=provisional,
            frontier=frontier,
            graph_node_id=node.node_id,
            context=DemandCompileContext(
                raw_data_must_remain_local=(
                    not request.allow_raw_to_trusted_site_edge
                ),
                eligible_source_ids_by_node={node.node_id: eligible_sources},
            ),
            # Seed bindings do not exist yet.  A stable structural placeholder
            # lets the provider chain be leased; the pending-seed gateway, not
            # this provisional demand, validates and introduces the concrete
            # identity from the provider observation.
            structural_universe=True,
        )
        if request.allowed_seed_event_time_interval is not None:
            # This is a live subscription to an actively replayed stream whose
            # event clock is the recording clock. It is not a request to read
            # old raw media, even though its timestamps precede wall time.
            demand = demand.model_copy(
                update={
                    "retrospective_context": {
                        **(demand.retrospective_context or {}),
                        "active_replay_stream": True,
                    }
                }
            )
        if recovery_node_id is not None:
            recovery_request_id = (
                f"{request.request_id}:seed-outage-recovery:{recovery_node_id}:"
                f"{demand.event_time_interval.start.isoformat()}:"
                f"{demand.event_time_interval.end.isoformat()}"
            )
            demand = PredicateDemand.model_validate(
                {
                    **demand.model_dump(mode="python"),
                    "retrospective_context": {
                        **(demand.retrospective_context or {}),
                        "outage_recovery": True,
                        "recovery_request_id": recovery_request_id,
                        "active_replay_stream": True,
                        "source_local_catchup": True,
                        "catch_up_and_follow": True,
                        "activation_lookback_ms": max(
                            1,
                            int(
                                (
                                    demand.event_time_interval.end
                                    - demand.event_time_interval.start
                                ).total_seconds()
                                * 1000
                            ),
                        ),
                        "outage_gap_start": demand.event_time_interval.start.isoformat(),
                        "outage_gap_end": demand.event_time_interval.end.isoformat(),
                        "recovery_policy_stage": "RAW_FALLBACK",
                        "durable_typed_evidence_checked": True,
                        "compact_evidence_replay_available": False,
                    },
                    "sharing_key": None,
                }
            )
        if request.baseline_id in AUTHORED_ALWAYS_ON_BASELINES:
            # B0/B1 are authored always-on whole-event pipelines. Materialize
            # every structurally executable demand before the seed arrives so
            # its fixed detector/evaluator placement processes the trace from
            # the beginning. Identity comparison itself remains late-bound:
            # there is no meaningful pair to compare until roles are bound.
            demands = TaskDemandUniverseBuilder(self.demand_compiler).build(
                graph=runtime.graph,
                hypothesis=provisional,
                context=DemandCompileContext(
                    raw_data_must_remain_local=(
                        not request.allow_raw_to_trusted_site_edge
                    ),
                    eligible_source_ids_by_node={
                        graph_node_id: eligible_sources
                        for graph_node_id in runtime.graph.executable_predicate_nodes()
                    },
                ),
                # Before the seed, alternate branches may require roles which
                # do not exist yet (for example THREAT_EVENT.person). They are
                # late-bound rather than grounds to reject an otherwise
                # executable authored pipeline.
                skip_uncompilable=True,
            )
            demands = tuple(
                item
                for item in demands
                if not (
                    item.semantic_predicate.predicate_id == "SAME_ENTITY"
                    and any(
                        str(value).startswith("__structural_unbound__:")
                        for value in item.bound_roles.values()
                    )
                )
            )
        else:
            demands = (
                _partition_multi_seed_acquisition_demand(
                    demand,
                    eligible_source_ids=eligible_sources,
                    deployment=self.deployment,
                )
                if request.max_seed_hypotheses > 1
                else (demand,)
            )
        demands = scope_demands_to_nodes(demands, request.allowed_execution_node_ids)
        graph = self.graph_builder.build(demands, now=now)
        # Raw offload expands the placement product substantially.  A bounded
        # graph can then spend its per-chain budget on several placements for
        # the first live source before it reaches the remaining cameras.  Seed
        # acquisition is a coverage obligation, however: placement diversity
        # must not erase source diversity before any evidence has localized
        # the event.  Add one genuinely source-local realization per eligible
        # source.  With raw-local nominal planning these alternatives already
        # exist and the deterministic-id merge is therefore a no-op.
        if (
            request.allow_raw_to_trusted_site_edge
            and node.predicate.predicate_id in SENSOR_LOCAL_FANOUT_PREDICATES
            and request.baseline_id not in AUTHORED_ALWAYS_ON_BASELINES
        ):
            coverage_alternatives = []
            coverage_nodes = []
            coverage_edges = []
            coverage_pruned = []
            for source_id in eligible_sources:
                source = self.deployment.sources.get(source_id)
                if source is None:
                    continue
                local_constraints = demand.hard_constraints.model_copy(
                    update={
                        "raw_data_must_remain_local": True,
                        "allowed_node_ids": (source.node_id,),
                    }
                )
                local_demand = demand.model_copy(
                    update={
                        "eligible_source_ids": (source_id,),
                        "hard_constraints": local_constraints,
                    }
                )
                local_graph = self.graph_builder.build((local_demand,), now=now)
                local_graph = executable_runtime_graph(
                    local_graph,
                    runtime_resolver=self.runtimes,
                    allow_reference_runtimes=False,
                    allowed_node_ids=(source.node_id,),
                )
                by_chain = {}
                for alternative in local_graph.alternatives:
                    live_source_ids = {
                        item.source_id
                        for item in alternative.external_inputs
                        if item.kind.value == "LIVE_SOURCE"
                    }
                    if live_source_ids != {source_id}:
                        continue
                    if not alternative.step_placements or any(
                        step.node_id != source.node_id
                        for step in alternative.step_placements
                    ):
                        continue
                    previous = by_chain.get(alternative.chain_id)
                    if previous is None or (
                        alternative.estimated_completion_ms,
                        alternative.estimated_transfer_bytes,
                        alternative.alternative_id,
                    ) < (
                        previous.estimated_completion_ms,
                        previous.estimated_transfer_bytes,
                        previous.alternative_id,
                    ):
                        by_chain[alternative.chain_id] = alternative
                coverage_alternatives.extend(by_chain.values())
                coverage_nodes.extend(local_graph.nodes)
                coverage_edges.extend(local_graph.edges)
                coverage_pruned.extend(local_graph.pruned)
            if coverage_alternatives:
                graph = graph.model_copy(
                    update={
                        "alternatives": tuple(
                            {
                                item.alternative_id: item
                                for item in (
                                    *graph.alternatives,
                                    *coverage_alternatives,
                                )
                            }.values()
                        ),
                        "nodes": tuple(
                            {
                                item.node_id: item
                                for item in (*graph.nodes, *coverage_nodes)
                            }.values()
                        ),
                        "edges": tuple(
                            {
                                item.edge_id: item
                                for item in (*graph.edges, *coverage_edges)
                            }.values()
                        ),
                        "pruned": (*graph.pruned, *coverage_pruned),
                    }
                )
        graph = executable_runtime_graph(
            graph,
            runtime_resolver=self.runtimes,
            allow_reference_runtimes=False,
            allowed_node_ids=request.allowed_execution_node_ids,
        )
        if request.baseline_id in AUTHORED_ALWAYS_ON_BASELINES:
            # Some predicates compile structurally but still have no physical
            # realization until earlier roles or retained intervals exist.
            # Start every realizable authored chain now and defer only this
            # non-executable remainder to fixed late-bound instantiation.
            covered_demand_ids = {
                alternative.demand_id for alternative in graph.alternatives
            }
            demands = tuple(
                item for item in demands if item.demand_id in covered_demand_ids
            )
        seed_node_ids = {
            self.deployment.sources[source_id].node_id
            for source_id in eligible_sources
            if source_id in self.deployment.sources
        }
        local_seed_alternatives = tuple(
            alternative
            for alternative in graph.alternatives
            if any(
                step.provider_id.startswith("yolo_")
                and step.node_id in seed_node_ids
                for step in alternative.step_placements
            )
        )
        if (
            local_seed_alternatives
            and request.baseline_id not in AUTHORED_ALWAYS_ON_BASELINES
            and not request.allow_raw_to_trusted_site_edge
        ):
            graph = graph.model_copy(
                update={"alternatives": local_seed_alternatives}
            )
        if not graph.alternatives:
            raise ValueError(
                f"no executable seed acquisition alternative for {node.predicate.predicate_id}"
            )
        bridge = LivePlanningBridge(
            coordinator=ControlledPlanningCoordinator(
                # Bootstrap acquisition is normally common control-plane
                # scaffolding. B0/B1 are exceptions: their treatment contract is
                # an authored whole-event realization active from time zero,
                # so even the seed chain uses the baseline's authored chain policy.
                # Other baselines retain the common bounded seed selector.
                build_baseline_policy(
                    (
                        request.baseline_id
                        if request.baseline_id in AUTHORED_ALWAYS_ON_BASELINES
                        else BaselineId.FABLE
                    ),
                    planner=self.planner,
                )
            ),
            provider_registry=self.providers,
            dispatcher=self.dispatcher,
            deployment=self.deployment,
            # A seed is not yet localized evidence.  Watching only the cheapest
            # available microphone/camera can miss the event even though the
            # chosen device is technically valid.  Bootstrap acquisition is
            # common control-plane scaffolding for every evaluated policy, so
            # fan it out over the scenario-selected, modality-valid nodes.
            fanout_predicate_ids=(
                SENSOR_LOCAL_FANOUT_PREDICATES
                | {
                    item.semantic_predicate.predicate_id
                    for item in demands
                }
                if request.baseline_id == BaselineId.B0_PRODUCE_ALL
                else SENSOR_LOCAL_FANOUT_PREDICATES
            ),
            fanout_node_ids=_authored_fanout_nodes(request, seed_node_ids),
            fanout_batch_size=self.fanout_batch_size,
            fanout_batch_interval_seconds=self.fanout_batch_interval_seconds,
        )
        case = BaselinePlanningCase(
            run_id=request.run_id,
            trace_id=request.trace_id,
            request_id=request.request_id,
            event_family=compiled.family_id,
            placement_id=request.baseline_placement_id,
            # Multi-hypothesis seed acquisition partitions the provisional
            # semantic demand into one independently completable demand per
            # source.  The search input must use those same demand IDs as the
            # physical graph; retaining the pre-partition demand here both
            # violates the graph/search contract and prevents independent
            # migration of a source watch.
            frontier_demands=demands,
            all_task_demands=demands,
            frontier_graph=graph,
            whole_event_graph=graph,
            now=now,
        )
        self._capture_checkpoint_snapshot(case)
        planning = bridge.plan_and_dispatch(
            case,
            trigger=PlanningTrigger.ADMISSION,
            task_policy=TaskSchedulingPolicy(request_id=request.request_id),
        )
        if self.evaluation_record_sink is not None:
            for record in planning_records(
                case=case,
                baseline_id=request.baseline_id,
                planning=planning,
            ):
                self.evaluation_record_sink(record)
        self._seed_acquisitions[request.request_id] = _SeedAcquisitionPlan(
            request=request,
            demands=demands,
            case=case,
            bridge=bridge,
        )
        if not planning.candidates:
            raise ValueError("seed acquisition planning produced no candidates")
        if not planning.commands:
            admitted_without_activation = any(
                bool(getattr(batch, "admitted_plan_ids", ()))
                for batch in planning.admission_batches
            )
            controlled_baseline = request.baseline_id in {
                BaselineId.B0_PRODUCE_ALL,
                BaselineId.B1_STATIC_WHOLE_EVENT,
                BaselineId.B1_HANDWRITTEN_STATIC,
                BaselineId.B2_FRONTIER_FIXED_REALIZATION,
                BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
            }
            if not (controlled_baseline and admitted_without_activation):
                raise ValueError(
                    "seed acquisition planning produced no provider commands"
                )
            # A lease can attach to an already active, demand-agnostic seed
            # provider without emitting a redundant ACTIVATE command. This is
            # valid baseline execution; the subsequent bounded seed timeout
            # still detects a stale or non-producing reused instance.
        return planning

    def release_seed_acquisition(
        self, request_id: str, demand_ids: set[str], *, reason: str
    ) -> int:
        self._seed_acquisitions.pop(request_id, None)
        leases = tuple(
            managed
            for managed in self.lifecycle.leases.values()
            if managed.request_id == request_id
            and str(managed.lease.demand_id) in demand_ids
            and managed.lease.status.value not in {"RELEASED", "FAILED", "EXPIRED"}
        )
        self.dispatcher.cancel_leases(leases, reason=reason)
        # Seed acquisition is a provisional bootstrap plan. Once its one
        # immutable observation has been captured, retaining these instances
        # through the normal idle grace can block the authoritative successor
        # plan on the same GPU/node. Retire bootstrap leases synchronously;
        # ordinary frontier/provider releases keep their reuse grace.
        for managed in leases:
            self.lifecycle.release_lease(
                managed.lease.lease_id,
                immediate=True,
            )
        for demand_id in {managed.lease.demand_id for managed in leases}:
            self.lifecycle.cancel_demand(demand_id)
        return len(leases)

    def handle_seed_resource_change(
        self,
        *,
        observed_at: datetime,
        reason: str,
    ) -> tuple[tuple[str, LivePlanningResult], ...]:
        """Replan provisional seed watches against the latest deployment.

        Seed acquisition precedes the authoritative semantic hypothesis, but
        it is still live provider work.  Keeping it outside resource epochs
        made disturbances acknowledged-but-invisible until after the seed had
        already been observed.
        """

        replans: list[tuple[str, LivePlanningResult]] = []
        for request_id, acquisition in tuple(self._seed_acquisitions.items()):
            graph = self.graph_builder.build(acquisition.demands, now=observed_at)
            graph = executable_runtime_graph(
                graph,
                runtime_resolver=self.runtimes,
                allow_reference_runtimes=False,
                allowed_node_ids=acquisition.request.allowed_execution_node_ids,
            )
            resource_epoch = acquisition.case.resource_epoch + 1
            updated = replace(
                acquisition.case,
                now=observed_at,
                frontier_graph=graph,
                whole_event_graph=graph,
                resource_epoch=resource_epoch,
                replan_trigger=f"RESOURCE_EPOCH:{reason}:PENDING_SEED",
            )
            acquisition.bridge.deployment = self.deployment
            planning = acquisition.bridge.plan_and_dispatch(
                updated,
                trigger=PlanningTrigger.RESOURCE_EPOCH,
                task_policy=TaskSchedulingPolicy(request_id=request_id),
            )
            acquisition.case = updated
            if self.evaluation_record_sink is not None:
                for record in planning_records(
                    case=updated,
                    baseline_id=acquisition.request.baseline_id,
                    planning=planning,
                ):
                    self.evaluation_record_sink(record)
            replans.append((request_id, planning))
        return tuple(replans)

    def admit(self, request: LiveComplexEventRequest) -> LiveRequestAdmission:
        if request.seed is None:
            return self._reject(request, "request is waiting for an observed seed")
        if request.request_id in self._executions:
            return self._reject(request, "request_id is already active")
        if request.baseline_id not in {
            BaselineId.B0_PRODUCE_ALL,
            BaselineId.B1_STATIC_WHOLE_EVENT,
            BaselineId.B1_HANDWRITTEN_STATIC,
            BaselineId.FABLE,
            BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
            BaselineId.B4_GREEDY_FRONTIER,
            BaselineId.B2_FRONTIER_FIXED_REALIZATION,
            BaselineId.FABLE_NO_SHARING,
        }:
            return self._reject(
                request,
                "baseline is not supported by typed live admission",
            )
        compiled = EventRequestCompiler().compile(
            {"family_id": request.family_id, "parameters": request.parameters}
        )
        runtime = SemanticRuntime(
            compiled.graph,
            config=SemanticRuntimeConfig(
                request_id=request.request_id,
                hypothesis_horizon_ms=request.hypothesis_horizon_ms,
                deadline_offset_ms=request.deadline_offset_ms,
            ),
        )
        try:
            node = runtime.graph.nodes_by_key[request.seed.graph_node_key]
        except KeyError:
            return self._reject(request, "seed graph_node_key is not in the authored graph")
        if node.predicate is None:
            return self._reject(request, "seed graph node is not a primitive predicate")
        seed = SeedPredicateResult(
            occurrence_id=request.seed.occurrence_id,
            request_id=request.request_id,
            graph_hash=runtime.graph.graph.graph_hash,
            graph_node_id=node.node_id,
            semantic_predicate=node.predicate,
            truth=TruthValue.TRUE,
            confidence=request.seed.confidence,
            event_time_interval=request.seed.event_time_interval,
            binding_delta=BindingDelta(introduced=request.seed.introduced_bindings),
            provenance=ResultProvenance(
                provider_id="observed_seed_gateway",
                provider_contract_version=1,
                node_id=request.seed.node_id,
                source_ids=(request.seed.source_id,),
            ),
            observed_at=utc_now(),
        )
        transition = runtime.seed(seed)
        if not transition.hypothesis_ids:
            return self._reject(request, transition.reason or "seed did not create a hypothesis")
        hypothesis_id = transition.hypothesis_ids[0]
        hypothesis = runtime.get_hypothesis(hypothesis_id)
        frontier = runtime.get_frontier(hypothesis_id)
        if frontier is None:
            return self._reject(request, "seed completed the graph; no executable frontier remains")
        execution_node_ids = _authored_execution_nodes(
            request, request.allowed_execution_node_ids
        )
        allowed_execution_nodes = set(execution_node_ids)
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
        # A source-local reference introduced by the seed is stronger than
        # the request-wide source universe.  Keep downstream PASSES/EXITS work
        # on that camera unless a later authored cross-sensor predicate
        # explicitly introduces another source.
        reference_binding = hypothesis.role_bindings.get("reference")
        reference = (
            reference_binding.canonical_entity_id
            if reference_binding is not None
            else ""
        )
        referenced_sources = _source_for_camera_reference(reference, self.deployment)
        if referenced_sources:
            eligible_sources = tuple(
                source_id
                for source_id in eligible_sources
                if source_id in referenced_sources
            )
        demands = self.demand_compiler.compile_frontier(
            graph=runtime.graph,
            hypothesis=hypothesis,
            frontier=frontier,
            context=DemandCompileContext(
                raw_data_must_remain_local=(
                    not request.allow_raw_to_trusted_site_edge
                ),
                eligible_source_ids_by_node={
                    graph_node_id: eligible_sources
                    for graph_node_id in frontier.snapshot.enabled_node_ids
                }
            ),
        )
        demands = scope_demands_to_nodes(
            demands,
            execution_node_ids,
        )
        no_sharing = request.baseline_id == BaselineId.FABLE_NO_SHARING
        graph_builder = (
            PhysicalAlternativeGraphBuilder(
                provider_registry=self.graph_builder.providers,
                artifact_catalog=self.graph_builder.artifacts,
                deployment=self.graph_builder.deployment,
                config=self.graph_builder.config.model_copy(
                    update={
                        "allow_active_provider_reuse": False,
                        "allow_produced_artifact_reuse": False,
                    }
                ),
                active_providers=self.graph_builder.active_providers,
            )
            if no_sharing
            else self.graph_builder
        )
        frontier_graph = graph_builder.build(demands, now=seed.observed_at)
        frontier_graph = executable_runtime_graph(
            frontier_graph,
            runtime_resolver=self.runtimes,
            allow_reference_runtimes=False,
            allowed_node_ids=execution_node_ids,
        )
        missing = {item.demand_id for item in demands} - {
            item.demand_id for item in frontier_graph.alternatives
        }
        if missing:
            predicates = sorted(
                item.semantic_predicate.predicate_id
                for item in demands
                if item.demand_id in missing
            )
            return self._reject(
                request,
                "no executable live alternative for: " + ", ".join(predicates),
            )
        issues = validate_live_provider_coverage(
            frontier_graph,
            registry=self.providers,
            runtimes=self.runtimes,
        )
        if issues:
            detail = "; ".join(
                f"{item.node_id}/{item.provider_id}: {item.reason}" for item in issues
            )
            return self._reject(request, "provider coverage failed: " + detail)

        whole_event_demands = TaskDemandUniverseBuilder(
            self.demand_compiler
        ).build(
            graph=runtime.graph,
            hypothesis=hypothesis,
            context=DemandCompileContext(
                raw_data_must_remain_local=(
                    not request.allow_raw_to_trusted_site_edge
                ),
                eligible_source_ids_by_node={
                    graph_node_id: eligible_sources
                    for graph_node_id in runtime.graph.executable_predicate_nodes()
                }
            ),
            # B0/B1/B3 freeze a complete-task *physical policy*, but an
            # identity-dependent predicate is not executable until an earlier
            # frontier introduces its roles.  Skipping that structurally
            # ungrounded demand here is not adaptation: AuthoritativeLiveExecution
            # appends the same predicate when semantic progression grounds it,
            # at which point B1's frozen trace placement is enforced.
            skip_uncompilable=request.baseline_id in {
                BaselineId.B0_PRODUCE_ALL,
                BaselineId.B1_STATIC_WHOLE_EVENT,
                BaselineId.B1_HANDWRITTEN_STATIC,
                BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
            },
        )
        if request.baseline_id in {
            BaselineId.B0_PRODUCE_ALL,
            BaselineId.B1_STATIC_WHOLE_EVENT,
            BaselineId.B1_HANDWRITTEN_STATIC,
            BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
        }:
            # Whole-event baselines can compile placeholders for roles created
            # by earlier graph nodes. Such a demand has no executable physical
            # contract yet, regardless of predicate type. Defer every
            # structurally unbound demand until semantic progression grounds
            # it; the baseline-specific policy still chooses its chain and
            # placement when that contract becomes executable. This is scoped
            # to B0/B1/B3 and does not alter FABLE's frontier planning.
            if request.baseline_id in B1_BASELINES:
                # B1 is an authored pipeline which is active for the complete
                # trace.  Structural producer roles (for example the vehicle
                # consumed by EXITS) are intentionally unresolved at
                # admission: the always-on provider grounds them when an
                # observation arrives.  Only an identity comparison must wait
                # for both concrete endpoints; running SAME_ENTITY against
                # symbolic roles is neither useful nor a valid fixed chain.
                whole_event_demands = tuple(
                    demand
                    for demand in whole_event_demands
                    if not (
                        demand.semantic_predicate.predicate_id == "SAME_ENTITY"
                        and any(
                            str(value).startswith("__structural_unbound__:")
                            for value in demand.bound_roles.values()
                        )
                    )
                )
            else:
                whole_event_demands = tuple(
                    demand
                    for demand in whole_event_demands
                    if not any(
                        str(value).startswith("__structural_unbound__:")
                        for value in demand.bound_roles.values()
                    )
                )
        whole_event_demands = scope_demands_to_nodes(
            whole_event_demands,
            execution_node_ids,
        )
        whole_event_graph = graph_builder.build(
            whole_event_demands,
            now=seed.observed_at,
        )
        whole_event_graph = executable_runtime_graph(
            whole_event_graph,
            runtime_resolver=self.runtimes,
            allow_reference_runtimes=False,
            allowed_node_ids=execution_node_ids,
        )
        whole_event_baseline = request.baseline_id in {
            BaselineId.B0_PRODUCE_ALL,
            BaselineId.B1_STATIC_WHOLE_EVENT,
            BaselineId.B1_HANDWRITTEN_STATIC,
            BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
        }
        if whole_event_baseline:
            missing_whole = {
                item.demand_id for item in whole_event_demands
            } - {
                item.demand_id for item in whole_event_graph.alternatives
            }
            if missing_whole:
                predicates = sorted(
                    item.semantic_predicate.predicate_id
                    for item in whole_event_demands
                    if item.demand_id in missing_whole
                )
                return self._reject(
                    request,
                    "no executable whole-event alternative for: "
                    + ", ".join(predicates),
                )
            whole_issues = validate_live_provider_coverage(
                whole_event_graph,
                registry=self.providers,
                runtimes=self.runtimes,
            )
            if whole_issues:
                detail = "; ".join(
                    f"{item.node_id}/{item.provider_id}: {item.reason}"
                    for item in whole_issues
                )
                return self._reject(
                    request,
                    "whole-event provider coverage failed: " + detail,
                )

        policy = build_baseline_policy(
            request.baseline_id,
            planner=self.planner,
        )
        bridge = LivePlanningBridge(
            coordinator=ControlledPlanningCoordinator(policy),
            provider_registry=self.providers,
            dispatcher=self.dispatcher,
            deployment=self.deployment,
            fanout_predicate_ids=(
                SENSOR_LOCAL_FANOUT_PREDICATES
                | {
                    node.predicate.predicate_id
                    for node in runtime.graph.nodes_by_id.values()
                    if node.predicate is not None
                }
                if request.baseline_id == BaselineId.B0_PRODUCE_ALL
                else SENSOR_LOCAL_FANOUT_PREDICATES
            ),
            # Seed nodes are an acquisition preference, not the exhaustive
            # successor search space.  Sensor-local successor predicates must
            # cover the scenario-selected execution nodes; otherwise the
            # first camera to emit a seed permanently pins later evidence to
            # that camera even when replay is active elsewhere.
            fanout_node_ids=_authored_fanout_nodes(
                request, set(execution_node_ids)
            ),
            fanout_batch_size=self.fanout_batch_size,
            fanout_batch_interval_seconds=self.fanout_batch_interval_seconds,
        )
        task_policy = TaskSchedulingPolicy(request_id=request.request_id)
        case = BaselinePlanningCase(
            run_id=request.run_id,
            trace_id=request.trace_id,
            request_id=request.request_id,
            event_family=compiled.family_id,
            placement_id=request.baseline_placement_id,
            frontier_demands=demands,
            all_task_demands=whole_event_demands,
            frontier_graph=frontier_graph,
            whole_event_graph=whole_event_graph,
            now=seed.observed_at,
        )
        self._capture_checkpoint_snapshot(case)
        planning = bridge.plan_and_dispatch(
            case,
            trigger=PlanningTrigger.ADMISSION,
            task_policy=task_policy,
        )
        if self.evaluation_record_sink is not None:
            for record in planning_records(
                case=case,
                baseline_id=request.baseline_id,
                planning=planning,
            ):
                self.evaluation_record_sink(record)
        if not planning.candidates:
            return self._reject(
                request,
                "planning produced no dispatchable candidate: "
                + planning.decision.reason,
            )
        execution = AuthoritativeLiveExecution(
            demand_compiler=self.demand_compiler,
            graph_builder=graph_builder,
            deployment=self.deployment,
            bridge=bridge,
            checkpoint_controller=self.checkpoints,
            runtime_resolver=self.runtimes,
        )
        execution.register(
            LiveRequestState(
                run_id=request.run_id,
                trace_id=request.trace_id,
                event_family=compiled.family_id,
                placement_id=request.baseline_placement_id,
                runtime=runtime,
                whole_event_demands=whole_event_demands,
                whole_event_graph=whole_event_graph,
                task_policy=task_policy,
                baseline_id=request.baseline_id,
                evaluation_record_sink=self.evaluation_record_sink,
                checkpoint_snapshot_sink=self._capture_checkpoint_snapshot,
                planning_cases={
                    str(hypothesis_id): case
                    for hypothesis_id in transition.hypothesis_ids
                },
                allowed_execution_node_ids=request.allowed_execution_node_ids,
                allow_raw_to_trusted_site_edge=(
                    request.allow_raw_to_trusted_site_edge
                ),
                coverage_node_id=request.seed.node_id,
                active_spatial_deployment_id=(
                    str(request.parameters["active_spatial_deployment_id"])
                    if request.parameters.get("active_spatial_deployment_id")
                    else None
                ),
                spatial_maximum_observation_groups=int(
                    request.parameters.get(
                        "spatial_maximum_observation_groups", 1
                    )
                ),
                # Provider outputs can race semantic-frontier activation for
                # every policy, including FABLE.  Keep a bounded request-local
                # observation history so correctness does not depend on MQTT
                # arrival order.  Planning remains dynamic; this retains typed
                # results, not always-on provider execution.
                early_observations=EarlyObservationBuffer(
                    max_observations=16_384,
                    retention_ms=request.hypothesis_horizon_ms,
                ),
            )
        )
        self._executions[request.request_id] = execution
        return LiveRequestAdmission(
            response=LiveComplexEventResponse(
                request_message_id=request.message_id,
                request_id=request.request_id,
                accepted=True,
                status="ADMITTED",
                hypothesis_ids=tuple(str(item) for item in transition.hypothesis_ids),
                candidate_ids=tuple(
                    item.candidate_id or "" for item in planning.candidates
                ),
                command_count=len(planning.commands),
                seed_occurrence_id=request.seed.occurrence_id,
                seed_action="INITIAL_ADMISSION",
                active_seed_hypothesis_count=len(runtime.active_hypotheses),
            ),
            planning=planning,
        )

    def admit_additional_seed(
        self,
        request: LiveComplexEventRequest,
    ) -> LiveRequestAdmission:
        """Add one bounded seed to an already-active semantic runtime."""

        if request.seed is None:
            return self._reject(request, "additional admission requires an observed seed")
        execution = self._executions.get(request.request_id)
        if execution is None:
            return self._reject(request, "request is no longer active")
        state = execution.request_state(request.request_id)
        try:
            node = state.runtime.graph.nodes_by_key[request.seed.graph_node_key]
        except KeyError:
            return self._reject(request, "seed graph_node_key is not in the authored graph")
        if node.predicate is None:
            return self._reject(request, "seed graph node is not a primitive predicate")
        seed = SeedPredicateResult(
            occurrence_id=request.seed.occurrence_id,
            request_id=request.request_id,
            graph_hash=state.runtime.graph.graph.graph_hash,
            graph_node_id=node.node_id,
            semantic_predicate=node.predicate,
            truth=TruthValue.TRUE,
            confidence=request.seed.confidence,
            event_time_interval=request.seed.event_time_interval,
            binding_delta=BindingDelta(introduced=request.seed.introduced_bindings),
            provenance=ResultProvenance(
                provider_id="observed_seed_gateway",
                provider_contract_version=1,
                node_id=request.seed.node_id,
                source_ids=(request.seed.source_id,),
            ),
            observed_at=utc_now(),
        )
        progression = execution.add_seed(seed)
        accepted = bool(progression.transition.hypothesis_ids)
        return LiveRequestAdmission(
            response=LiveComplexEventResponse(
                request_message_id=request.message_id,
                request_id=request.request_id,
                accepted=accepted,
                status="ADMITTED" if accepted else "REJECTED",
                hypothesis_ids=tuple(
                    str(item) for item in progression.transition.hypothesis_ids
                ),
                candidate_ids=tuple(
                    candidate.candidate_id or ""
                    for planning in progression.planning
                    for candidate in planning.candidates
                ),
                command_count=sum(
                    len(planning.commands) for planning in progression.planning
                ),
                seed_occurrence_id=request.seed.occurrence_id,
                seed_action=(
                    "ADDITIONAL_HYPOTHESIS" if accepted else "REJECTED"
                ),
                active_seed_hypothesis_count=len(
                    state.runtime.active_hypotheses
                ),
                reason=progression.transition.reason,
            ),
            planning=(
                progression.planning[0] if progression.planning else None
            ),
        )

    def handle_result(self, result):
        execution = self._executions.get(result.request_id)
        if execution is None:
            raise ValueError(f"no live request manager state for {result.request_id}")
        progression = execution.handle_result(result)
        if not execution.has_request(result.request_id):
            self._executions.pop(result.request_id, None)
            self._completed_recovery_scopes.pop(result.request_id, None)
        return progression

    def inject_b1_static_vehicle_history(
        self,
        request_id: str,
        observations: list[dict[str, object]],
    ) -> int:
        """Satisfy B1 lookback from compact evidence produced while always on.

        This is deliberately B1-only and does not read raw segments or invoke
        a retrospective provider.  It rematerializes already-observed vehicle
        facts against the authoritative late-bound demand envelope.
        """

        execution = self._executions.get(request_id)
        if execution is None:
            return 0
        state = execution.request_state(request_id)
        demands = [
            item
            for item in state.whole_event_demands
            if item.semantic_predicate.predicate_id == "VEHICLE_PRESENT_BEFORE"
        ]
        emitted = 0
        for demand in demands:
            role_variables = {
                role.role_name: role.variable
                for role in demand.semantic_predicate.roles
            }
            seen: set[tuple[str, str]] = set()
            for observation in sorted(
                observations,
                key=lambda item: str(
                    (item.get("event_time_interval") or {}).get("end") or ""
                ),
                reverse=True,
            ):
                interval = EventTimeInterval.model_validate(
                    observation["event_time_interval"]
                )
                if not demand.event_time_interval.overlaps(interval):
                    continue
                bindings = dict(observation.get("bindings") or {})
                vehicle = str(bindings.get("vehicle") or "")
                source_id = str(observation.get("source_id") or "")
                if not vehicle or (vehicle, source_id) in seen:
                    continue
                seen.add((vehicle, source_id))
                introduced = {}
                vehicle_variable = role_variables.get("vehicle")
                if vehicle_variable:
                    introduced[vehicle_variable] = vehicle
                validated = {
                    role_variables[role_name]: value
                    for role_name, value in demand.bound_roles.items()
                    if role_name in role_variables
                    and not value.startswith("__structural_unbound__:")
                }
                now = utc_now()
                result = PredicateResult(
                    occurrence_id=(
                        "b1-static-history:"
                        + str(observation.get("occurrence_id") or uuid7())
                    ),
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
                    confidence=float(observation.get("confidence") or 1.0),
                    event_time_interval=interval,
                    binding_delta=BindingDelta(
                        introduced=introduced,
                        validated=validated,
                    ),
                    provenance=ResultProvenance(
                        provider_id="b1_static_live_history",
                        provider_contract_version=1,
                        node_id=str(observation.get("node_id") or source_id),
                        source_ids=(source_id,),
                    ),
                    processing_started_at=now,
                    processing_completed_at=now,
                )
                # Provider results may complete the CE while this compact
                # history loop is still iterating.  Terminal cleanup forgets
                # the admission decision immediately; do not submit another
                # historical candidate into that retired planning context.
                if not execution.bridge.coordinator.has_decision(request_id):
                    return emitted
                self.handle_result(result)
                emitted += 1
                # The authored matcher is existential and historically emits
                # a bounded candidate set. Preserve that same upper bound.
                if emitted >= 8:
                    return emitted
        return emitted

    def handle_heartbeat(self, heartbeat):
        progressions = []
        for request_id, execution in tuple(self._executions.items()):
            for progression in execution.handle_heartbeat(heartbeat):
                progressions.append((request_id, progression))
            if not execution.has_request(request_id):
                self._executions.pop(request_id, None)
                self._completed_recovery_scopes.pop(request_id, None)
        return tuple(progressions)

    def handle_resource_change(
        self,
        *,
        demand_ids,
        reason: str,
        observed_at,
    ):
        affected = {str(item) for item in demand_ids}
        replans = []
        affected_requests: list[tuple[str, AuthoritativeLiveExecution]] = []
        for request_id, execution in tuple(self._executions.items()):
            state = execution.request_state(request_id)
            request_demands = {
                str(item.demand_id) for item in state.whole_event_demands
            }
            request_demands.update(
                str(item.demand_id)
                for case in state.planning_cases.values()
                for item in (*case.frontier_demands, *case.all_task_demands)
            )
            direct_matches = affected.intersection(request_demands)
            lineage_matches = {
                demand_id
                for demand_id in affected
                if self._demand_request_history.get(demand_id) == request_id
            }
            if affected and not direct_matches and not lineage_matches:
                continue
            for demand_id in direct_matches:
                self._demand_request_history[demand_id] = request_id
            affected_requests.append((request_id, execution))

        cross_request_joint = (
            os.environ.get("FABLE_JOINT_RESOURCE_EPOCH_PLANNING", "0") == "1"
            and len(affected_requests) > 1
            and all(
                execution.request_state(request_id).baseline_id
                == BaselineId.FABLE
                for request_id, execution in affected_requests
            )
        )
        if cross_request_joint:
            prepared: list[
                tuple[
                    str,
                    AuthoritativeLiveExecution,
                    tuple[BaselinePlanningCase, ...],
                ]
            ] = []
            all_cases: list[BaselinePlanningCase] = []
            for request_id, execution in affected_requests:
                cases = execution.prepare_resource_epoch_cases(
                    request_id,
                    observed_at=observed_at,
                    reason=reason,
                )
                prepared.append((request_id, execution, cases))
                all_cases.extend(cases)
            if all_cases:
                batch_request_id = "resource-batch:" + ":".join(
                    sorted(request_id for request_id, _ in affected_requests)
                )
                joint = joint_batch_case(
                    all_cases,
                    run_id=all_cases[0].run_id,
                    request_id=batch_request_id,
                )
                joint = replace(
                    joint,
                    resource_epoch=max(case.resource_epoch for case in all_cases),
                    replan_trigger=(
                        f"RESOURCE_EPOCH:{reason}:JOINT_REQUESTS"
                    ),
                )
                policy = affected_requests[0][1].bridge.coordinator.policy
                global_decision = policy.plan(joint)
                selected = set(global_decision.selected_alternative_ids)
                for request_id, execution, cases in prepared:
                    scoped_cases = tuple(
                        replace(
                            case,
                            frontier_graph=case.frontier_graph.model_copy(
                                update={
                                    "alternatives": tuple(
                                        alternative
                                        for alternative in case.frontier_graph.alternatives
                                        if alternative.alternative_id in selected
                                    )
                                }
                            ),
                            replan_trigger=(
                                f"RESOURCE_EPOCH:{reason}:JOINT_REQUESTS"
                            ),
                        )
                        for case in cases
                    )
                    planning = execution.dispatch_prepared_resource_epoch(
                        request_id,
                        scoped_cases,
                        reason=f"{reason}:JOINT_REQUESTS",
                        allow_joint_hypotheses=False,
                    )
                    if planning:
                        replans.append((request_id, planning))
            return tuple(replans)

        for request_id, execution in affected_requests:
            planning = execution.handle_resource_epoch(
                request_id,
                observed_at=observed_at,
                reason=reason,
            )
            if planning:
                replans.append((request_id, planning))
        return tuple(replans)

    def handle_source_recovery(
        self,
        *,
        node_id: str,
        recovery_intervals: dict[str, EventTimeInterval],
        demand_ids,
        reason: str,
        observed_at,
    ):
        """Replan active work with an exact, source-local outage catch-up gap."""

        affected = {str(item) for item in demand_ids}
        # A recovered node with no demands that were active when it became
        # unavailable has no CE evidence obligation. Treating the empty set as
        # a wildcard caused unrelated cameras (notably orin16 in the convoy
        # pilot) to receive retrospective replay work.
        if not affected:
            return ()
        replans = []
        for request_id, execution in tuple(self._executions.items()):
            state = execution.request_state(request_id)
            request_demands = {
                str(item.demand_id) for item in state.whole_event_demands
            }
            request_demands.update(
                str(item.demand_id)
                for case in state.planning_cases.values()
                for item in (*case.frontier_demands, *case.all_task_demands)
            )
            direct_matches = affected.intersection(request_demands)
            lineage_matches = {
                demand_id
                for demand_id in affected
                if self._demand_request_history.get(demand_id) == request_id
            }
            if not direct_matches and not lineage_matches:
                continue
            effective_intervals = dict(recovery_intervals)
            if not effective_intervals:
                # Administrative loss can precede the first replay-source
                # watermark. Infer only this node's eligible sources and use
                # their already-bounded semantic demand intervals.
                normalized_node = node_id.removeprefix("dvpg_gq_")
                if normalized_node.startswith("orin_"):
                    normalized_node = "orin" + normalized_node.removeprefix("orin_")
                current_demands = tuple(state.whole_event_demands) + tuple(
                    item
                    for case in state.planning_cases.values()
                    for item in (*case.frontier_demands, *case.all_task_demands)
                )
                for demand in (
                    item
                    for item in current_demands
                    if str(item.demand_id) in direct_matches
                    or any(
                        self._demand_request_history.get(demand_id) == request_id
                        for demand_id in affected
                    )
                ):
                    for source_id in demand.eligible_source_ids:
                        if not (
                            source_id.startswith(f"{normalized_node}_")
                            or source_id.startswith(f"{normalized_node}:")
                        ):
                            continue
                        prior = effective_intervals.get(source_id)
                        interval = demand.event_time_interval
                        effective_intervals[source_id] = (
                            interval
                            if prior is None
                            else EventTimeInterval(
                                start=min(prior.start, interval.start),
                                end=max(prior.end, interval.end),
                            )
                        )
            if not effective_intervals:
                continue
            # Scope the recovered source against the request's current demand
            # generation. Old outage IDs remain useful only for identifying
            # request ownership; replay commands must carry current IDs.
            scoped_demand_ids = request_demands
            recovery_scope = (
                node_id,
                tuple(
                    sorted(
                        (
                            source_id,
                            interval.start.isoformat(),
                            interval.end.isoformat(),
                        )
                        for source_id, interval in effective_intervals.items()
                    )
                ),
                tuple(sorted(scoped_demand_ids)),
            )
            completed = self._completed_recovery_scopes.setdefault(request_id, set())
            if recovery_scope in completed:
                continue
            planning = execution.handle_resource_epoch(
                request_id,
                observed_at=observed_at,
                reason=f"NODE_RECOVERED:{node_id}:{reason}",
                recovery_intervals=effective_intervals,
                recovery_demand_ids=scoped_demand_ids,
            )
            if planning:
                completed.add(recovery_scope)
                replans.append((request_id, planning))
        return tuple(replans)

    def update_network_deployment(
        self,
        deployment: DeploymentGraph,
        *,
        execution_time_multiplier: float = 1.0,
        queue_delay_ms: int = 0,
        compute_target_node_id: str | None = None,
    ) -> None:
        """Install a validated planner-visible network profile for live replans."""

        self.deployment = deployment
        config = self.graph_builder.config
        multipliers = dict(config.node_execution_time_multipliers)
        queue_delays = dict(config.node_queue_delay_ms)
        if compute_target_node_id is not None:
            if execution_time_multiplier == 1.0:
                multipliers.pop(compute_target_node_id, None)
            else:
                multipliers[compute_target_node_id] = execution_time_multiplier
            if queue_delay_ms == 0:
                queue_delays.pop(compute_target_node_id, None)
            else:
                queue_delays[compute_target_node_id] = queue_delay_ms
        config = config.model_copy(
            update={
                "node_execution_time_multipliers": multipliers,
                "node_queue_delay_ms": queue_delays,
            }
        )
        self.graph_builder = PhysicalAlternativeGraphBuilder(
            provider_registry=self.graph_builder.providers,
            artifact_catalog=self.graph_builder.artifacts,
            deployment=deployment,
            config=config,
            active_providers=self.graph_builder.active_providers,
        )
        self.planner = BoundedLabelPlanner(
            provider_registry=self.providers,
            artifact_catalog=self.graph_builder.artifacts,
            deployment=deployment,
        )
        for execution in self._executions.values():
            execution.update_network_deployment(deployment)

    def has_request(self, request_id: str) -> bool:
        return request_id in self._executions

    def evict_unprogressed_seed(
        self,
        request_id: str,
        hypothesis_id: UUID,
        *,
        seed_event_time: EventTimeInterval,
        minimum_progress_gap_ms: int,
        force: bool = False,
    ) -> bool:
        execution = self._executions.get(request_id)
        if execution is None:
            return False
        return execution.evict_unprogressed_seed(
            request_id,
            hypothesis_id,
            seed_event_time=seed_event_time,
            minimum_progress_gap_ms=minimum_progress_gap_ms,
            force=force,
        )

    def cancel(
        self,
        request: LiveComplexEventCancelRequest,
    ) -> LiveComplexEventCancelResponse:
        execution = self._executions.pop(request.request_id, None)
        self._completed_recovery_scopes.pop(request.request_id, None)
        managed_leases = tuple(
            item
            for item in self.lifecycle.active_leases
            if item.request_id == request.request_id
        )
        sweep_commands = self.dispatcher.sweep_request(
            tuple(self.deployment.nodes),
            request_id=request.request_id,
            reason=request.reason,
        )
        if execution is None and not managed_leases:
            return LiveComplexEventCancelResponse(
                request_message_id=request.message_id,
                request_id=request.request_id,
                status="CANCELLED",
                cancel_command_count=len(sweep_commands),
                reason=(
                    f"{request.reason}; node cleanup sweep dispatched because "
                    "orchestrator state was absent"
                ),
            )
        commands = self.dispatcher.cancel_leases(
            managed_leases,
            reason=request.reason,
        )
        outcome = self.checkpoints.cancel_task(
            request_id=request.request_id,
            reason=request.reason,
        )
        if execution is not None:
            execution.cancel(request.request_id)
        persist_retirement = getattr(
            self.dispatcher, "persist_request_retirement", None
        )
        if persist_retirement is not None:
            persist_retirement(request.request_id, reason=request.reason)
        return LiveComplexEventCancelResponse(
            request_message_id=request.message_id,
            request_id=request.request_id,
            status="CANCELLED",
            cancelled_active_execution=execution is not None,
            cancelled_demand_count=len(outcome.cancelled_demand_ids),
            released_lease_count=len(outcome.released_lease_ids),
            cancel_command_count=len(commands) + len(sweep_commands),
            reason=request.reason,
        )

    @staticmethod
    def _reject(
        request: LiveComplexEventRequest, reason: str
    ) -> LiveRequestAdmission:
        return LiveRequestAdmission(
            response=LiveComplexEventResponse(
                request_message_id=request.message_id,
                request_id=request.request_id,
                accepted=False,
                status="REJECTED",
                reason=reason,
            )
        )


@dataclass
class _AdmittedSeedCandidate:
    hypothesis_id: UUID
    seed_event_time: EventTimeInterval
    partition: str
    identity: str


@dataclass
class _PendingSeed:
    request: LiveComplexEventRequest
    predicate_id: str
    predicate_parameters: dict[str, JSONValue]
    binding_variables: dict[str, str]
    registered_at: datetime
    expires_at: datetime
    first_matching_event_time: datetime | None = None
    latest_matching_event_time: datetime | None = None
    admitted_seed_count: int = 0
    observed_seed_keys: set[tuple[str, str, str, str]] | None = None
    admitted_seed_partitions: set[str] | None = None
    admitted_seed_partition_counts: dict[str, int] | None = None
    admitted_seed_candidates: list[_AdmittedSeedCandidate] | None = None
    reported_duplicate_partitions: set[str] | None = None
    acquisition_demand_ids: set[str] | None = None
    # B1's authored visual pipeline runs from the beginning. Retain its
    # compact typed vehicle predicates while the audio seed is pending; B1 is
    # not permitted to request raw retrospective replay after the seed.
    static_vehicle_history: list[dict[str, object]] = field(default_factory=list)
    static_vehicle_history_injected: bool = False


class PendingSeedRegistry:
    """Match typed provider observations to authored pending requests."""

    def __init__(self, manager: LiveRequestManager) -> None:
        self.manager = manager
        self._pending: dict[str, _PendingSeed] = {}
        # The path is campaign-frozen for the lifetime of the orchestrator.
        # Parsing the YAML for every high-rate track/predicate observation
        # starves sparse audio seeds and can consume the complete watch window.
        self._static_pipeline_registry = StaticPipelineRegistry.load(
            static_pipeline_registry_path()
        )

    def register(self, request: LiveComplexEventRequest) -> LiveComplexEventResponse:
        if request.seed is not None:
            return self.manager.admit(request).response
        assert request.seed_graph_node_key is not None
        if request.request_id in self._pending or self.manager.has_request(
            request.request_id
        ):
            return LiveComplexEventResponse(
                request_message_id=request.message_id,
                request_id=request.request_id,
                accepted=False,
                status="REJECTED",
                reason="request_id is already active or waiting",
            )
        compiled = EventRequestCompiler().compile(
            {"family_id": request.family_id, "parameters": request.parameters}
        )
        runtime = SemanticRuntime(
            compiled.graph,
            config=SemanticRuntimeConfig(request_id=request.request_id),
        )
        try:
            node = runtime.graph.nodes_by_key[request.seed_graph_node_key]
        except KeyError:
            return LiveComplexEventResponse(
                request_message_id=request.message_id,
                request_id=request.request_id,
                accepted=False,
                status="REJECTED",
                reason="seed_graph_node_key is not in the authored graph",
            )
        if node.predicate is None:
            return LiveComplexEventResponse(
                request_message_id=request.message_id,
                request_id=request.request_id,
                accepted=False,
                status="REJECTED",
                reason="seed graph node is not a primitive predicate",
            )
        registered_at = utc_now()
        try:
            acquisition = self.manager.acquire_seed(request)
        except Exception as exc:
            return LiveComplexEventResponse(
                request_message_id=request.message_id,
                request_id=request.request_id,
                accepted=False,
                status="REJECTED",
                reason=f"seed acquisition planning failed: {exc}",
            )
        # Track the complete admitted acquisition scope, not merely demands
        # which happened to emit a new ACTIVATE command. A candidate may
        # attach to a reused process and emit no command while still owning a
        # lease/capacity reservation. Omitting that demand stranded structural
        # B1 leases across the seed handoff and caused the correctly authored
        # successor plan to reuse stale, unbound execution state.
        acquisition_demand_ids = {
            str(demand.demand_id)
            for candidate in acquisition.candidates
            for demand in candidate.demands
        }
        self._pending[request.request_id] = _PendingSeed(
            request=request,
            predicate_id=node.predicate.predicate_id,
            predicate_parameters=node.predicate.parameters,
            binding_variables={
                key: variable
                for role in node.predicate.roles
                for key, variable in (
                    (role.role_name, role.variable),
                    (role.variable, role.variable),
                )
                if variable
                in {item.role_name for item in runtime.graph.graph.roles}
            },
            registered_at=registered_at,
            expires_at=registered_at
            + timedelta(seconds=request.seed_timeout_seconds),
            acquisition_demand_ids=acquisition_demand_ids,
        )
        return LiveComplexEventResponse(
            request_message_id=request.message_id,
            request_id=request.request_id,
            accepted=True,
            status="WATCHING",
            seed_action="WATCH_REGISTERED",
            reason=f"watching for authored seed predicate {node.predicate.predicate_id}",
        )

    def cancel(
        self,
        request: LiveComplexEventCancelRequest,
    ) -> LiveComplexEventCancelResponse:
        pending = self._pending.pop(request.request_id, None)
        if pending is not None:
            self.manager.release_seed_acquisition(
                request.request_id,
                pending.acquisition_demand_ids or set(),
                reason=request.reason or "seed watch cancelled",
            )
            if pending.admitted_seed_count:
                active = self.manager.cancel(request)
                return active.model_copy(
                    update={"cancelled_pending_watch": True}
                )
            return LiveComplexEventCancelResponse(
                request_message_id=request.message_id,
                request_id=request.request_id,
                status="CANCELLED",
                cancelled_pending_watch=True,
                reason=request.reason,
            )
        return self.manager.cancel(request)

    def recover_node(
        self,
        *,
        node_id: str,
        demand_ids,
    ) -> tuple[LivePlanningResult, ...]:
        """Recover pending seed watches whose acquisition ran on this node."""

        affected = {str(item) for item in demand_ids}
        recovered: list[LivePlanningResult] = []
        for pending in self._pending.values():
            if not affected.intersection(pending.acquisition_demand_ids or set()):
                continue
            planning = self.manager.acquire_seed(
                pending.request,
                recovery_node_id=node_id,
            )
            pending.acquisition_demand_ids = {
                *(pending.acquisition_demand_ids or set()),
                *(
                    str(demand.demand_id)
                    for candidate in planning.candidates
                    for demand in candidate.demands
                ),
            }
            recovered.append(planning)
        return tuple(recovered)

    def observe(
        self,
        topic: str,
        payload: bytes | str | dict[str, object],
    ) -> tuple[LiveComplexEventResponse, ...]:
        observation = self._normalize_observation(topic, payload)
        if observation is None:
            return ()
        now = utc_now()
        responses = []
        pending_replay_ids = {
            pending_request_id: item.request.replay_id
            for pending_request_id, item in self._pending.items()
        }
        for request_id, pending in tuple(self._pending.items()):
            trusted_request_id = observation.get("_trusted_request_id")
            if trusted_request_id is not None and trusted_request_id != request_id:
                # A shared provider instance emits one authoritative result
                # envelope, but the underlying observation may satisfy more
                # than one concurrently registered FABLE watch. Permit that
                # fan-out only for the explicitly gated joint experiment and
                # only inside the exact same replay generation. Predicate,
                # source, node, parameter, and event-time checks below remain
                # mandatory for every consumer.
                shared_replay_evidence = (
                    os.environ.get(
                        "FABLE_JOINT_RESOURCE_EPOCH_PLANNING", "0"
                    )
                    == "1"
                    and pending.request.baseline_id == BaselineId.FABLE
                    and pending.request.replay_id is not None
                    and (
                        observation.get("replay_id")
                        == pending.request.replay_id
                        or pending_replay_ids.get(str(trusted_request_id))
                        == pending.request.replay_id
                    )
                )
                if not shared_replay_evidence:
                    continue
            if pending.expires_at <= now:
                self._pending.pop(request_id, None)
                self.manager.release_seed_acquisition(
                    request_id,
                    pending.acquisition_demand_ids or set(),
                    reason="multi-seed watch expired",
                )
                if not pending.admitted_seed_count:
                    responses.append(
                        LiveComplexEventResponse(
                            request_message_id=pending.request.message_id,
                            request_id=request_id,
                            accepted=False,
                            status="REJECTED",
                            seed_action="REJECTED",
                            reason="seed watch expired",
                        )
                    )
                continue
            if observation["predicate_id"] == "__TRACK_SET__":
                # B1 is authored as an always-on fixed pipeline. Preserve the
                # compact tracker output produced before its trigger instead
                # of relying on the tracker to happen to finalize an EXITS
                # predicate before seed admission.
                if pending.request.baseline_id not in B1_BASELINES:
                    continue
                replay_id = str(observation.get("replay_id") or "")
                if not pending.request.replay_id or replay_id != pending.request.replay_id:
                    continue
                source_id = str(observation.get("source_id") or "")
                source = self.manager.deployment.sources.get(source_id)
                node_id = source.node_id if source is not None else source_id
                placement = self._static_pipeline_registry.get_placement(
                    pending.request.baseline_placement_id,
                    trace_id=pending.request.trace_id,
                )
                if placement is not None and (
                    node_id not in placement.allowed_node_ids
                    or source_id not in placement.allowed_source_ids
                ):
                    continue
                event_time = str(observation.get("event_time") or "")
                for track in observation.get("tracks") or ():
                    class_name = str(track.get("class_name") or "").lower()
                    if class_name not in {
                        "car", "truck", "bus", "motorcycle", "vehicle"
                    }:
                        continue
                    vehicle = str(track.get("scoped_track_id") or "")
                    track_time = str(track.get("event_time") or event_time)
                    if not vehicle or not track_time:
                        continue
                    pending.static_vehicle_history.append({
                        "occurrence_id": (
                            f"b1-track-history:{replay_id}:{vehicle}:{track_time}"
                        ),
                        "event_time_interval": {
                            "start": track_time,
                            "end": track_time,
                        },
                        "bindings": {"vehicle": vehicle},
                        "confidence": float(track.get("confidence") or 1.0),
                        "source_id": source_id,
                        "node_id": node_id,
                        "replay_id": replay_id,
                        "history_artifact_type": "track_set.v1",
                    })
                if len(pending.static_vehicle_history) > 16_384:
                    del pending.static_vehicle_history[:4096]
                continue
            if (
                pending.request.baseline_id in B1_BASELINES
                and observation.get("_trusted_request_id") is None
                and observation.get("predicate_id")
                in {
                    "INSIDE",
                    "ENTERS",
                    "MOVING",
                    "STOPPED",
                    "PASSES",
                    "FOLLOWS",
                    "DISTANCE_LT",
                }
                and pending.request.replay_id is not None
                and observation.get("replay_id") == pending.request.replay_id
            ):
                placement = self._static_pipeline_registry.get_placement(
                    pending.request.baseline_placement_id,
                    trace_id=pending.request.trace_id,
                )
                raw_source = str(observation.get("source_id") or "")
                raw_node = str(observation.get("node_id") or raw_source)
                if placement is None or raw_node in placement.allowed_node_ids:
                    pending.static_vehicle_history.append(dict(observation))
                    if len(pending.static_vehicle_history) > 16_384:
                        del pending.static_vehicle_history[:4096]
            if observation["predicate_id"] != pending.predicate_id:
                continue
            if not self._parameters_match(
                pending.predicate_parameters,
                observation,
            ):
                continue
            observed_interval = EventTimeInterval.model_validate(
                observation["event_time_interval"]
            )
            allowed_interval = pending.request.allowed_seed_event_time_interval
            if (
                allowed_interval is not None
                and not allowed_interval.overlaps(observed_interval)
            ):
                continue
            minimum_watch_age_ms = int(
                pending.predicate_parameters.get("minimum_watch_age_ms") or 0
            )
            pending.first_matching_event_time = min(
                pending.first_matching_event_time or observed_interval.start,
                observed_interval.start,
            )
            pending.latest_matching_event_time = max(
                pending.latest_matching_event_time or observed_interval.end,
                observed_interval.end,
            )
            if minimum_watch_age_ms > 0 and pending.latest_matching_event_time < (
                pending.first_matching_event_time
                + timedelta(milliseconds=minimum_watch_age_ms)
            ):
                # A noisy classifier can emit alarm-like evidence immediately
                # when replay begins. Measure confirmation in provider event
                # time from the first matching observation. Watch registration
                # happens before the readiness barrier and is therefore not a
                # valid replay-time origin.
                continue
            if (
                pending.request.reject_seed_before_registration
                and allowed_interval is None
                and observed_interval.end < pending.registered_at
            ):
                # Provider queues and restarted analytics may flush observations
                # from a previous live stream. For replay, freshness is instead
                # established by the exact replay_id and approved source-time
                # interval below; comparing recording time to wall registration
                # time would reject every valid historical trace.
                continue
            if (
                pending.request.replay_id is not None
                and trusted_request_id is None
                and observation.get("replay_id")
                != pending.request.replay_id
            ):
                continue
            source_id = str(observation["source_id"])
            # Vehicle/audio services publish their concrete node identifier
            # (for example ``dvpg_gq_orin_14``), whereas authored replay
            # requests are scoped to deployment source identifiers such as
            # ``orin14_camera``. Resolve that boundary through the deployment
            # graph, but only when it yields exactly one explicitly allowed
            # source. Ambiguous or out-of-scope evidence remains rejected.
            semantic_source_ids = (
                pending.request.semantic_seed_source_ids
                or pending.request.allowed_seed_source_ids
            )
            semantic_node_ids = (
                pending.request.semantic_seed_node_ids
                or pending.request.allowed_seed_node_ids
            )
            if semantic_source_ids and source_id not in semantic_source_ids:
                aliases = tuple(
                    candidate_id
                    for candidate_id in semantic_source_ids
                    if (
                        (candidate := self.manager.deployment.sources.get(candidate_id))
                        is not None
                        and candidate.node_id == source_id
                    )
                )
                if len(aliases) == 1:
                    source_id = aliases[0]
            if semantic_source_ids and source_id not in semantic_source_ids:
                continue
            source = self.manager.deployment.sources.get(source_id)
            node_id = (
                source.node_id
                if source is not None
                else str(observation.get("node_id") or source_id)
            )
            if (
                semantic_node_ids
                and not (
                    _node_id_aliases(node_id) & set(semantic_node_ids)
                )
            ):
                continue
            bindings = dict(observation.get("bindings") or {})
            identity = "|".join(
                sorted(
                    f"{key}={value}"
                    for key, value in bindings.items()
                    if str(key) in pending.binding_variables
                )
            )
            seed_key = (
                str(observation["occurrence_id"]),
                source_id,
                str(bindings.get("reference") or ""),
                identity,
            )
            if pending.observed_seed_keys is None:
                pending.observed_seed_keys = set()
            if seed_key in pending.observed_seed_keys:
                continue
            pending.observed_seed_keys.add(seed_key)
            partition = str(bindings.get("reference") or source_id)
            if pending.request.seed_admission_strategy == "reference_diverse":
                # Tracker fragmentation can emit many short-lived identities
                # from one camera before another view is observed.  Repeated
                # visit requests need coverage diversity, not the first N
                # local tracks from a single source.  Use the provider's
                # concrete reference when available and fall back to source.
                if pending.admitted_seed_partitions is None:
                    pending.admitted_seed_partitions = set()
                if partition in pending.admitted_seed_partitions:
                    if pending.reported_duplicate_partitions is None:
                        pending.reported_duplicate_partitions = set()
                    if partition not in pending.reported_duplicate_partitions:
                        pending.reported_duplicate_partitions.add(partition)
                        responses.append(
                            LiveComplexEventResponse(
                                request_message_id=pending.request.message_id,
                                request_id=request_id,
                                accepted=True,
                                status="ADMITTED",
                                seed_occurrence_id=str(observation["occurrence_id"]),
                                seed_action="DUPLICATE_IGNORED",
                                active_seed_hypothesis_count=pending.admitted_seed_count,
                                reason=(
                                    "new seed hypothesis suppressed because this "
                                    "concrete reference already has an active "
                                    "hypothesis; the observation remains eligible "
                                    "for active deployed graph demands"
                                ),
                            )
                        )
                    continue
            elif pending.request.seed_admission_strategy == "reference_bounded":
                if pending.admitted_seed_partition_counts is None:
                    pending.admitted_seed_partition_counts = {}
                candidates = pending.admitted_seed_candidates or []
                # Candidate bookkeeping is maintained outside SemanticRuntime
                # because it also controls fair per-camera seed admission.
                # Results can complete or invalidate a hypothesis between seed
                # observations, however, so the bookkeeping must not treat
                # those terminal candidates as occupied slots.  A stale slot
                # both suppresses later tracks from the same camera and makes
                # late provider results target hypotheses that no longer exist.
                # Reconcile from the authoritative runtime before every bounded
                # admission decision.
                execution = self.manager._executions.get(request_id)
                if execution is not None:
                    runtime = execution.request_state(request_id).runtime
                    active_candidates = []
                    active_partition_counts: dict[str, int] = {}
                    for candidate in candidates:
                        if (
                            runtime.get_hypothesis(candidate.hypothesis_id)
                            .lifecycle.value
                            != "ACTIVE"
                        ):
                            continue
                        active_candidates.append(candidate)
                        active_partition_counts[candidate.partition] = (
                            active_partition_counts.get(candidate.partition, 0)
                            + 1
                        )
                    candidates = active_candidates
                    pending.admitted_seed_candidates = candidates
                    pending.admitted_seed_partition_counts = (
                        active_partition_counts
                    )
                    pending.admitted_seed_count = len(candidates)
                # A provider may finalize the same camera-local track more
                # than once as its interval grows.  Keep that observation
                # available to active graph demands, but do not consume a new
                # rolling seed slot for the same concrete identity.
                duplicate_candidate = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.partition == partition
                        and candidate.identity == identity
                        and self.manager._executions[request_id]
                        .request_state(request_id)
                        .runtime.get_hypothesis(candidate.hypothesis_id)
                        .lifecycle.value
                        == "ACTIVE"
                    ),
                    None,
                )
                if duplicate_candidate is not None:
                    responses.append(
                        LiveComplexEventResponse(
                            request_message_id=pending.request.message_id,
                            request_id=request_id,
                            accepted=True,
                            status="ADMITTED",
                            seed_occurrence_id=str(observation["occurrence_id"]),
                            seed_action="DUPLICATE_IGNORED",
                            active_seed_hypothesis_count=pending.admitted_seed_count,
                            reason=(
                                "camera-local seed identity already has an active "
                                "hypothesis; observation remains eligible for its "
                                "deployed successor demand"
                            ),
                        )
                    )
                    continue
                partition_count = pending.admitted_seed_partition_counts.get(
                    partition, 0
                )
                reference_count = max(
                    1, len(pending.request.allowed_seed_node_ids)
                )
                per_reference_limit = max(
                    1,
                    (
                        pending.request.max_seed_hypotheses
                        + reference_count
                        - 1
                    )
                    // reference_count,
                )
                if partition_count >= per_reference_limit:
                    candidates = pending.admitted_seed_candidates or []
                    evicted = next(
                        (
                            candidate
                            for candidate in candidates
                            if candidate.partition == partition
                            and self.manager.evict_unprogressed_seed(
                                request_id,
                                candidate.hypothesis_id,
                                seed_event_time=candidate.seed_event_time,
                                minimum_progress_gap_ms=int(
                                    pending.request.parameters.get(
                                        "minimum_return_gap_ms", 30_000
                                    )
                                ),
                            )
                        ),
                        None,
                    )
                    if evicted is None:
                        continue
                    candidates.remove(evicted)
                    pending.admitted_seed_candidates = candidates
                    pending.admitted_seed_count -= 1
                    pending.admitted_seed_partition_counts[partition] -= 1
                if (
                    pending.admitted_seed_count
                    >= pending.request.max_seed_hypotheses
                ):
                    candidates = pending.admitted_seed_candidates or []
                    # A global full pool must not defeat the per-camera
                    # reservation. If this camera is under its fair share,
                    # reclaim the oldest slot from a more represented view,
                    # even when that noisy candidate saw unrelated progress.
                    reclaim_partition = partition_count < per_reference_limit
                    evicted = next(
                        (
                            candidate
                            for candidate in candidates
                            if (
                                not reclaim_partition
                                or pending.admitted_seed_partition_counts.get(
                                    candidate.partition, 0
                                ) > partition_count
                            )
                            if self.manager.evict_unprogressed_seed(
                                request_id,
                                candidate.hypothesis_id,
                                seed_event_time=candidate.seed_event_time,
                                minimum_progress_gap_ms=int(
                                    pending.request.parameters.get(
                                        "minimum_return_gap_ms", 30_000
                                    )
                                ),
                                force=reclaim_partition,
                            )
                        ),
                        None,
                    )
                    if evicted is None:
                        continue
                    candidates.remove(evicted)
                    pending.admitted_seed_candidates = candidates
                    pending.admitted_seed_count -= 1
                    pending.admitted_seed_partition_counts[evicted.partition] -= 1
            seeded = pending.request.model_copy(
                update={
                    "seed_graph_node_key": None,
                    "seed": ObservedSeed(
                        graph_node_key=pending.request.seed_graph_node_key or "",
                        occurrence_id=str(observation["occurrence_id"]),
                        source_id=source_id,
                        node_id=node_id,
                        event_time_interval=observed_interval,
                        introduced_bindings={
                            pending.binding_variables[str(key)]: str(value)
                            for key, value in bindings.items()
                            if str(key) in pending.binding_variables
                        },
                        confidence=float(observation.get("confidence") or 1.0),
                    ),
                }
            )
            # A single-seed request has already captured the immutable seed
            # observation. Release its provisional acquisition leases before
            # authoritative admission so they cannot make the successor
            # frontier appear capacity-infeasible. This applies to authored
            # B0/B1 as well: their chain/placement contract is frozen, but a
            # structurally unbound successor (for example retrospective
            # recovery after AUDIO_EVENT) cannot attach to the seed provider.
            # Keeping the old lease until after ``admit`` caused B1 to record
            # the right successor plan while the scheduler admitted zero of
            # its providers from the shared desktop resource pool.
            if (
                not self.manager.has_request(request_id)
                and pending.admitted_seed_count == 0
                and pending.request.max_seed_hypotheses == 1
            ):
                self.manager.release_seed_acquisition(
                    request_id,
                    pending.acquisition_demand_ids or set(),
                    reason="seed observed; hand off to authoritative frontier",
                )
            admission = (
                self.manager.admit_additional_seed(seeded)
                if self.manager.has_request(request_id)
                else self.manager.admit(seeded)
            )
            responses.append(admission.response)
            if admission.response.accepted:
                if (
                    pending.request.baseline_id
                    in B1_BASELINES
                    and pending.static_vehicle_history
                    and not pending.static_vehicle_history_injected
                ):
                    # Mark first so a terminal progression/cancellation racing
                    # another queued audio observation cannot replay the same
                    # compact history into a retired request.
                    pending.static_vehicle_history_injected = True
                    self.manager.inject_b1_static_vehicle_history(
                        request_id,
                        pending.static_vehicle_history,
                    )
                pending.admitted_seed_count += 1
                if pending.request.seed_admission_strategy == "reference_diverse":
                    assert pending.admitted_seed_partitions is not None
                    pending.admitted_seed_partitions.add(partition)
                elif pending.request.seed_admission_strategy == "reference_bounded":
                    assert pending.admitted_seed_partition_counts is not None
                    pending.admitted_seed_partition_counts[partition] = (
                        pending.admitted_seed_partition_counts.get(partition, 0) + 1
                    )
                    if pending.admitted_seed_candidates is None:
                        pending.admitted_seed_candidates = []
                    state = self.manager._executions[request_id].request_state(
                        request_id
                    )
                    for hypothesis_id in admission.response.hypothesis_ids:
                        hypothesis = state.runtime.get_hypothesis(
                            UUID(hypothesis_id)
                        )
                        pending.admitted_seed_candidates.append(
                            _AdmittedSeedCandidate(
                                hypothesis_id=hypothesis.hypothesis_id,
                                seed_event_time=observed_interval,
                                partition=partition,
                                identity=identity,
                            )
                        )
            if (
                not admission.response.accepted
                and not self.manager.has_request(request_id)
            ) or (
                pending.admitted_seed_count
                >= pending.request.max_seed_hypotheses
                and pending.request.seed_admission_strategy
                != "reference_bounded"
            ):
                self.manager.release_seed_acquisition(
                    request_id,
                    pending.acquisition_demand_ids or set(),
                    reason="multi-seed candidate pool complete",
                )
                self._pending.pop(request_id, None)
        return tuple(responses)

    @staticmethod
    def _parameters_match(
        expected: dict[str, JSONValue],
        observation: dict[str, object],
    ) -> bool:
        label = expected.get("label")
        if label is not None and str(observation.get("label")) != str(label):
            return False
        minimum_confidence = expected.get("minimum_confidence")
        if minimum_confidence is not None and float(
            observation.get("confidence") or 0.0
        ) < float(minimum_confidence):
            return False
        return True

    @staticmethod
    def _normalize_observation(
        topic: str,
        payload: bytes | str | dict[str, object],
    ) -> dict[str, object] | None:
        try:
            document = (
                payload
                if isinstance(payload, dict)
                else json.loads(
                    payload.decode("utf-8")
                    if isinstance(payload, bytes)
                    else payload
                )
            )
        except Exception:
            return None
        if not isinstance(document, dict):
            return None
        schema = document.get("schema_version")
        if schema in {
            "vehicle_predicate_observation.v1",
            "interaction_predicate_observation.v1",
        }:
            if document.get("truth") is not True:
                return None
            source_ids = document.get("source_ids") or ()
            if not source_ids:
                return None
            return {
                **document,
                "source_id": source_ids[0],
            }
        if schema == "track_set.v1":
            return {
                **document,
                "predicate_id": "__TRACK_SET__",
            }
        if schema == "audio_event_observation.v1":
            source_id = str(document.get("source_id") or "unknown_audio_source")
            # A non-localizing microphone still observes a concrete sensor
            # scene. Bind that scene deterministically so a trigger-directed
            # retrospective demand is compilable; do not pretend this is
            # building-level or metric localization.
            bindings = {
                "location": str(
                    document.get("localized_zone_id")
                    or f"sensor_scene:{source_id}"
                )
            }
            return {
                **document,
                "predicate_id": "AUDIO_EVENT",
                "bindings": bindings,
            }
        if schema == "fable.predicate_result.v1":
            # Durable provider results are already request/demand scoped and
            # pass through the orchestrator's processed ledger.  Feeding them
            # to a pending seed watch closes the QoS-0 raw-topic race without
            # admitting stale evidence from another request or replay.
            if str(document.get("truth")) not in {"TRUE", "TruthValue.TRUE"}:
                return None
            predicate = document.get("semantic_predicate") or {}
            provenance = document.get("provenance") or {}
            sources = provenance.get("source_ids") or ()
            bindings = {}
            delta = document.get("binding_delta") or {}
            bindings.update(delta.get("introduced") or {})
            bindings.update(delta.get("validated") or {})
            parameters = predicate.get("parameters") or {}
            return {
                "predicate_id": predicate.get("predicate_id"),
                "label": parameters.get("label"),
                "confidence": document.get("confidence"),
                "event_time_interval": document.get("event_time_interval"),
                "occurrence_id": document.get("occurrence_id"),
                "source_id": sources[0] if sources else provenance.get("node_id"),
                "node_id": provenance.get("node_id"),
                "bindings": bindings,
                "_trusted_request_id": document.get("request_id"),
            }
        return None
