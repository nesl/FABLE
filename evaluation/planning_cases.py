"""Compile catalog event variants into family-specific baseline planning cases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from fable.common.schemas import PredicateDemand
from fable.common.time import EventTimeInterval
from fable.planning import (
    DemandCompileContext,
    DemandCompiler,
    PhysicalAlternativeGraphBuilder,
    default_predicate_registry,
)
from fable.planning.artifact_catalog import ArtifactCatalog
from fable.planning.deployment import DeploymentGraph
from fable.planning.provider_registry import ProviderRegistry
from fable.distributed.config import ProviderRuntimeResolver
from fable.distributed.models import RuntimeMode
from fable.planning.models import PhysicalAlternativeGraph
from fable.planning.alternative_graph import AlternativeBuildConfig
from fable.semantic import (
    EventRequestCompiler,
    ScriptedResultSpec,
    SemanticRuntime,
    SemanticRuntimeConfig,
    predicate_result_from_spec,
    seed_result_from_spec,
)

from evaluation.baselines.models import BaselinePlanningCase
from evaluation.task_universe import TaskDemandUniverseBuilder


@dataclass(frozen=True)
class EventFamilyPlanningTemplate:
    family_id: str
    seed_node_key: str
    preferred_branch_node_keys: tuple[str, ...] = ()
    request_parameters: dict[str, object] | None = None
    max_seed_hypotheses: int = 1
    seed_admission_strategy: str = "first_distinct"


VARIANT_TEMPLATES = {
    "Cross-sensor robbery": EventFamilyPlanningTemplate(
        "robbery",
        "gunshot_branch",
        ("prior_entry",),
    ),
    "Pass-follow-clear convoy": EventFamilyPlanningTemplate(
        "convoy",
        "leader_passes",
        request_parameters={"evaluation_profile": "sequential_passes"},
        max_seed_hypotheses=4,
    ),
    "Robbery with alarm": EventFamilyPlanningTemplate(
        "robbery",
        "alarm_branch",
        request_parameters={"evaluation_profile": "alarm_departure"},
    ),
    "Route convoy": EventFamilyPlanningTemplate(
        "convoy",
        "leader_passes",
        request_parameters={"evaluation_profile": "sequential_passes"},
        max_seed_hypotheses=4,
    ),
    "Talking/rendezvous": EventFamilyPlanningTemplate(
        "rendezvous",
        "arrival",
        request_parameters={"evaluation_profile": "full_talking"},
    ),
    "Vehicle rendezvous": EventFamilyPlanningTemplate(
        "vehicle_convergence",
        "seed_passes",
        max_seed_hypotheses=4,
    ),
    "Three-visit stalking": EventFamilyPlanningTemplate(
        "repeated_visit",
        "first_visit",
        request_parameters={
            "visit_count": 3,
            # Uncalibrated PASSES is finalized after tracker disappearance;
            # the recorded inter-visit gap is therefore shorter than the
            # physical absence interval. Ten seconds rejects fragmentation
            # while retaining the observed repeated visits in this corpus.
            "minimum_return_gap_ms": 10_000,
            "evaluation_profile": "uncalibrated_passes",
            "identity_confirmation": True,
        },
        # Keep a bounded set of camera-local identity candidates. The useful
        # recurring vehicle may enter after an unrelated object has already
        # consumed that camera's first slot, so reference-only diversity is
        # insufficient. Successor graph bindings still keep each candidate on
        # the camera where it was seeded.
        max_seed_hypotheses=20,
        seed_admission_strategy="reference_bounded",
    ),
    "Two-vehicle chase": EventFamilyPlanningTemplate(
        "two_vehicle_chase",
        "leader_passes",
        request_parameters={"evaluation_profile": "sequential_passes"},
        max_seed_hypotheses=4,
    ),
    "Two-visit stalking": EventFamilyPlanningTemplate(
        "repeated_visit",
        "first_visit",
        request_parameters={
            "visit_count": 2,
            "evaluation_profile": "uncalibrated_passes",
            "identity_confirmation": True,
        },
        max_seed_hypotheses=20,
        seed_admission_strategy="reference_bounded",
    ),
    "Vehicle convergence": EventFamilyPlanningTemplate(
        "vehicle_convergence",
        "seed_passes",
        request_parameters={"departure_policy": "scene_departures"},
        # PASSES is finalized when a track leaves, so the first result across
        # several cameras is not necessarily the camera containing the later
        # convergence. Retain a bounded camera-fair pool instead of cancelling
        # discovery after that timing-dependent first result.
        max_seed_hypotheses=20,
        seed_admission_strategy="reference_bounded",
    ),
}


class PlanningCaseCompileError(ValueError):
    pass


def scope_demands_to_nodes(
    demands: tuple[PredicateDemand, ...],
    allowed_node_ids: tuple[str, ...],
) -> tuple[PredicateDemand, ...]:
    """Apply a live run's execution scope before alternative enumeration.

    Filtering a capped physical graph after construction is insufficient:
    out-of-scope placements can consume the enumeration budget and crowd out
    the one valid node-local realization.
    """

    allowed = set(allowed_node_ids)
    if not allowed:
        return demands
    scoped = []
    for demand in demands:
        existing = set(demand.hard_constraints.allowed_node_ids)
        effective = allowed if not existing else allowed & existing
        scoped.append(
            demand.model_copy(
                update={
                    "hard_constraints": demand.hard_constraints.model_copy(
                        update={"allowed_node_ids": tuple(sorted(effective))}
                    ),
                    # The sharing key is derived from the semantic demand fields,
                    # including hard constraints.  Clear the old derived value so
                    # the next validated boundary recomputes it for this scope.
                    "sharing_key": None,
                }
            )
        )
    return tuple(scoped)


def compile_evaluation_planning_case(
    *,
    variant: str,
    run_id: str,
    trace_id: str,
    request_id: str,
    now,
    provider_registry: ProviderRegistry,
    artifact_catalog: ArtifactCatalog,
    deployment: DeploymentGraph,
    runtime_resolver: ProviderRuntimeResolver | None = None,
    allow_reference_runtimes: bool = False,
    frontier_index: int = 0,
) -> BaselinePlanningCase:
    if frontier_index < 0:
        raise PlanningCaseCompileError("frontier_index must be non-negative")
    try:
        template = VARIANT_TEMPLATES[variant]
    except KeyError as exc:
        raise PlanningCaseCompileError(f"unsupported catalog variant: {variant}") from exc
    semantic = EventRequestCompiler().compile(
        {
            "family_id": template.family_id,
            "parameters": template.request_parameters or {},
        }
    ).graph
    runtime = SemanticRuntime(
        semantic,
        config=SemanticRuntimeConfig(request_id=request_id),
    )
    transition = runtime.seed(
        seed_result_from_spec(
            runtime,
            _result_spec(runtime, template.seed_node_key, now, index=0),
        )
    )
    if not transition.hypothesis_ids:
        raise PlanningCaseCompileError(
            f"seed {template.seed_node_key} did not create a hypothesis"
        )
    hypothesis_id = transition.hypothesis_ids[0]
    seeded_hypothesis = runtime.get_hypothesis(hypothesis_id).model_copy(deep=True)
    demand_compiler = DemandCompiler(
        predicate_registry=default_predicate_registry(),
        deployment=deployment,
    )
    graph_builder = PhysicalAlternativeGraphBuilder(
        provider_registry=provider_registry,
        artifact_catalog=artifact_catalog,
        deployment=deployment,
        # Whole-event definitions now include identity-preserving completion
        # stages. Keep offline case construction bounded per demand so repeated
        # EXITS stages do not multiply equivalent synthetic placements before
        # the bounded label planner can prune them.
        config=AlternativeBuildConfig(
            max_total_alternatives=32,
            max_alternatives_per_chain=8,
        ),
    )

    all_demands: list[PredicateDemand] = []
    seen_node_keys: set[str] = set()
    frontier_demands: tuple[PredicateDemand, ...] | None = None
    for index in range(32):
        hypothesis = runtime.get_hypothesis(hypothesis_id)
        frontier = runtime.get_frontier(hypothesis_id)
        if frontier is None:
            break
        context = DemandCompileContext(
            eligible_source_ids_by_node={
                node_id: tuple(sorted(deployment.sources))
                for node_id in frontier.snapshot.enabled_node_ids
            }
        )
        demands = demand_compiler.compile_frontier(
            graph=runtime.graph,
            hypothesis=hypothesis,
            frontier=frontier,
            context=context,
        )
        preferred_enabled = set(template.preferred_branch_node_keys) & {
            runtime.graph.nodes_by_id[item.graph_node_id].authored_key
            for item in demands
        }
        if preferred_enabled:
            demands = tuple(
                item
                for item in demands
                if runtime.graph.nodes_by_id[item.graph_node_id].authored_key
                in preferred_enabled
            )
        if index == frontier_index:
            frontier_demands = demands
        for demand in demands:
            node_key = runtime.graph.nodes_by_id[demand.graph_node_id].authored_key
            if node_key not in seen_node_keys:
                seen_node_keys.add(node_key)
                all_demands.append(demand)

        preferred_node_ids = [
            node_id
            for node_id in frontier.snapshot.enabled_node_ids
            if runtime.graph.nodes_by_id[node_id].authored_key
            in set(template.preferred_branch_node_keys)
        ]
        node_id = (
            preferred_node_ids[0]
            if preferred_node_ids
            else frontier.snapshot.enabled_node_ids[0]
        )
        node_key = runtime.graph.nodes_by_id[node_id].authored_key
        result = predicate_result_from_spec(
            runtime,
            hypothesis_id,
            _result_spec(runtime, node_key, now, index=index + 1),
        )
        transition = runtime.apply(result)
        if transition.hypothesis_ids:
            hypothesis_id = transition.hypothesis_ids[0]
    else:
        raise PlanningCaseCompileError("semantic traversal exceeded 32 frontiers")

    universe_context = DemandCompileContext(
        eligible_source_ids_by_node={
            node_id: tuple(sorted(deployment.sources))
            for node_id in runtime.graph.executable_predicate_nodes()
        }
    )
    all_demands = list(
        TaskDemandUniverseBuilder(demand_compiler).build(
            graph=runtime.graph,
            hypothesis=seeded_hypothesis,
            context=universe_context,
        )
    )
    if frontier_demands is None:
        raise PlanningCaseCompileError(
            f"{variant} has no executable semantic frontier at index {frontier_index}"
        )
    if not frontier_demands or not all_demands:
        raise PlanningCaseCompileError(
            f"{variant} produced no post-seed executable demands"
        )
    frontier_graph = graph_builder.build(frontier_demands, now=now)
    whole_graph = graph_builder.build(tuple(all_demands), now=now)
    if runtime_resolver is not None:
        frontier_graph = executable_runtime_graph(
            frontier_graph,
            runtime_resolver=runtime_resolver,
            allow_reference_runtimes=allow_reference_runtimes,
        )
        whole_graph = executable_runtime_graph(
            whole_graph,
            runtime_resolver=runtime_resolver,
            allow_reference_runtimes=allow_reference_runtimes,
        )
    missing = {
        demand.demand_id for demand in all_demands
    } - {alternative.demand_id for alternative in whole_graph.alternatives}
    if missing:
        predicates = sorted(
            demand.semantic_predicate.predicate_id
            for demand in all_demands
            if demand.demand_id in missing
        )
        pruning = sorted({
            f"{item.code}: {item.reason}"
            for item in whole_graph.pruned
            if item.demand_id in missing
        })
        raise PlanningCaseCompileError(
            "no physical alternatives for whole-event predicates: "
            + ", ".join(predicates)
            + ("; pruning: " + " | ".join(pruning) if pruning else "")
        )
    return BaselinePlanningCase(
        run_id=run_id,
        trace_id=trace_id,
        request_id=request_id,
        event_family=_normalized_event_family(variant),
        placement_id=variant,
        frontier_demands=frontier_demands,
        all_task_demands=tuple(all_demands),
        frontier_graph=frontier_graph,
        whole_event_graph=whole_graph,
        now=now,
    )


def executable_runtime_graph(
    graph: PhysicalAlternativeGraph,
    *,
    runtime_resolver: ProviderRuntimeResolver,
    allow_reference_runtimes: bool = False,
    allowed_node_ids: tuple[str, ...] = (),
) -> PhysicalAlternativeGraph:
    """Remove planner alternatives that the live node agents cannot execute.

    Reference runtimes are useful deterministic fixtures, but their configured
    truth values make them invalid accuracy-evaluation providers.  They are
    excluded by default at the planning boundary rather than rejected after
    admission by the dispatcher.
    """

    executable = []
    allowed_nodes = set(allowed_node_ids)
    for alternative in graph.alternatives:
        valid = True
        for placement in alternative.step_placements:
            if allowed_nodes and placement.node_id not in allowed_nodes:
                valid = False
                break
            if not runtime_resolver.has(placement.node_id, placement.provider_id):
                valid = False
                break
            runtime = runtime_resolver.resolve(
                node_id=placement.node_id,
                provider_id=placement.provider_id,
            )
            if not allow_reference_runtimes and runtime.mode == RuntimeMode.REFERENCE:
                valid = False
                break
        if valid:
            executable.append(alternative)
    return graph.model_copy(update={"alternatives": tuple(executable)})


def _result_spec(runtime, node_key: str, now, *, index: int) -> ScriptedResultSpec:
    node = runtime.graph.nodes_by_key[node_key]
    hypothesis = runtime.active_hypotheses[0] if runtime.active_hypotheses else None
    graph_roles = {item.role_name for item in runtime.graph.graph.roles}
    introduced: dict[str, str] = {}
    validated: dict[str, str] = {}
    for role in node.predicate.roles:
        if role.variable not in graph_roles:
            continue
        existing = (
            hypothesis.role_bindings.get(role.variable)
            if hypothesis is not None
            else None
        )
        if existing is None:
            introduced[role.variable] = f"{role.variable}_entity"
        else:
            validated[role.variable] = existing.canonical_entity_id
    source_id = (
        "microphone_store"
        if node.predicate.predicate_id == "AUDIO_EVENT"
        else "camera_mobile"
    )
    return ScriptedResultSpec(
        node_key=node_key,
        source_id=source_id,
        event_time_interval=EventTimeInterval(
            # Keep synthetic planning traversal beyond the authored 30-second
            # repeated-visit guard while remaining inside the shortest
            # 60-second evaluation windows. Four-second observations also
            # satisfy authored dwell guards during structural traversal.
            start=now + timedelta(seconds=index * 31),
            end=now + timedelta(seconds=index * 31 + 4),
        ),
        introduced=introduced,
        validated=validated,
    )


def _normalized_event_family(variant: str) -> str:
    return {
        "Cross-sensor robbery": "robbery_with_alarm",
        "Pass-follow-clear convoy": "route_convoy",
        "Robbery with alarm": "robbery_with_alarm",
        "Route convoy": "route_convoy",
        "Talking/rendezvous": "rendezvous",
        "Vehicle rendezvous": "rendezvous",
        "Three-visit stalking": "repeated_visit_stalking",
        "Two-vehicle chase": "two_vehicle_chase",
        "Two-visit stalking": "repeated_visit_stalking",
        "Vehicle convergence": "vehicle_convergence",
    }[variant]
