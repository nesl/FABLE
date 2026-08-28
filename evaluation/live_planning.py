"""Live adapters that connect controlled evaluation baselines to FableController.

The baseline policies remain evaluation code.  These adapters implement the
core ``ControllerPlanningPolicy`` seam and map task-level baseline choices onto
the currently active semantic frontier without bypassing FABLE execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from fable.common.ids import uuid7
from fable.common.schemas import PredicateDemand
from fable.orchestration.planning_policy import (
    ControllerPlanningContext,
    ControllerPlanningDecision,
)
from fable.planning import DemandCompiler, PhysicalAlternativeGraphBuilder, default_predicate_registry
from fable.planning.beam_search import BoundedLabelPlanner
from fable.planning.models import PhysicalAlternative, PhysicalAlternativeGraph

from evaluation.baselines.models import BaselinePlanningCase
from evaluation.baselines.policies import (
    FablePolicy,
    GreedyFrontierPolicy,
    HandwrittenStaticPolicy,
    TaskResourceAdaptivePolicy,
)
from evaluation.baselines.static_registry import StaticPipelineRegistry
from evaluation.schemas import BaselineId
from evaluation.task_universe import TaskDemandUniverseBuilder


_FAMILY_ALIASES = {
    "convoy": "route_convoy",
    "robbery": "robbery_with_alarm",
    "repeated_visit": "repeated_visit_stalking",
}


@dataclass(frozen=True)
class _AlternativeTemplate:
    graph_node_id: str
    chain_id: str
    providers_and_nodes: tuple[tuple[str, str], ...]
    source_ids: tuple[str, ...]


@dataclass
class _RequestTaskState:
    all_task_demands: tuple[PredicateDemand, ...]
    frozen_templates: tuple[_AlternativeTemplate, ...] | None = None
    # B2 fixes the physical realization the first time each semantic graph
    # node becomes active, while still following later semantic frontiers.
    # Keying by authored graph node (rather than demand UUID) lets a grounded
    # successor reuse that realization without adapting to resource changes.
    frontier_templates: dict[str, tuple[_AlternativeTemplate, ...]] | None = None


class LiveBaselinePlanningPolicy:
    """Live evaluation-policy adapter for the redesigned controller.

    B1 selects a task-level physical template once and keeps it frozen. B3 keeps
    the same task-level semantic work set but recomputes the physical template
    whenever ``resource_epoch`` changes. In both cases only the subset matching
    the current frontier is activated, so the semantic runtime remains the sole
    authority over event progress.
    """

    def __init__(
        self,
        baseline_id: BaselineId,
        *,
        static_registry_path: str = "evaluation/manifests/baselines/static_pipelines.yaml",
    ) -> None:
        if baseline_id not in {
            BaselineId.B0_PRODUCE_ALL,
            BaselineId.B1_HANDWRITTEN_STATIC,
            BaselineId.B2_FRONTIER_FIXED_REALIZATION,
            BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
            BaselineId.B4_GREEDY_FRONTIER,
        }:
            raise ValueError("unsupported redesigned live baseline adapter")
        self.baseline_id = baseline_id
        self.policy_id = baseline_id.value
        self.static_registry_path = static_registry_path
        self._requests: dict[str, _RequestTaskState] = {}
        self._b3_templates: dict[
            tuple[str, str, str, int], tuple[_AlternativeTemplate, ...]
        ] = {}

    def constrain_frontier_demands(
        self,
        *,
        trace_id: str,
        placement_id: str,
        demands: tuple[PredicateDemand, ...],
    ) -> tuple[PredicateDemand, ...]:
        """Apply B1's immutable physical contract before enumeration.

        Candidate enumeration is intentionally bounded. Filtering a static
        authored placement only after enumeration allows unrelated cameras to
        consume that budget and starve the one valid B1 realization. This hook
        is B1-only; B3 and FABLE retain their ordinary candidate universes.
        """

        if self.baseline_id not in {
            BaselineId.B0_PRODUCE_ALL,
            BaselineId.B1_HANDWRITTEN_STATIC,
        }:
            return demands
        placement = StaticPipelineRegistry.load(
            self.static_registry_path
        ).get_placement(placement_id, trace_id=trace_id)
        if placement is None:
            return demands
        # B1 freezes exact sources/nodes. B0 deliberately takes the same
        # CE-specific provider/chain contract but broadcasts it, so applying
        # B1's physical placement here would collapse B0 back into B1.
        exact_placement = self.baseline_id == BaselineId.B1_HANDWRITTEN_STATIC
        allowed_sources = set(placement.allowed_source_ids) if exact_placement else set()
        allowed_nodes = set(placement.allowed_node_ids) if exact_placement else set()
        allowed_branches = set(placement.allowed_branch_ids)
        constrained: list[PredicateDemand] = []
        for demand in demands:
            branch_label = str(
                demand.semantic_predicate.parameters.get("label") or ""
            )
            if (
                allowed_branches
                and branch_label
                and branch_label not in allowed_branches
            ):
                continue
            eligible_sources = tuple(
                source_id
                for source_id in demand.eligible_source_ids
                if not allowed_sources or source_id in allowed_sources
            )
            existing_nodes = set(demand.hard_constraints.allowed_node_ids)
            fixed_nodes = (
                existing_nodes.intersection(allowed_nodes)
                if existing_nodes and allowed_nodes
                else allowed_nodes or existing_nodes
            )
            payload = demand.model_dump(mode="python")
            payload.update(
                {
                    "eligible_source_ids": eligible_sources,
                    "source_preferences": tuple(
                        preference
                        for preference in demand.source_preferences
                        if preference.source_id in eligible_sources
                    ),
                    "hard_constraints": demand.hard_constraints.model_copy(
                        update={"allowed_node_ids": tuple(sorted(fixed_nodes))}
                    ),
                    # Source/node constraints are part of the semantic sharing
                    # identity. Revalidate so PredicateDemand derives a new key
                    # rather than carrying the unconstrained demand's key.
                    "sharing_key": None,
                }
            )
            constrained.append(PredicateDemand.model_validate(payload))
        return tuple(constrained)

    def expand_admission_demands(
        self,
        *,
        semantic_graph,
        hypothesis,
        demand_context: DemandCompileContext,
        deployment,
        trace_id: str,
        placement_id: str,
        active_demands: tuple[PredicateDemand, ...],
    ) -> tuple[PredicateDemand, ...] | None:
        """Materialize B1's fixed producer pipeline at request admission.

        The redesigned controller normally compiles only the active semantic
        frontier. That is correct for FABLE, but it silently changed B1 from a
        whole-event authored pipeline into a frontier-activated policy. Keep
        identity comparison late-bound because it requires two concrete
        endpoints; start every other structurally executable B1 provider now.
        """

        if self.baseline_id not in {
            BaselineId.B0_PRODUCE_ALL,
            BaselineId.B1_HANDWRITTEN_STATIC,
        }:
            return None
        compiler = DemandCompiler(
            predicate_registry=default_predicate_registry(),
            deployment=deployment,
        )
        demands = TaskDemandUniverseBuilder(compiler).build(
            graph=semantic_graph,
            hypothesis=hypothesis,
            context=demand_context,
            skip_uncompilable=True,
        )
        active_node_ids = {demand.graph_node_id for demand in active_demands}
        future_demands = tuple(
            demand
            for demand in demands
            if demand.graph_node_id not in active_node_ids
            if not (
                demand.semantic_predicate.predicate_id == "SAME_ENTITY"
                and any(
                    value.startswith("__structural_unbound__:")
                    for value in demand.bound_roles.values()
                )
            )
        )
        return self.constrain_frontier_demands(
            trace_id=trace_id,
            placement_id=placement_id,
            # Preserve the authoritative frontier/demand IDs exactly. Only
            # future fixed-provider watches use structural demand envelopes.
            demands=(*active_demands, *future_demands),
        )

    def select(self, context: ControllerPlanningContext) -> ControllerPlanningDecision:
        if self.baseline_id == BaselineId.B0_PRODUCE_ALL:
            return ControllerPlanningDecision(
                policy_id=self.policy_id,
                allowed_alternative_ids=tuple(
                    item.alternative_id for item in context.frontier_graph.alternatives
                ),
                reason=(
                    "B0 executes the CE-specific authored provider-chain set on "
                    "every eligible sensor node and freezes it for the request."
                ),
                frozen=True,
            )
        if self.baseline_id == BaselineId.B1_HANDWRITTEN_STATIC:
            # The physical contract is frozen in the trace registry, while
            # role binding is owned by the semantic runtime.  Instantiating
            # every future predicate before its predecessor binds identities
            # is invalid (notably SAME_ENTITY.left).  Apply the immutable
            # chain/node/source allowlist to each *grounded* frontier instead.
            baseline = HandwrittenStaticPolicy(
                StaticPipelineRegistry.load(self.static_registry_path)
            )
            decision = baseline.plan(
                _case(context, context.frontier_demands, context.frontier_graph)
            )
            return ControllerPlanningDecision(
                policy_id=self.policy_id,
                allowed_alternative_ids=decision.selected_alternative_ids,
                reason=(
                    "B1 exact trace-authored physical contract applied to the "
                    "current grounded frontier; placement remains frozen. "
                    + decision.reason
                ),
                frozen=True,
            )
        elif self.baseline_id == BaselineId.B4_GREEDY_FRONTIER:
            decision = GreedyFrontierPolicy().plan(
                _case(context, context.frontier_demands, context.frontier_graph)
            )
            return ControllerPlanningDecision(
                policy_id=self.policy_id,
                allowed_alternative_ids=decision.selected_alternative_ids,
                reason=decision.reason,
                frozen=False,
            )
        elif self.baseline_id == BaselineId.B2_FRONTIER_FIXED_REALIZATION:
            state = self._requests.get(context.request_id)
            if state is None:
                state = _RequestTaskState(
                    all_task_demands=(), frontier_templates={}
                )
                self._requests[context.request_id] = state
            if state.frontier_templates is None:
                state.frontier_templates = {}
            selected: list[str] = []
            missing = []
            for demand in context.frontier_demands:
                templates = state.frontier_templates.get(demand.graph_node_id)
                if templates is None:
                    missing.append(demand)
                    continue
                selected.extend(
                    _map_templates_to_frontier(
                        templates,
                        context.frontier_graph,
                        (demand,),
                    )
                )
            if missing:
                planner = BoundedLabelPlanner(
                    provider_registry=context.provider_registry,
                    artifact_catalog=context.artifact_catalog,
                    deployment=context.deployment,
                )
                decision = FablePolicy(planner).plan(
                    _case(context, tuple(missing), context.frontier_graph)
                )
                templates = _templates_from_decision(
                    decision.selected_alternative_ids,
                    context.frontier_graph,
                    tuple(missing),
                )
                for demand in missing:
                    owned = tuple(
                        item for item in templates
                        if item.graph_node_id == demand.graph_node_id
                    )
                    state.frontier_templates[demand.graph_node_id] = owned
                    selected.extend(
                        _map_templates_to_frontier(
                            owned, context.frontier_graph, (demand,)
                        )
                    )
            return ControllerPlanningDecision(
                policy_id=self.policy_id,
                allowed_alternative_ids=tuple(dict.fromkeys(selected)),
                reason=(
                    "B2 selected each semantic frontier realization once and "
                    "reused the frozen provider/placement/source template."
                ),
                frozen=True,
            )
        else:
            state = self._requests.get(context.request_id)
            if state is None:
                state = _RequestTaskState(
                    all_task_demands=_compile_whole_task_demands(context)
                )
                self._requests[context.request_id] = state
            # B3 remains task-level (it optimizes every predicate rather than
            # FABLE's active frontier), but task demands must be instantiated
            # with the current hypothesis bindings.  Reusing admission-time
            # ungrounded source/identity contracts makes an exact template
            # impossible to map after a camera-local seed has been observed.
            # Hypothesis/version are therefore realization keys, not new
            # semantic information in B3's optimization objective.
            current_task_demands = _compile_whole_task_demands(context)
            key = (
                context.request_id,
                _deployment_signature(context),
                str(context.hypothesis.hypothesis_id),
                context.hypothesis.version,
            )
            templates = self._b3_templates.get(key)
            whole_graph = _whole_task_graph(context, current_task_demands)
            if templates is None:
                planner = BoundedLabelPlanner(
                    provider_registry=context.provider_registry,
                    artifact_catalog=context.artifact_catalog,
                    deployment=context.deployment,
                )
                baseline = TaskResourceAdaptivePolicy(planner)
                case = _case(context, current_task_demands, whole_graph)
                decision = baseline.plan(case)
                templates = _templates_from_decision(
                    decision.selected_alternative_ids,
                    whole_graph,
                    current_task_demands,
                )
                self._b3_templates[key] = templates
            structurally_unrealizable_node_ids = {
                demand.graph_node_id for demand in current_task_demands
            } - {
                demand.graph_node_id
                for demand in current_task_demands
                for alternative in whole_graph.alternatives
                if alternative.demand_id == demand.demand_id
            }
            frozen = False
            reason = (
                "B3 task-level template recomputed for resource epoch "
                f"{context.resource_epoch} and mapped onto current frontier."
            )

        if (
            self.baseline_id == BaselineId.B3_TASK_RESOURCE_ADAPTIVE
            and not context.frontier_graph.alternatives
        ):
            return ControllerPlanningDecision(
                policy_id=self.policy_id,
                allowed_alternative_ids=None,
                reason=(
                    "B3 leaves an abstract checkpoint with no physical "
                    "alternatives unrestricted so the shared seed gateway can "
                    "materialize its concrete provider watches."
                ),
                frozen=False,
            )

        selected = _map_templates_to_frontier(
            templates or (),
            context.frontier_graph,
            context.frontier_demands,
            allow_grounded_fallback_node_ids=(
                structurally_unrealizable_node_ids
                if self.baseline_id == BaselineId.B3_TASK_RESOURCE_ADAPTIVE
                else set()
            ),
        )
        if (
            self.baseline_id == BaselineId.B3_TASK_RESOURCE_ADAPTIVE
            and not selected
            and context.frontier_graph.alternatives
        ):
            # A structural whole-task template can become unmappable when an
            # abstract checkpoint is grounded into a concrete seed/provider
            # watch (for example THREAT_EVENT -> alarm or gunshot).  An empty
            # allowlist means "reject every plan", not "defer selection".
            # Preserve B3's resource-adaptive policy by optimizing the live,
            # grounded frontier with the same planner rather than admitting
            # every candidate or changing FABLE's planning path.
            planner = BoundedLabelPlanner(
                provider_registry=context.provider_registry,
                artifact_catalog=context.artifact_catalog,
                deployment=context.deployment,
            )
            fallback = TaskResourceAdaptivePolicy(planner).plan(
                _case(context, context.frontier_demands, context.frontier_graph)
            )
            selected = fallback.selected_alternative_ids
            reason += (
                " Whole-task templates had no realizable grounded mapping; "
                "B3 re-optimized the authoritative live frontier."
            )
        return ControllerPlanningDecision(
            policy_id=self.policy_id,
            allowed_alternative_ids=selected,
            reason=reason,
            frozen=frozen,
        )


def _deployment_signature(context: ControllerPlanningContext) -> str:
    """Hash material scheduling state, excluding heartbeat sequence churn."""

    deployment = context.deployment
    payload = {
        "nodes": [
            deployment.nodes[key].model_dump(mode="json")
            for key in sorted(deployment.nodes)
        ],
        "sources": [
            deployment.sources[key].model_dump(mode="json")
            for key in sorted(deployment.sources)
        ],
        "links": [
            item.model_dump(mode="json")
            for item in sorted(
                deployment.links,
                key=lambda value: (
                    value.source_node_id,
                    value.target_node_id,
                ),
            )
        ],
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _compile_whole_task_demands(
    context: ControllerPlanningContext,
) -> tuple[PredicateDemand, ...]:
    compiler = DemandCompiler(
        predicate_registry=default_predicate_registry(),
        deployment=context.deployment,
    )
    # B3 reasons over the complete remaining task, including predicates whose
    # consumer-only roles cannot be grounded until an earlier frontier emits
    # an identity. The structural universe builder represents those roles with
    # typed placeholders. Using ordinary frontier compilation here rejected
    # valid SAME_ENTITY stages before the seed watch could even be registered.
    return TaskDemandUniverseBuilder(compiler).build(
        graph=context.semantic_graph,
        hypothesis=context.hypothesis,
        context=context.demand_context,
    )


def _whole_task_graph(
    context: ControllerPlanningContext,
    demands: tuple[PredicateDemand, ...],
) -> PhysicalAlternativeGraph:
    graph = PhysicalAlternativeGraphBuilder(
        provider_registry=context.provider_registry,
        artifact_catalog=context.artifact_catalog,
        deployment=context.deployment,
    ).build(demands)
    allowed = tuple(
        alternative
        for alternative in graph.alternatives
        if all(
            (placement.node_id, placement.provider_id) in context.runtime_provider_keys
            for placement in alternative.step_placements
        )
    )
    return graph.model_copy(update={"alternatives": allowed})


def _case(
    context: ControllerPlanningContext,
    all_task_demands: tuple[PredicateDemand, ...],
    whole_graph: PhysicalAlternativeGraph,
) -> BaselinePlanningCase:
    return BaselinePlanningCase(
        run_id=context.request_id,
        trace_id=context.trace_id,
        request_id=context.request_id,
        event_family=_FAMILY_ALIASES.get(context.family_id, context.family_id),
        placement_id=context.placement_id,
        frontier_demands=context.frontier_demands,
        all_task_demands=all_task_demands,
        frontier_graph=context.frontier_graph,
        whole_event_graph=whole_graph,
        now=context.hypothesis.updated_at,
        resource_epoch=context.resource_epoch,
        semantic_epoch=context.semantic_epoch,
    )


def _templates_from_decision(
    selected_ids: tuple[str, ...],
    graph: PhysicalAlternativeGraph,
    demands: tuple[PredicateDemand, ...],
) -> tuple[_AlternativeTemplate, ...]:
    by_alt = {item.alternative_id: item for item in graph.alternatives}
    by_demand = {item.demand_id: item for item in demands}
    rows: list[_AlternativeTemplate] = []
    for alternative_id in selected_ids:
        alternative = by_alt.get(alternative_id)
        if alternative is None:
            continue
        demand = by_demand.get(alternative.demand_id)
        if demand is None:
            continue
        rows.append(_template(alternative, demand.graph_node_id))
    return tuple(rows)


def _template(alternative: PhysicalAlternative, graph_node_id: str) -> _AlternativeTemplate:
    return _AlternativeTemplate(
        graph_node_id=graph_node_id,
        chain_id=alternative.chain_id,
        providers_and_nodes=tuple(
            (item.provider_id, item.node_id) for item in alternative.step_placements
        ),
        source_ids=tuple(
            sorted(
                item.source_id
                for item in alternative.external_inputs
                if item.source_id is not None
            )
        ),
    )


def _map_templates_to_frontier(
    templates: tuple[_AlternativeTemplate, ...],
    graph: PhysicalAlternativeGraph,
    demands: tuple[PredicateDemand, ...],
    allow_grounded_fallback_node_ids: set[str] | None = None,
) -> tuple[str, ...]:
    allow_grounded_fallback_node_ids = allow_grounded_fallback_node_ids or set()
    by_node: dict[str, list[_AlternativeTemplate]] = {}
    for item in templates:
        by_node.setdefault(item.graph_node_id, []).append(item)
    demand_by_id = {item.demand_id: item for item in demands}
    selected: list[str] = []
    for demand in demands:
        candidates = [item for item in graph.alternatives if item.demand_id == demand.demand_id]
        expected = by_node.get(demand.graph_node_id, [])
        if not expected and demand.graph_node_id in allow_grounded_fallback_node_ids:
            # A structural future-task demand can be intentionally unplaceable
            # before its roles are grounded (the abstract robbery threat seed
            # is one example). If the authoritative live frontier now has
            # concrete alternatives, B3 must admit that grounded work instead
            # of turning a representational limitation into task infeasibility.
            selected.extend(item.alternative_id for item in candidates)
            continue
        matched: PhysicalAlternative | None = None
        for template in expected:
            exact = [
                item
                for item in candidates
                if item.chain_id == template.chain_id
                and tuple((step.provider_id, step.node_id) for step in item.step_placements)
                == template.providers_and_nodes
                and tuple(
                    sorted(
                        value.source_id
                        for value in item.external_inputs
                        if value.source_id is not None
                    )
                )
                == template.source_ids
            ]
            if exact:
                matched = min(exact, key=lambda item: item.alternative_id)
                break
        if matched is not None:
            selected.append(matched.alternative_id)
    return tuple(selected)


__all__ = ["LiveBaselinePlanningPolicy"]
