"""Convert physical-planner output into scheduler-facing plan candidates.

This is an adapter, not an ordering or admission algorithm. Its input is a
selected/fallback search result plus demands and selected alternatives; its
output packages the concrete plan and predicted costs as ``PlanCandidate``.
"""

from __future__ import annotations

from collections.abc import Iterable

from fable.common.enums import ExecutionInputKind, PlanStatus
from fable.common.ids import deterministic_id
from fable.common.schemas import (
    ExecutionInput,
    ExecutionPlan,
    PlanStep,
    PredicateDemand,
    ResourceReservation,
)
from fable.planning.models import ExternalInputKind, PhysicalAlternative, PhysicalAlternativeGraph
from fable.planning.provider_registry import ProviderRegistry
from fable.planning.search_models import LabelSearchState, PlanSearchResult
from fable.planning.search.projection import ExecutionPlanProjector

from .models import PlanCandidate, TaskSchedulingPolicy


class CandidateAdapterError(ValueError):
    pass


def candidate_from_alternatives(
    alternatives: tuple[PhysicalAlternative, ...],
    demands: tuple[PredicateDemand, ...],
    *,
    provider_registry: ProviderRegistry,
    task_policy: TaskSchedulingPolicy,
    predicted_completion_ms: int | None = None,
    allow_replicated_demand: bool = False,
) -> PlanCandidate:
    """Project an explicitly controlled baseline selection into execution.

    B0/B1/B4 select concrete alternatives without producing an internal beam
    search label.  This adapter preserves that controlled selection while
    emitting the same self-contained execution contract used by FABLE.
    """

    if not alternatives or not demands:
        raise CandidateAdapterError("at least one alternative and demand is required")
    checkpoints = {item.checkpoint_id for item in alternatives}
    if len(checkpoints) != 1:
        raise CandidateAdapterError("one candidate may cover only one checkpoint")
    demand_map = {item.demand_id: item for item in demands}
    alternative_demand_ids = {item.demand_id for item in alternatives}
    if not alternative_demand_ids.issubset(demand_map):
        raise CandidateAdapterError("alternative references an unknown demand")
    if not allow_replicated_demand and len(alternative_demand_ids) != len(alternatives):
        raise CandidateAdapterError("duplicate demand alternatives require explicit replication")

    steps: list[PlanStep] = []
    resources: dict[str, list[float]] = {}
    startup_cost_ms = 0
    transfer_bytes = 0
    for alternative in alternatives:
        demand = demand_map[alternative.demand_id]
        chain = provider_registry.chain(alternative.chain_id)
        chain_steps = {item.step_id: item for item in chain.steps}
        external = {item.input_name: item for item in alternative.external_inputs}
        placement_by_id = {item.step_id: item for item in alternative.step_placements}
        for placement in alternative.step_placements:
            authored = chain_steps[placement.step_id]
            inputs = []
            dependencies = []
            input_types = []
            for port_name, source_ref in authored.bindings.items():
                if source_ref.startswith("external."):
                    item = external.get(source_ref.removeprefix("external."))
                    if item is None or item.kind == ExternalInputKind.OMITTED_OPTIONAL:
                        continue
                    inputs.append(
                        ExecutionInput(
                            name=port_name,
                            data_type=item.data_type,
                            kind=ExecutionInputKind(item.kind.value),
                            node_id=item.node_id,
                            source_id=item.source_id,
                            artifact_id=item.artifact_id,
                            bytes=item.bytes,
                            expires_at=item.expires_at,
                        )
                    )
                    input_types.append(item.data_type)
                else:
                    source_step = source_ref.split(".", 1)[0]
                    dependencies.append(f"{alternative.alternative_id}:{source_step}")
                    port = provider_registry.provider_input_ports(placement.provider_id).get(port_name)
                    if port is not None:
                        input_types.append(port.data_type)
            transfer_ms = sum(
                item.estimated_ms
                for item in alternative.transfers
                if item.target_step_id == placement.step_id
            )
            step_transfer_bytes = sum(
                item.bytes
                for item in alternative.transfers
                if item.target_step_id == placement.step_id
            )
            output_types = tuple(
                port.data_type
                for port in provider_registry.provider_output_ports(placement.provider_id).values()
            )
            steps.append(
                PlanStep(
                    step_id=f"{alternative.alternative_id}:{placement.step_id}",
                    provider_id=placement.provider_id,
                    node_id=placement.node_id,
                    demand_id=alternative.demand_id,
                    alternative_id=alternative.alternative_id,
                    chain_id=alternative.chain_id,
                    execution_mode=alternative.execution_mode,
                    inputs=tuple(inputs),
                    input_artifact_ids=tuple(
                        item.artifact_id for item in inputs if item.artifact_id is not None
                    ),
                    input_data_types=tuple(input_types),
                    output_data_types=output_types,
                    parameters=tuple(
                        sorted(demand.semantic_predicate.parameters.items())
                    ),
                    depends_on_step_ids=tuple(dependencies),
                    cpu_cores=placement.cpu_cores,
                    memory_mb=placement.memory_mb,
                    gpu_memory_mb=placement.gpu_memory_mb,
                    quality_score=placement.quality_score,
                    reused_provider_instance_id=placement.reused_provider_instance_id,
                    estimated_startup_ms=placement.startup_ms,
                    estimated_execution_ms=placement.execution_ms,
                    estimated_transfer_ms=transfer_ms,
                    estimated_transfer_bytes=step_transfer_bytes,
                )
            )
            totals = resources.setdefault(placement.node_id, [0.0, 0.0, 0.0, 0.0])
            totals[0] += placement.cpu_cores
            totals[1] += placement.memory_mb
            totals[2] += placement.gpu_memory_mb
            totals[3] += step_transfer_bytes
            startup_cost_ms += placement.startup_ms
            transfer_bytes += step_transfer_bytes

    demand_ids = tuple(sorted(demand_map, key=str))
    label_id = deterministic_id(
        "controlled_alternatives",
        {
            "checkpoint_id": str(next(iter(checkpoints))),
            "alternative_ids": sorted(item.alternative_id for item in alternatives),
            "demand_ids": [str(item) for item in demand_ids],
        },
        length=40,
    )
    plan = ExecutionPlan(
        label_id=label_id,
        checkpoint_id=next(iter(checkpoints)),
        demand_ids=demand_ids,
        steps=tuple(steps),
        reservations=tuple(
            ResourceReservation(
                node_id=node_id,
                cpu_cores=values[0],
                memory_mb=int(values[1]),
                gpu_memory_mb=int(values[2]),
                network_bytes=int(values[3]),
            )
            for node_id, values in sorted(resources.items())
        ),
        status=PlanStatus.CANDIDATE,
    )
    completion = predicted_completion_ms
    if completion is None:
        completion = max(item.estimated_completion_ms for item in alternatives)
    return PlanCandidate(
        plan=plan,
        demands=tuple(demand_map[item] for item in demand_ids),
        task_policy=task_policy,
        predicted_completion_ms=completion,
        startup_cost_ms=startup_cost_ms,
        incremental_resource_cost_units=sum(
            values[0] + values[1] / 1024.0 + 2.0 * values[2] / 1024.0
            for values in resources.values()
        ),
        transfer_bytes=transfer_bytes,
        alternatives=alternatives,
        replicated_demand_execution=allow_replicated_demand,
    )


def candidate_from_search_result(
    result: PlanSearchResult,
    graph: PhysicalAlternativeGraph,
    demands: Iterable[PredicateDemand],
    *,
    task_policy: TaskSchedulingPolicy,
    fallback_index: int | None = None,
) -> PlanCandidate:
    """Build a scheduling candidate from one selected or fallback search label.

    The selected label is projected to ``ExecutionPlan`` when necessary, its
    referenced alternatives are resolved, and plan/demands/policy plus startup,
    completion, transfer, and resource estimates are packaged for ordering and
    admission. No worker is started here.
    """
    demand_tuple = tuple(demands)
    if fallback_index is None:
        state = result.selected
        plan = result.execution_plan
        fallback_rank = 0
    else:
        try:
            state = result.fallbacks[fallback_index]
        except IndexError as exc:
            raise CandidateAdapterError(f"unknown fallback index {fallback_index}") from exc
        # Recreate the immutable execution-plan projection for the fallback.
        plan = ExecutionPlanProjector().project(state)
        assert plan is not None
        fallback_rank = fallback_index + 1
    if state is None or plan is None:
        raise CandidateAdapterError("search result has no feasible selected plan")
    if result.trace.graph_id != graph.graph_id:
        raise CandidateAdapterError("search result and physical graph do not match")
    return PlanCandidate(
        plan=plan,
        demands=demand_tuple,
        task_policy=task_policy,
        predicted_completion_ms=state.label.cost.predicted_completion_ms,
        startup_cost_ms=state.label.cost.startup_cost_ms,
        incremental_resource_cost_units=state.label.cost.resource_cost_units,
        transfer_bytes=state.label.cost.transfer_bytes,
        fallback_rank=fallback_rank,
        alternatives=tuple(
            item for item in graph.alternatives
            if item.alternative_id in set(state.selected_alternative_ids)
        ),
    )
