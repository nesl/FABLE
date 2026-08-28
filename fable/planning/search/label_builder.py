"""Immutable label construction for bounded physical-plan search."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from uuid import UUID

from fable.common.enums import ProviderPortKind
from fable.common.schemas import ExecutionInput, PhysicalPlanLabel, PlanCost, PlanStep, PredicateDemand
from fable.planning.models import ExternalInputKind, PhysicalAlternative
from fable.planning.provider_registry import ProviderRegistry
from fable.planning.representation import RepresentationCompatibility
from fable.planning.search.resources import combine_resources
from fable.planning.search_models import LabelSearchState


class LabelBuilder:
    """Build a new immutable search label from one selected alternative."""

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry,
        representation_compatibility: RepresentationCompatibility,
        near_expiry_horizon_ms: int,
    ) -> None:
        self.providers = provider_registry
        self.representations = representation_compatibility
        self.near_expiry_horizon_ms = near_expiry_horizon_ms

    def extend_label(
            self,
            parent: LabelSearchState | None,
            alternative: PhysicalAlternative,
            demand: PredicateDemand,
            *,
            demand_map: Mapping[UUID, PredicateDemand],
            now: datetime,
        ) -> LabelSearchState:
            """Return a new immutable label; the parent is never modified."""

            chain = self.providers.chain(alternative.chain_id)
            prefix = alternative.alternative_id
            new_steps: list[PlanStep] = []
            for placement in alternative.step_placements:
                step_contract = next(
                    step for step in chain.steps if step.step_id == placement.step_id
                )
                external_root_names = _external_roots_for_step(chain, placement.step_id)
                step_external_inputs = tuple(
                    item
                    for item in alternative.external_inputs
                    if item.input_name in external_root_names
                )
                dependencies = tuple(
                    sorted(
                        f"{prefix}:{source_ref.split('.', 1)[0]}"
                        for source_ref in step_contract.bindings.values()
                        if not source_ref.startswith("external.") and "." in source_ref
                    )
                )
                input_types = tuple(
                    sorted(
                        transfer.data_type
                        for transfer in alternative.transfers
                        if transfer.target_step_id == placement.step_id
                    )
                )
                output_types = tuple(
                    sorted(
                        port.data_type
                        for port in self.providers.provider(placement.provider_id).ports
                        if port.kind in (ProviderPortKind.OUTPUT, ProviderPortKind.STATE_OUTPUT)
                    )
                )
                transfer_ms = sum(
                    transfer.estimated_ms
                    for transfer in alternative.transfers
                    if transfer.target_step_id == placement.step_id
                )
                transfer_bytes = sum(
                    transfer.bytes
                    for transfer in alternative.transfers
                    if transfer.target_step_id == placement.step_id
                )
                new_steps.append(
                    PlanStep(
                        step_id=f"{prefix}:{placement.step_id}",
                        provider_id=placement.provider_id,
                        node_id=placement.node_id,
                        demand_id=demand.demand_id,
                        alternative_id=alternative.alternative_id,
                        chain_id=alternative.chain_id,
                        execution_mode=alternative.execution_mode,
                        inputs=tuple(
                            ExecutionInput(
                                name=item.input_name,
                                data_type=item.data_type,
                                kind=item.kind,
                                node_id=item.node_id,
                                source_id=item.source_id,
                                artifact_id=item.artifact_id,
                                bytes=item.bytes,
                                expires_at=item.expires_at,
                            )
                            for item in step_external_inputs
                        ),
                        input_artifact_ids=tuple(
                            sorted(
                                (
                                    item.artifact_id
                                    for item in step_external_inputs
                                    if item.artifact_id is not None
                                ),
                                key=str,
                            )
                        ),
                        input_data_types=input_types,
                        output_data_types=output_types,
                        depends_on_step_ids=dependencies,
                        cpu_cores=placement.cpu_cores,
                        memory_mb=placement.memory_mb,
                        gpu_memory_mb=placement.gpu_memory_mb,
                        quality_score=placement.quality_score,
                        reused_provider_instance_id=placement.reused_provider_instance_id,
                        estimated_startup_ms=placement.startup_ms,
                        estimated_execution_ms=placement.execution_ms,
                        estimated_transfer_ms=transfer_ms,
                        estimated_transfer_bytes=transfer_bytes,
                    )
                )

            parent_steps = parent.label.steps if parent is not None else ()
            covered = (
                (*parent.label.covered_demand_ids, demand.demand_id)
                if parent is not None
                else (demand.demand_id,)
            )
            selected_alternatives = (
                (*parent.selected_alternative_ids, alternative.alternative_id)
                if parent is not None
                else (alternative.alternative_id,)
            )
            selected_chains = (
                (*parent.selected_chain_ids, alternative.chain_id)
                if parent is not None
                else (alternative.chain_id,)
            )
            completion_by_demand = dict(parent.completion_by_demand_ms) if parent else {}
            completion_by_demand[demand.demand_id] = alternative.estimated_completion_ms
            predicted_completion = max(completion_by_demand.values())
            slacks = [
                int((demand_map[demand_id].deadline.latest_useful_completion - now).total_seconds() * 1000)
                - completion_ms
                for demand_id, completion_ms in completion_by_demand.items()
            ]

            resources = combine_resources(
                parent.resource_map() if parent else {},
                alternative,
                existing_provider_keys=frozenset(
                    (step.node_id, step.provider_id)
                    for step in (parent.label.steps if parent else ())
                    if self.providers.provider(
                        step.provider_id
                    ).execution_capabilities.supports_shared_execution
                ),
            )
            resource_tuple = tuple(sorted(resources.values(), key=lambda item: item.node_id))
            total_cpu = sum(item.cpu_cores for item in resource_tuple)
            total_memory = sum(item.memory_mb for item in resource_tuple)
            total_gpu = sum(item.gpu_memory_mb for item in resource_tuple)
            all_steps = (*parent_steps, *new_steps)
            startup_ms = sum(step.estimated_startup_ms for step in all_steps)
            transfer_bytes = sum(step.estimated_transfer_bytes for step in all_steps)
            resource_cost_units = total_cpu + (total_memory / 1024.0) + (2.0 * total_gpu / 1024.0)

            continuation_types = tuple(
                sorted(
                    set(parent.label.continuation_output_types if parent else ())
                    | set(alternative.continuation_output_types)
                )
            )
            continuation_consumers = tuple(
                sorted(self.representations.combined_consumers(continuation_types))
            )
            desired_types = {
                desired
                for demand_id in covered
                for desired in demand_map[demand_id].desired_continuation_types
            }
            missing_desired_types = tuple(sorted(desired_types - set(continuation_types)))
            artifact_ids = tuple(
                sorted(
                    set(parent.label.input_artifact_ids if parent else ())
                    | {
                        item.artifact_id
                        for item in alternative.external_inputs
                        if item.artifact_id is not None
                    },
                    key=str,
                )
            )
            minimum_quality = (
                min(parent.minimum_quality_score, alternative.minimum_quality_score)
                if parent is not None
                else alternative.minimum_quality_score
            )
            perishability = min(
                parent.perishability_rank if parent is not None else 99,
                self._perishability_rank(alternative, demand, now=now),
            )
            spatial_penalty = (
                (parent.spatial_preference_penalty if parent is not None else 0)
                + alternative.spatial_preference_penalty
            )
            expiry_candidates = [
                item.expires_at
                for item in alternative.external_inputs
                if item.expires_at is not None
            ]
            if parent is not None and parent.expires_at is not None:
                expiry_candidates.append(parent.expires_at)
            expiry_candidates.extend(
                demand_map[demand_id].deadline.latest_useful_completion
                for demand_id in covered
            )
            expires_at = min(expiry_candidates) if expiry_candidates else None

            physical = PhysicalPlanLabel(
                checkpoint_id=demand.checkpoint_id,
                covered_demand_ids=covered,
                steps=all_steps,
                input_artifact_ids=artifact_ids,
                continuation_output_types=continuation_types,
                cost=PlanCost(
                    predicted_completion_ms=predicted_completion,
                    deadline_slack_ms=min(slacks),
                    startup_cost_ms=startup_ms,
                    resource_cost_units=resource_cost_units,
                    transfer_bytes=transfer_bytes,
                ),
                hard_constraints_satisfied=True,
                quality_floor_satisfied=True,
                feasibility_reasons=("all Phase-4 hard feasibility filters passed",),
                parent_label_id=(parent.label_id if parent is not None else None),
            )
            return LabelSearchState(
                label=physical,
                selected_alternative_ids=selected_alternatives,
                selected_chain_ids=selected_chains,
                node_resources=resource_tuple,
                completion_by_demand_ms=tuple(
                    sorted(completion_by_demand.items(), key=lambda item: str(item[0]))
                ),
                minimum_quality_score=minimum_quality,
                perishability_rank=perishability,
                spatial_preference_penalty=spatial_penalty,
                continuation_consumer_set=continuation_consumers,
                missing_desired_continuation_types=missing_desired_types,
                total_cpu_cores=total_cpu,
                total_memory_mb=total_memory,
                total_gpu_memory_mb=total_gpu,
                expires_at=expires_at,
            )

    def _perishability_rank(
            self,
            alternative: PhysicalAlternative,
            demand: PredicateDemand,
            *,
            now: datetime,
        ) -> int:
            if any(
                item.kind == ExternalInputKind.LIVE_SOURCE
                for item in alternative.external_inputs
            ):
                return 0
            expirations = sorted(
                item.expires_at
                for item in alternative.external_inputs
                if item.expires_at is not None
            )
            if not expirations:
                return 3
            earliest = expirations[0]
            if earliest <= now + timedelta(milliseconds=self.near_expiry_horizon_ms):
                return 0
            if earliest <= demand.deadline.latest_useful_completion:
                return 1
            return 2


def _external_roots_for_step(chain, step_id: str) -> set[str]:
    """Return external input names that transitively feed one chain step."""

    step_by_id = {step.step_id: step for step in chain.steps}
    cache: dict[str, set[str]] = {}

    def visit(current: str, stack: set[str]) -> set[str]:
        if current in cache:
            return cache[current]
        if current in stack:
            raise ValueError(f"cycle in provider chain {chain.chain_id}")
        roots: set[str] = set()
        next_stack = {*stack, current}
        step = step_by_id[current]
        for source_ref in step.bindings.values():
            if source_ref.startswith("external."):
                roots.add(source_ref.split(".", 1)[1])
            elif "." in source_ref:
                upstream = source_ref.split(".", 1)[0]
                if upstream in step_by_id:
                    roots.update(visit(upstream, next_stack))
        cache[current] = roots
        return roots

    return visit(step_id, set())


__all__ = ["LabelBuilder"]
