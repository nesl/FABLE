"""Checkpoint-bounded, representation-aware label-driven beam search.

The planner consumes the Phase-3 physical alternative graph.  It never changes
semantic hypotheses; it selects one physical realization for each predicate
demand needed to resolve the current checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from itertools import product
from math import prod
from uuid import UUID

from pydantic import Field

from fable.common.base import FableModel
from fable.common.enums import (
    ArtifactAccessMode,
    BindingCapability,
    PlanStatus,
    ProviderPortKind,
)
from fable.common.ids import deterministic_id
from fable.common.schemas import (
    ExecutionPlan,
    PhysicalPlanLabel,
    PlanCost,
    PlanStep,
    PredicateDemand,
    ResourceReservation,
)
from fable.common.time import ensure_utc, utc_now

from .artifact_catalog import ArtifactCatalog, ArtifactCatalogError
from .deployment import DeploymentGraph, DeploymentGraphError
from .models import (
    ExternalInputKind,
    PhysicalAlternative,
    PhysicalAlternativeGraph,
    TransferMode,
)
from .provider_registry import ProviderRegistry, ProviderRegistryError
from .representation import RepresentationCompatibility
from .search_models import (
    BeamBoundaryTrace,
    FeasibilityFailure,
    LabelSearchState,
    NodeResourceFootprint,
    OracleComparison,
    OracleStatus,
    PlanSearchResult,
    PlanSearchTrace,
    PruneCode,
    PruningRecord,
)


class PlanSearchError(ValueError):
    """Raised when a Phase-4 search request is internally inconsistent."""


class BeamSearchConfig(FableModel):
    beam_width: int = Field(default=8, ge=1)
    fallback_count: int = Field(default=2, ge=0)
    minimum_quality_score: float = Field(default=0.0, ge=0, le=1)
    minimum_quality_by_predicate: dict[str, float] = Field(default_factory=dict)
    near_expiry_horizon_ms: int = Field(default=5_000, ge=0)
    require_declared_binding_capabilities: bool = False
    run_oracle: bool = True
    oracle_max_combinations: int = Field(default=50_000, ge=1)


class BoundedLabelPlanner:
    """Bounded multi-label planner with hard filtering and dominance pruning."""

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry,
        artifact_catalog: ArtifactCatalog,
        deployment: DeploymentGraph,
        config: BeamSearchConfig | None = None,
        representation_compatibility: RepresentationCompatibility | None = None,
    ) -> None:
        self.providers = provider_registry
        self.artifacts = artifact_catalog
        self.deployment = deployment
        self.config = config or BeamSearchConfig()
        self.representations = representation_compatibility or RepresentationCompatibility(
            provider_registry
        )

    def search(
        self,
        graph: PhysicalAlternativeGraph,
        demands: Iterable[PredicateDemand],
        *,
        now: datetime | None = None,
        required_checkpoint_consumers: Iterable[str] = (),
    ) -> PlanSearchResult:
        observed_now = ensure_utc(now or utc_now())
        demand_map = {demand.demand_id: demand for demand in demands}
        if not demand_map:
            raise PlanSearchError("at least one predicate demand is required")
        if set(graph.demand_ids) != set(demand_map):
            raise PlanSearchError("physical graph demand IDs do not match the supplied demands")
        checkpoint_ids = {demand.checkpoint_id for demand in demand_map.values()}
        if len(checkpoint_ids) != 1:
            raise PlanSearchError("one search invocation must cover exactly one semantic checkpoint")
        checkpoint_id = next(iter(checkpoint_ids))
        if set(graph.checkpoint_ids) != {checkpoint_id}:
            raise PlanSearchError("physical graph checkpoint does not match the demands")

        required_consumers = tuple(sorted(set(required_checkpoint_consumers)))
        demand_order = self._demand_order(demand_map.values())
        alternatives_by_demand = self._alternatives_by_demand(graph)
        phase3_pruning = tuple(
            PruningRecord(
                boundary_index=0,
                code=PruneCode.PHASE3_PRUNED,
                reason=f"{item.code}: {item.reason}",
                demand_id=item.demand_id,
                alternative_id=item.candidate_id,
            )
            for item in graph.pruned
        )

        retained: tuple[LabelSearchState | None, ...] = (None,)
        boundary_traces: list[BeamBoundaryTrace] = []
        final_dominated_archive: tuple[LabelSearchState, ...] = ()

        for boundary_index, demand_id in enumerate(demand_order, start=1):
            demand = demand_map[demand_id]
            generated: list[LabelSearchState] = []
            pruning: list[PruningRecord] = []
            alternatives = alternatives_by_demand.get(demand_id, ())
            for parent in retained:
                for alternative in alternatives:
                    failures = self.check_extension(
                        parent,
                        alternative,
                        demand,
                        demand_map=demand_map,
                        now=observed_now,
                    )
                    if failures:
                        for failure in failures:
                            pruning.append(
                                PruningRecord(
                                    boundary_index=boundary_index,
                                    code=failure.code,
                                    reason=failure.reason,
                                    demand_id=demand_id,
                                    alternative_id=alternative.alternative_id,
                                    parent_label_id=(parent.label_id if parent else None),
                                )
                            )
                        continue
                    generated.append(
                        self.extend_label(
                            parent,
                            alternative,
                            demand,
                            demand_map=demand_map,
                            now=observed_now,
                        )
                    )

            unique, duplicate_records = self._deduplicate(
                generated,
                boundary_index=boundary_index,
                demand_id=demand_id,
            )
            pruning.extend(duplicate_records)
            nondominated, dominance_records = self._dominance_prune(
                unique,
                boundary_index=boundary_index,
                demand_id=demand_id,
            )
            pruning.extend(dominance_records)
            if boundary_index == len(demand_order):
                nondominated_ids = {item.label_id for item in nondominated}
                final_dominated_archive = tuple(
                    item for item in unique if item.label_id not in nondominated_ids
                )
            ranked = tuple(sorted(nondominated, key=self.rank_key))
            kept = ranked[: self.config.beam_width]
            for removed in ranked[self.config.beam_width :]:
                pruning.append(
                    PruningRecord(
                        boundary_index=boundary_index,
                        code=PruneCode.BEAM_LIMIT,
                        reason=f"label falls outside beam width {self.config.beam_width}",
                        demand_id=demand_id,
                        label_id=removed.label_id,
                    )
                )
            boundary_traces.append(
                BeamBoundaryTrace(
                    boundary_index=boundary_index,
                    demand_id=demand_id,
                    generated_label_ids=tuple(item.label_id for item in generated),
                    feasible_label_ids=tuple(item.label_id for item in ranked),
                    retained_label_ids=tuple(item.label_id for item in kept),
                    pruning_records=tuple(pruning),
                )
            )
            retained = tuple(kept)
            if not retained:
                break

        complete = tuple(item for item in retained if item is not None)
        if required_consumers:
            accepted: list[LabelSearchState] = []
            terminal_pruning: list[PruningRecord] = []
            for state in complete:
                if set(required_consumers).issubset(set(state.continuation_consumer_set)):
                    accepted.append(state)
                else:
                    missing = sorted(
                        set(required_consumers) - set(state.continuation_consumer_set)
                    )
                    terminal_pruning.append(
                        PruningRecord(
                            boundary_index=len(demand_order) + 1,
                            code=PruneCode.CHECKPOINT_CONTINUATION_INCOMPATIBLE,
                            reason=f"checkpoint continuation cannot serve consumers {missing}",
                            label_id=state.label_id,
                        )
                    )
            if terminal_pruning:
                boundary_traces.append(
                    BeamBoundaryTrace(
                        boundary_index=len(demand_order) + 1,
                        demand_id=None,
                        generated_label_ids=tuple(item.label_id for item in complete),
                        feasible_label_ids=tuple(item.label_id for item in accepted),
                        retained_label_ids=tuple(item.label_id for item in accepted),
                        pruning_records=tuple(terminal_pruning),
                    )
                )
            complete = tuple(accepted)

        ranked_complete = tuple(sorted(complete, key=self.rank_key))
        selected = ranked_complete[0] if ranked_complete else None
        fallback_candidates = [*ranked_complete[1:], *final_dominated_archive]
        if required_consumers:
            fallback_candidates = [
                item
                for item in fallback_candidates
                if set(required_consumers).issubset(item.continuation_consumer_set)
            ]
        fallback_by_id = {
            item.label_id: item
            for item in fallback_candidates
            if selected is None or item.label_id != selected.label_id
        }
        fallbacks = tuple(
            sorted(fallback_by_id.values(), key=self.rank_key)[: self.config.fallback_count]
        )
        execution_plan = self.to_execution_plan(selected) if selected is not None else None

        oracle = OracleComparison(status=OracleStatus.NOT_RUN)
        if self.config.run_oracle:
            oracle = self._oracle_compare(
                graph=graph,
                demand_map=demand_map,
                demand_order=demand_order,
                selected=selected,
                now=observed_now,
                required_checkpoint_consumers=required_consumers,
            )

        search_id = deterministic_id(
            "search",
            {
                "graph_id": graph.graph_id,
                "checkpoint_id": checkpoint_id,
                "demand_order": demand_order,
                "beam_width": self.config.beam_width,
                "required_consumers": required_consumers,
                "now": observed_now,
            },
            length=32,
        )
        selection_rank = self.rank_key(selected) if selected is not None else ()
        trace = PlanSearchTrace(
            search_id=search_id,
            graph_id=graph.graph_id,
            checkpoint_id=checkpoint_id,
            beam_width=self.config.beam_width,
            demand_order=demand_order,
            boundaries=tuple(boundary_traces),
            phase3_pruning=phase3_pruning,
            selected_label_id=(selected.label_id if selected else None),
            fallback_label_ids=tuple(item.label_id for item in fallbacks),
            required_checkpoint_consumers=required_consumers,
            selection_rank=selection_rank,
            oracle=oracle,
            created_at=observed_now,
        )
        return PlanSearchResult(
            selected=selected,
            fallbacks=fallbacks,
            execution_plan=execution_plan,
            trace=trace,
        )

    def check_extension(
        self,
        parent: LabelSearchState | None,
        alternative: PhysicalAlternative,
        demand: PredicateDemand,
        *,
        demand_map: Mapping[UUID, PredicateDemand],
        now: datetime,
    ) -> tuple[FeasibilityFailure, ...]:
        """Apply all hard Phase-4 feasibility filters to one immutable extension."""

        failures: list[FeasibilityFailure] = []
        if alternative.demand_id != demand.demand_id:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.DEMAND_MISMATCH,
                    reason="physical alternative belongs to a different predicate demand",
                )
            )
        if alternative.checkpoint_id != demand.checkpoint_id:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.CHECKPOINT_MISMATCH,
                    reason="physical alternative belongs to a different semantic checkpoint",
                )
            )
        if parent is not None and parent.label.checkpoint_id != demand.checkpoint_id:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.CHECKPOINT_MISMATCH,
                    reason="partial label and demand belong to different checkpoints",
                )
            )
        if alternative.result_output_type not in demand.acceptable_output_types:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.RESULT_SCHEMA_INCOMPATIBLE,
                    reason=(
                        f"result type {alternative.result_output_type} is not one of "
                        f"{sorted(demand.acceptable_output_types)}"
                    ),
                )
            )

        try:
            chain = self.providers.chain(alternative.chain_id)
        except ProviderRegistryError as exc:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.INPUT_SCHEMA_INCOMPATIBLE,
                    reason=str(exc),
                )
            )
            return tuple(failures)

        if chain.output_types.get("result") != alternative.result_output_type:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.RESULT_SCHEMA_INCOMPATIBLE,
                    reason=(
                        f"alternative result {alternative.result_output_type} does not match "
                        f"chain result {chain.output_types.get('result')}"
                    ),
                )
            )
        if not set(alternative.continuation_output_types).issubset(
            set(chain.continuation_output_types)
        ):
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.INPUT_SCHEMA_INCOMPATIBLE,
                    reason="alternative declares continuation outputs absent from its chain contract",
                )
            )
        expected_external = {item.name: item for item in chain.external_inputs}
        realized_external = {item.input_name: item for item in alternative.external_inputs}
        for input_name, external in expected_external.items():
            realized = realized_external.get(input_name)
            if realized is None:
                failures.append(
                    FeasibilityFailure(
                        code=PruneCode.INPUT_SCHEMA_INCOMPATIBLE,
                        reason=f"chain input {input_name} is missing from the alternative",
                    )
                )
                continue
            if realized.data_type != external.data_type:
                failures.append(
                    FeasibilityFailure(
                        code=PruneCode.INPUT_SCHEMA_INCOMPATIBLE,
                        reason=(
                            f"chain input {input_name} requires {external.data_type}, "
                            f"got {realized.data_type}"
                        ),
                    )
                )
            if realized.kind == ExternalInputKind.OMITTED_OPTIONAL and not external.optional:
                failures.append(
                    FeasibilityFailure(
                        code=PruneCode.INPUT_SCHEMA_INCOMPATIBLE,
                        reason=f"required chain input {input_name} was omitted",
                    )
                )
        available_required_inputs = {
            item.data_type
            for item in alternative.external_inputs
            if item.kind != ExternalInputKind.OMITTED_OPTIONAL
        }
        if parent is not None:
            available_required_inputs.update(parent.label.continuation_output_types)
        missing_required_inputs = set(demand.required_input_artifact_types) - available_required_inputs
        if missing_required_inputs:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.INPUT_SCHEMA_INCOMPATIBLE,
                    reason=f"required input artifact types are unavailable: {sorted(missing_required_inputs)}",
                )
            )

        unexpected_external = set(realized_external) - set(expected_external)
        if unexpected_external:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.INPUT_SCHEMA_INCOMPATIBLE,
                    reason=f"alternative includes unknown chain inputs {sorted(unexpected_external)}",
                )
            )
        expected_steps = {step.step_id: step.provider_id for step in chain.steps}
        realized_steps = {step.step_id: step.provider_id for step in alternative.step_placements}
        if expected_steps != realized_steps:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.INPUT_SCHEMA_INCOMPATIBLE,
                    reason="alternative step/provider assignments do not match the chain contract",
                )
            )

        for item in alternative.external_inputs:
            if item.kind == ExternalInputKind.OMITTED_OPTIONAL:
                continue
            try:
                self.providers.data_type(item.data_type)
            except ProviderRegistryError as exc:
                failures.append(
                    FeasibilityFailure(
                        code=PruneCode.INPUT_SCHEMA_INCOMPATIBLE,
                        reason=str(exc),
                    )
                )
                continue
            if item.kind == ExternalInputKind.LIVE_SOURCE:
                if item.source_id is None:
                    failures.append(
                        FeasibilityFailure(
                            code=PruneCode.SOURCE_UNAVAILABLE,
                            reason=f"live input {item.input_name} has no source identifier",
                        )
                    )
                    continue
                try:
                    source = self.deployment.source(item.source_id)
                except DeploymentGraphError as exc:
                    failures.append(
                        FeasibilityFailure(code=PruneCode.SOURCE_UNAVAILABLE, reason=str(exc))
                    )
                    continue
                if not source.available or source.node_id != item.node_id:
                    failures.append(
                        FeasibilityFailure(
                            code=PruneCode.SOURCE_UNAVAILABLE,
                            reason=f"source {item.source_id} is unavailable or on a different node",
                        )
                    )
                if item.data_type not in source.live_data_types:
                    failures.append(
                        FeasibilityFailure(
                            code=PruneCode.INPUT_SCHEMA_INCOMPATIBLE,
                            reason=f"source {item.source_id} does not expose {item.data_type}",
                        )
                    )
                if demand.eligible_source_ids and item.source_id not in demand.eligible_source_ids:
                    failures.append(
                        FeasibilityFailure(
                            code=PruneCode.POLICY_VIOLATION,
                            reason=f"source {item.source_id} is outside demand eligibility",
                        )
                    )
                if demand.event_time_interval.end < now:
                    if (
                        source.raw_buffer_interval is None
                        or not source.raw_buffer_interval.contains_interval(
                            demand.event_time_interval
                        )
                    ):
                        failures.append(
                            FeasibilityFailure(
                                code=PruneCode.EVENT_TIME_UNAVAILABLE,
                                reason=f"source {item.source_id} no longer retains the event interval",
                            )
                        )
            elif item.kind in (
                ExternalInputKind.RETAINED_ARTIFACT,
                ExternalInputKind.DEPLOYMENT_ARTIFACT,
            ):
                if item.artifact_id is None:
                    failures.append(
                        FeasibilityFailure(
                            code=PruneCode.ARTIFACT_MISSING,
                            reason=f"input {item.input_name} has no artifact identifier",
                        )
                    )
                    continue
                try:
                    artifact = self.artifacts.get(item.artifact_id)
                except ArtifactCatalogError as exc:
                    failures.append(
                        FeasibilityFailure(code=PruneCode.ARTIFACT_MISSING, reason=str(exc))
                    )
                    continue
                if (
                    artifact.artifact_type != item.data_type
                    or artifact.artifact_schema_version != item.data_type
                ):
                    failures.append(
                        FeasibilityFailure(
                            code=PruneCode.INPUT_SCHEMA_INCOMPATIBLE,
                            reason=(
                                f"artifact {item.artifact_id} is "
                                f"{artifact.artifact_type}/{artifact.artifact_schema_version}, "
                                f"not {item.data_type}"
                            ),
                        )
                    )
                producer_contract = self.providers.providers.get(artifact.producer.provider_id)
                if (
                    producer_contract is not None
                    and artifact.producer.provider_contract_version
                    != producer_contract.contract_version
                ):
                    failures.append(
                        FeasibilityFailure(
                            code=PruneCode.INPUT_SCHEMA_INCOMPATIBLE,
                            reason=(
                                f"artifact {item.artifact_id} was produced by contract version "
                                f"{artifact.producer.provider_contract_version}, registry has "
                                f"{producer_contract.contract_version}"
                            ),
                        )
                    )
                if not artifact.event_time_interval.contains_interval(
                    demand.event_time_interval
                ):
                    failures.append(
                        FeasibilityFailure(
                            code=PruneCode.EVENT_TIME_UNAVAILABLE,
                            reason=f"artifact {item.artifact_id} does not cover the demand interval",
                        )
                    )
                finish = now + timedelta(milliseconds=alternative.estimated_completion_ms)
                if artifact.expires_at is not None and artifact.expires_at <= finish:
                    failures.append(
                        FeasibilityFailure(
                            code=PruneCode.ARTIFACT_EXPIRED,
                            reason=f"artifact {item.artifact_id} expires before predicted completion",
                        )
                    )
                if artifact.valid_until is not None and artifact.valid_until < demand.event_time_interval.end:
                    failures.append(
                        FeasibilityFailure(
                            code=PruneCode.EVENT_TIME_UNAVAILABLE,
                            reason=f"artifact {item.artifact_id} becomes invalid during the demand interval",
                        )
                    )
                if demand.hard_constraints.required_access_modes and not set(
                    demand.hard_constraints.required_access_modes
                ).issubset(set(artifact.access_modes)):
                    failures.append(
                        FeasibilityFailure(
                            code=PruneCode.POLICY_VIOLATION,
                            reason=f"artifact {item.artifact_id} lacks a required access mode",
                        )
                    )

        for data_type in alternative.continuation_output_types:
            try:
                self.providers.data_type(data_type)
            except ProviderRegistryError as exc:
                failures.append(
                    FeasibilityFailure(
                        code=PruneCode.INPUT_SCHEMA_INCOMPATIBLE,
                        reason=f"unknown continuation schema: {exc}",
                    )
                )

        semantic_providers = []
        for step in chain.steps:
            provider = self.providers.provider(step.provider_id)
            if demand.semantic_predicate.predicate_id in provider.semantic_capabilities.predicate_ids:
                semantic_providers.append(provider)
        if not semantic_providers:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.BINDING_CAPABILITY_MISSING,
                    reason=(
                        f"chain {chain.chain_id} has no provider declaring predicate "
                        f"{demand.semantic_predicate.predicate_id}"
                    ),
                )
            )
        else:
            for role_name, required_mode in demand.binding_policy.role_modes.items():
                declared: set[BindingCapability] = set()
                for provider in semantic_providers:
                    for capability in provider.semantic_capabilities.role_capabilities:
                        if capability.role_name == role_name:
                            declared.update(capability.capabilities)
                if not declared and not self.config.require_declared_binding_capabilities:
                    continue
                if not _binding_mode_supported(required_mode, declared):
                    failures.append(
                        FeasibilityFailure(
                            code=PruneCode.BINDING_CAPABILITY_MISSING,
                            reason=(
                                f"chain {chain.chain_id} does not declare {required_mode.value} "
                                f"for role {role_name}"
                            ),
                        )
                    )

        allowed_nodes = set(demand.hard_constraints.allowed_node_ids)
        allowed_regions = set(demand.hard_constraints.allowed_regions)
        selected_node_capabilities: set[str] = set()
        for step in alternative.step_placements:
            try:
                node = self.deployment.node(step.node_id)
            except DeploymentGraphError as exc:
                failures.append(
                    FeasibilityFailure(code=PruneCode.POLICY_VIOLATION, reason=str(exc))
                )
                continue
            if not node.available:
                failures.append(
                    FeasibilityFailure(
                        code=PruneCode.POLICY_VIOLATION,
                        reason=f"execution node {node.node_id} is unavailable",
                    )
                )
            if allowed_nodes and node.node_id not in allowed_nodes:
                failures.append(
                    FeasibilityFailure(
                        code=PruneCode.POLICY_VIOLATION,
                        reason=f"execution node {node.node_id} is not allowed",
                    )
                )
            if allowed_regions and node.region not in allowed_regions:
                failures.append(
                    FeasibilityFailure(
                        code=PruneCode.POLICY_VIOLATION,
                        reason=f"execution node {node.node_id} is outside allowed regions",
                    )
                )
            selected_node_capabilities.update(node.capabilities)
        available_capabilities = selected_node_capabilities | set(chain.capability_tags)
        missing_capabilities = set(demand.required_capabilities) - available_capabilities
        if missing_capabilities:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.REQUIRED_CAPABILITY_MISSING,
                    reason=f"physical plan lacks capabilities {sorted(missing_capabilities)}",
                )
            )

        if (
            demand.hard_constraints.maximum_transfer_bytes is not None
            and alternative.estimated_transfer_bytes
            > demand.hard_constraints.maximum_transfer_bytes
        ):
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.POLICY_VIOLATION,
                    reason="alternative exceeds maximum transfer bytes",
                )
            )
        for transfer in alternative.transfers:
            if transfer.source_node_id not in self.deployment.nodes or transfer.target_node_id not in self.deployment.nodes:
                failures.append(
                    FeasibilityFailure(
                        code=PruneCode.POLICY_VIOLATION,
                        reason="transfer references an unknown deployment node",
                    )
                )
                continue
            try:
                definition = self.providers.data_type(transfer.data_type)
            except ProviderRegistryError:
                continue
            if (
                definition.kind == "raw_sensor"
                and demand.hard_constraints.raw_data_must_remain_local
                and transfer.mode != TransferMode.LOCAL
            ):
                failures.append(
                    FeasibilityFailure(
                        code=PruneCode.POLICY_VIOLATION,
                        reason=f"raw input {transfer.data_type} would leave its source node",
                    )
                )

        quality_floor = max(
            self.config.minimum_quality_score,
            self.config.minimum_quality_by_predicate.get(
                demand.semantic_predicate.predicate_id, 0.0
            ),
        )
        if alternative.minimum_quality_score < quality_floor:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.QUALITY_FLOOR,
                    reason=(
                        f"quality {alternative.minimum_quality_score:.3f} is below "
                        f"required {quality_floor:.3f}"
                    ),
                )
            )

        parent_resources = parent.resource_map() if parent is not None else {}
        next_resources = _combine_resources(parent_resources, alternative)
        for node_id, footprint in next_resources.items():
            node = self.deployment.node(node_id)
            if (
                footprint.cpu_cores > node.capacity.cpu_cores
                or footprint.memory_mb > node.capacity.memory_mb
                or footprint.gpu_memory_mb > node.capacity.gpu_memory_mb
            ):
                failures.append(
                    FeasibilityFailure(
                        code=PruneCode.CAPACITY_EXCEEDED,
                        reason=(
                            f"node {node_id} would require cpu={footprint.cpu_cores}, "
                            f"memory={footprint.memory_mb} MB, gpu={footprint.gpu_memory_mb} MB"
                        ),
                    )
                )

        remaining_ms = int(
            (demand.deadline.latest_useful_completion - now).total_seconds() * 1000
        )
        if alternative.estimated_completion_ms > remaining_ms:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.DEADLINE_INFEASIBLE,
                    reason=(
                        f"estimated completion {alternative.estimated_completion_ms} ms "
                        f"exceeds remaining {remaining_ms} ms"
                    ),
                )
            )
        if parent is not None and parent.label.cost.deadline_slack_ms < 0:
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.DEADLINE_INFEASIBLE,
                    reason="parent label has already exceeded a covered demand deadline",
                )
            )

        produced_continuations = set(alternative.continuation_output_types)
        required_types = {
            requirement.artifact_type for requirement in demand.continuation_requirements
        }
        if not required_types.issubset(produced_continuations):
            failures.append(
                FeasibilityFailure(
                    code=PruneCode.CONTINUATION_INCOMPATIBLE,
                    reason=(
                        f"alternative does not produce continuation types "
                        f"{sorted(required_types - produced_continuations)}"
                    ),
                )
            )
        for requirement in demand.continuation_requirements:
            if requirement.compatible_consumer_families and not self.representations.supports(
                alternative.continuation_output_types,
                requirement.compatible_consumer_families,
            ):
                failures.append(
                    FeasibilityFailure(
                        code=PruneCode.CONTINUATION_INCOMPATIBLE,
                        reason=(
                            f"continuation {requirement.artifact_type} cannot serve consumers "
                            f"{sorted(requirement.compatible_consumer_families)}"
                        ),
                    )
                )

        # Preserve every independently meaningful failure while avoiding repeats.
        deduplicated: dict[tuple[PruneCode, str], FeasibilityFailure] = {}
        for failure in failures:
            deduplicated[(failure.code, failure.reason)] = failure
        return tuple(deduplicated.values())

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
                    input_artifact_ids=tuple(
                        sorted(
                            (
                                item.artifact_id
                                for item in alternative.external_inputs
                                if item.artifact_id is not None
                            ),
                            key=str,
                        )
                    ),
                    input_data_types=input_types,
                    output_data_types=output_types,
                    depends_on_step_ids=dependencies,
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

        resources = _combine_resources(parent.resource_map() if parent else {}, alternative)
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

    def rank_key(self, state: LabelSearchState | None) -> tuple:
        if state is None:
            return ()
        continuation_penalty = len(state.missing_desired_continuation_types)
        return (
            state.spatial_preference_penalty,
            state.label.cost.predicted_completion_ms,
            -state.label.cost.deadline_slack_ms,
            state.perishability_rank,
            state.label.cost.startup_cost_ms,
            state.total_cpu_cores,
            state.total_gpu_memory_mb,
            state.total_memory_mb,
            state.label.cost.transfer_bytes,
            continuation_penalty,
            -len(state.continuation_consumer_set),
            state.label_id,
        )

    def dominates(self, left: LabelSearchState, right: LabelSearchState) -> bool:
        if left.label.checkpoint_id != right.label.checkpoint_id:
            return False
        if set(left.label.covered_demand_ids) != set(right.label.covered_demand_ids):
            return False
        if not left.label.hard_constraints_satisfied or not left.label.quality_floor_satisfied:
            return False
        if not set(left.continuation_consumer_set).issuperset(
            right.continuation_consumer_set
        ):
            return False
        weak = (
            left.spatial_preference_penalty <= right.spatial_preference_penalty
            and left.label.cost.predicted_completion_ms
            <= right.label.cost.predicted_completion_ms
            and left.label.cost.deadline_slack_ms
            >= right.label.cost.deadline_slack_ms
            and left.label.cost.startup_cost_ms <= right.label.cost.startup_cost_ms
            and left.total_cpu_cores <= right.total_cpu_cores
            and left.total_memory_mb <= right.total_memory_mb
            and left.total_gpu_memory_mb <= right.total_gpu_memory_mb
            and left.label.cost.transfer_bytes <= right.label.cost.transfer_bytes
            and left.minimum_quality_score >= right.minimum_quality_score
        )
        if not weak:
            return False
        strict = (
            left.spatial_preference_penalty < right.spatial_preference_penalty
            or left.label.cost.predicted_completion_ms
            < right.label.cost.predicted_completion_ms
            or left.label.cost.deadline_slack_ms
            > right.label.cost.deadline_slack_ms
            or left.label.cost.startup_cost_ms < right.label.cost.startup_cost_ms
            or left.total_cpu_cores < right.total_cpu_cores
            or left.total_memory_mb < right.total_memory_mb
            or left.total_gpu_memory_mb < right.total_gpu_memory_mb
            or left.label.cost.transfer_bytes < right.label.cost.transfer_bytes
            or left.minimum_quality_score > right.minimum_quality_score
            or set(left.continuation_consumer_set)
            > set(right.continuation_consumer_set)
        )
        return strict

    def to_execution_plan(self, state: LabelSearchState | None) -> ExecutionPlan | None:
        if state is None:
            return None
        network_bytes_by_node: dict[str, int] = {}
        for step in state.label.steps:
            network_bytes_by_node[step.node_id] = (
                network_bytes_by_node.get(step.node_id, 0)
                + step.estimated_transfer_bytes
            )
        return ExecutionPlan(
            label_id=state.label_id,
            checkpoint_id=state.label.checkpoint_id,
            demand_ids=state.label.covered_demand_ids,
            steps=state.label.steps,
            reservations=tuple(
                ResourceReservation(
                    node_id=item.node_id,
                    cpu_cores=item.cpu_cores,
                    memory_mb=item.memory_mb,
                    gpu_memory_mb=item.gpu_memory_mb,
                    network_bytes=network_bytes_by_node.get(item.node_id, 0),
                )
                for item in state.node_resources
            ),
            status=PlanStatus.CANDIDATE,
            expires_at=state.expires_at,
        )

    def _demand_order(self, demands: Iterable[PredicateDemand]) -> tuple[UUID, ...]:
        return tuple(
            demand.demand_id
            for demand in sorted(
                demands,
                key=lambda demand: (
                    demand.deadline.latest_useful_completion,
                    demand.event_time_interval.end,
                    str(demand.demand_id),
                ),
            )
        )

    @staticmethod
    def _alternatives_by_demand(
        graph: PhysicalAlternativeGraph,
    ) -> dict[UUID, tuple[PhysicalAlternative, ...]]:
        result: dict[UUID, list[PhysicalAlternative]] = {}
        for alternative in graph.alternatives:
            result.setdefault(alternative.demand_id, []).append(alternative)
        return {
            demand_id: tuple(sorted(items, key=lambda item: item.alternative_id))
            for demand_id, items in result.items()
        }

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
        if earliest <= now + timedelta(milliseconds=self.config.near_expiry_horizon_ms):
            return 0
        if earliest <= demand.deadline.latest_useful_completion:
            return 1
        return 2

    def _deduplicate(
        self,
        states: Iterable[LabelSearchState],
        *,
        boundary_index: int,
        demand_id: UUID,
    ) -> tuple[tuple[LabelSearchState, ...], tuple[PruningRecord, ...]]:
        unique: dict[str, LabelSearchState] = {}
        pruning: list[PruningRecord] = []
        for state in sorted(states, key=lambda item: item.label_id):
            if state.label_id in unique:
                pruning.append(
                    PruningRecord(
                        boundary_index=boundary_index,
                        code=PruneCode.DUPLICATE_LABEL,
                        reason="an identical immutable label was already generated",
                        demand_id=demand_id,
                        label_id=state.label_id,
                    )
                )
                continue
            unique[state.label_id] = state
        return tuple(unique.values()), tuple(pruning)

    def _dominance_prune(
        self,
        states: Iterable[LabelSearchState],
        *,
        boundary_index: int,
        demand_id: UUID,
    ) -> tuple[tuple[LabelSearchState, ...], tuple[PruningRecord, ...]]:
        ordered = tuple(sorted(states, key=self.rank_key))
        removed: set[str] = set()
        pruning: list[PruningRecord] = []
        for candidate in ordered:
            if candidate.label_id in removed:
                continue
            for other in ordered:
                if candidate.label_id == other.label_id or other.label_id in removed:
                    continue
                if self.dominates(candidate, other):
                    removed.add(other.label_id)
                    pruning.append(
                        PruningRecord(
                            boundary_index=boundary_index,
                            code=PruneCode.DOMINATED,
                            reason=(
                                "another label is no later/no more costly and supports "
                                "a superset of continuation consumers"
                            ),
                            demand_id=demand_id,
                            label_id=other.label_id,
                            dominated_by_label_id=candidate.label_id,
                        )
                    )
        return (
            tuple(item for item in ordered if item.label_id not in removed),
            tuple(pruning),
        )

    def _oracle_compare(
        self,
        *,
        graph: PhysicalAlternativeGraph,
        demand_map: Mapping[UUID, PredicateDemand],
        demand_order: tuple[UUID, ...],
        selected: LabelSearchState | None,
        now: datetime,
        required_checkpoint_consumers: tuple[str, ...],
    ) -> OracleComparison:
        alternatives = self._alternatives_by_demand(graph)
        counts = [len(alternatives.get(demand_id, ())) for demand_id in demand_order]
        combinations = prod(counts) if counts else 0
        if combinations > self.config.oracle_max_combinations:
            return OracleComparison(
                status=OracleStatus.SKIPPED_LIMIT,
                combinations_considered=0,
                selected_label_id=(selected.label_id if selected else None),
                reason=(
                    f"{combinations} combinations exceed oracle limit "
                    f"{self.config.oracle_max_combinations}"
                ),
            )
        feasible: list[LabelSearchState] = []
        considered = 0
        if all(count > 0 for count in counts):
            for combination in product(
                *(alternatives[demand_id] for demand_id in demand_order)
            ):
                considered += 1
                state: LabelSearchState | None = None
                valid = True
                for demand_id, alternative in zip(demand_order, combination, strict=True):
                    demand = demand_map[demand_id]
                    if self.check_extension(
                        state,
                        alternative,
                        demand,
                        demand_map=demand_map,
                        now=now,
                    ):
                        valid = False
                        break
                    state = self.extend_label(
                        state,
                        alternative,
                        demand,
                        demand_map=demand_map,
                        now=now,
                    )
                if not valid or state is None:
                    continue
                if required_checkpoint_consumers and not set(
                    required_checkpoint_consumers
                ).issubset(state.continuation_consumer_set):
                    continue
                feasible.append(state)
        if not feasible:
            return OracleComparison(
                status=OracleStatus.NO_FEASIBLE_PLAN,
                combinations_considered=considered,
                selected_label_id=(selected.label_id if selected else None),
                reason="exhaustive enumeration found no feasible complete label",
            )
        oracle = min(feasible, key=self.rank_key)
        if selected is not None and self.rank_key(selected) == self.rank_key(oracle):
            status = OracleStatus.MATCHED
        else:
            status = OracleStatus.GAP
        return OracleComparison(
            status=status,
            combinations_considered=considered,
            oracle_label_id=oracle.label_id,
            selected_label_id=(selected.label_id if selected else None),
            completion_gap_ms=(
                None
                if selected is None
                else selected.label.cost.predicted_completion_ms
                - oracle.label.cost.predicted_completion_ms
            ),
            startup_gap_ms=(
                None
                if selected is None
                else selected.label.cost.startup_cost_ms
                - oracle.label.cost.startup_cost_ms
            ),
            transfer_gap_bytes=(
                None
                if selected is None
                else selected.label.cost.transfer_bytes
                - oracle.label.cost.transfer_bytes
            ),
            reason=(
                "beam result matches the exhaustive oracle"
                if status == OracleStatus.MATCHED
                else "beam result differs from the exhaustive oracle over the bounded graph"
            ),
        )


def _binding_mode_supported(
    required: BindingCapability,
    declared: set[BindingCapability],
) -> bool:
    if required in declared:
        return True
    if required in (BindingCapability.INTRODUCE, BindingCapability.VALIDATE):
        return BindingCapability.INTRODUCE_OR_VALIDATE in declared
    if required == BindingCapability.INTRODUCE_OR_VALIDATE:
        return (
            BindingCapability.INTRODUCE_OR_VALIDATE in declared
            or {
                BindingCapability.INTRODUCE,
                BindingCapability.VALIDATE,
            }.issubset(declared)
        )
    return False


def _combine_resources(
    parent_resources: Mapping[str, NodeResourceFootprint],
    alternative: PhysicalAlternative,
) -> dict[str, NodeResourceFootprint]:
    values: dict[str, tuple[float, int, int]] = {
        node_id: (item.cpu_cores, item.memory_mb, item.gpu_memory_mb)
        for node_id, item in parent_resources.items()
    }
    for step in alternative.step_placements:
        # A compatible active provider is already consuming its resources; Phase 4
        # charges only incremental capacity.  Phase 5 will attach a lease.
        if step.reused_provider_instance_id is not None:
            continue
        cpu, memory, gpu = values.get(step.node_id, (0.0, 0, 0))
        values[step.node_id] = (
            cpu + step.cpu_cores,
            memory + step.memory_mb,
            gpu + step.gpu_memory_mb,
        )
    return {
        node_id: NodeResourceFootprint(
            node_id=node_id,
            cpu_cores=cpu,
            memory_mb=memory,
            gpu_memory_mb=gpu,
        )
        for node_id, (cpu, memory, gpu) in values.items()
    }
