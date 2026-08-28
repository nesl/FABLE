"""Hard feasibility checks for extending a physical-plan label."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from uuid import UUID

from fable.common.enums import BindingCapability
from fable.common.schemas import PredicateDemand
from fable.common.time import RAW_BUFFER_ALIGNMENT_TOLERANCE
from fable.planning.artifact_catalog import ArtifactCatalog, ArtifactCatalogError
from fable.planning.deployment import DeploymentGraph, DeploymentGraphError
from fable.planning.models import ExternalInputKind, PhysicalAlternative, TransferMode
from fable.planning.provider_registry import ProviderRegistry, ProviderRegistryError
from fable.planning.representation import RepresentationCompatibility
from fable.planning.search.config import BeamSearchConfig
from fable.planning.search.resources import combine_resources
from fable.planning.search_models import FeasibilityFailure, LabelSearchState, PruneCode


class PlanFeasibilityChecker:
    """Apply all hard feasibility constraints to one label extension."""

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry,
        artifact_catalog: ArtifactCatalog,
        deployment: DeploymentGraph,
        config: BeamSearchConfig,
        representation_compatibility: RepresentationCompatibility,
    ) -> None:
        self.providers = provider_registry
        self.artifacts = artifact_catalog
        self.deployment = deployment
        self.config = config
        self.representations = representation_compatibility

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
                                demand.event_time_interval,
                                tolerance=RAW_BUFFER_ALIGNMENT_TOLERANCE,
                            )
                        ):
                            failures.append(
                                FeasibilityFailure(
                                    code=PruneCode.EVENT_TIME_UNAVAILABLE,
                                    reason=(
                                        f"source {item.source_id} does not retain demand interval "
                                        f"{demand.event_time_interval.start.isoformat()}.."
                                        f"{demand.event_time_interval.end.isoformat()}; raw buffer="
                                        f"{source.raw_buffer_interval}"
                                    ),
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
            next_resources = combine_resources(
                parent_resources,
                alternative,
            )
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
            or BindingCapability.INTRODUCE in declared
            or BindingCapability.VALIDATE in declared
        )
    return False


__all__ = ["PlanFeasibilityChecker"]
