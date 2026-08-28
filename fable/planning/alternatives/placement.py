"""Provider placement and transfer enumeration for physical alternatives."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from itertools import product
import json
from typing import Any, Callable

from fable.common.enums import ArtifactAccessMode
from fable.common.ids import deterministic_id
from fable.common.schemas import PredicateDemand, ProviderContract
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import (
    ActiveProviderInstance,
    DataTransfer,
    ExternalInputKind,
    ExternalInputRealization,
    PrunedAlternative,
    StepPlacement,
    TransferMode,
)
from fable.planning.provider_registry import ProviderRegistry, ProviderRegistryError
from fable.planning.alternatives.config import AlternativeBuildConfig
from fable.planning.alternatives.internal import (
    DataRef,
    PlacementState,
    deduplicate_states,
    estimated_size,
)


class PlacementEnumerator:
    """Choose feasible provider nodes and data-movement modes for a chain."""

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry,
        deployment: DeploymentGraph,
        config: AlternativeBuildConfig,
        active_providers: tuple[ActiveProviderInstance, ...],
        placement_eligible: Callable[[str, str], bool] | None = None,
    ) -> None:
        self.providers = provider_registry
        self.deployment = deployment
        self.config = config
        self.active_providers = active_providers
        self.placement_eligible = placement_eligible

    def _placement_states(
            self,
            *,
            demand: PredicateDemand,
            chain_id: str,
            assignment: tuple[ExternalInputRealization, ...],
        ) -> tuple[tuple[PlacementState, ...], tuple[PrunedAlternative, ...]]:
            chain = self.providers.chain(chain_id)
            external_refs = {
                f"external.{item.input_name}": DataRef(
                    source_ref=f"external.{item.input_name}",
                    data_type=item.data_type,
                    node_id=item.node_id or "",
                    bytes=item.bytes,
                    source_id=item.source_id,
                    artifact_id=item.artifact_id,
                    access_modes=item.access_modes,
                )
                for item in assignment
                if item.kind != ExternalInputKind.OMITTED_OPTIONAL
            }
            states: list[PlacementState] = [
                PlacementState(steps=(), transfers=(), outputs=(), resource_usage=())
            ]
            pruned: list[PrunedAlternative] = []

            for step in chain.steps:
                next_states: list[PlacementState] = []
                rejection_diagnostics: list[dict[str, Any]] = []
                provider = self.providers.provider(step.provider_id)
                for state in states:
                    output_map = state.output_map()
                    input_refs: dict[str, DataRef] = {}
                    unresolved = False
                    for port_name, source_ref in step.bindings.items():
                        ref = external_refs.get(source_ref) or output_map.get(source_ref)
                        if ref is None:
                            # Optional external input may have been omitted.
                            port = self.providers.provider_input_ports(step.provider_id).get(port_name)
                            if port is not None and not port.required:
                                continue
                            unresolved = True
                            break
                        input_refs[port_name] = ref
                    if unresolved:
                        rejection_diagnostics.append({"reason": "required_input_unresolved"})
                        continue

                    candidate_nodes = self._candidate_nodes_for_step(demand, provider, input_refs)
                    local_internal_prefix = (
                        "sensor_prefix_local" in chain.capability_tags
                        and step.step_id
                        in {"track", "crops", "track_a", "track_b", "crops_a", "crops_b"}
                    )
                    if (
                        self.config.require_internal_step_colocation
                        or "source_local_pipeline" in chain.capability_tags
                        or local_internal_prefix
                    ):
                        internal_node_ids = {
                            ref.node_id
                            for ref in input_refs.values()
                            if not ref.source_ref.startswith("external.")
                        }
                        if internal_node_ids:
                            before_colocation = tuple(node.node_id for node in candidate_nodes)
                            candidate_nodes = tuple(
                                node
                                for node in candidate_nodes
                                if node.node_id in internal_node_ids
                            )
                            if not candidate_nodes:
                                rejection_diagnostics.append(
                                    {
                                        "reason": "internal_colocation",
                                        "required_nodes": sorted(internal_node_ids),
                                        "candidates_before": list(before_colocation),
                                    }
                                )
                    if not candidate_nodes:
                        rejection_diagnostics.append(
                            self._candidate_rejection_diagnostic(
                                demand=demand,
                                provider=provider,
                                input_refs=input_refs,
                            )
                        )
                    for node in candidate_nodes[: self.config.max_candidate_nodes_per_step]:
                        try:
                            profile = self.providers.profile(provider.provider_id, node.node_class)
                        except ProviderRegistryError as exc:
                            rejection_diagnostics.append(
                                {"node": node.node_id, "reason": "missing_profile", "detail": str(exc)}
                            )
                            continue
                        usage = state.usage_map()
                        cpu, memory, gpu = usage.get(node.node_id, (0.0, 0, 0))
                        next_usage = (
                            cpu + profile.cpu_cores,
                            memory + profile.memory_mb,
                            gpu + profile.gpu_memory_mb,
                        )
                        if (
                            next_usage[0] > node.capacity.cpu_cores
                            or next_usage[1] > node.capacity.memory_mb
                            or next_usage[2] > node.capacity.gpu_memory_mb
                        ):
                            rejection_diagnostics.append(
                                {
                                    "node": node.node_id,
                                    "reason": "capacity",
                                    "usage_before": [cpu, memory, gpu],
                                    "required": [profile.cpu_cores, profile.memory_mb, profile.gpu_memory_mb],
                                    "capacity": [
                                        node.capacity.cpu_cores,
                                        node.capacity.memory_mb,
                                        node.capacity.gpu_memory_mb,
                                    ],
                                }
                            )
                            continue
                        transfer_options = self._transfer_options(
                            demand=demand,
                            provider=provider,
                            target_node_id=node.node_id,
                            input_refs=input_refs,
                            target_step_id=step.step_id,
                        )
                        if transfer_options is None:
                            rejection_diagnostics.append(
                                {
                                    "node": node.node_id,
                                    "reason": "no_transfer_path",
                                    "inputs": self._input_diagnostic(input_refs),
                                }
                            )
                            continue
                        for transfers in transfer_options:
                            reused_provider = self._active_provider_instance(
                                provider_id=provider.provider_id,
                                node_id=node.node_id,
                            )
                            placements = (
                                *state.steps,
                                StepPlacement(
                                    step_id=step.step_id,
                                    provider_id=provider.provider_id,
                                    node_id=node.node_id,
                                    node_class=node.node_class,
                                    startup_ms=0 if reused_provider is not None else profile.startup_ms,
                                    execution_ms=(
                                        int(round(
                                            profile.execution_ms
                                            * self.config.node_execution_time_multipliers.get(
                                                node.node_id, 1.0
                                            )
                                        ))
                                        + self.config.node_queue_delay_ms.get(node.node_id, 0)
                                    ),
                                    cpu_cores=profile.cpu_cores,
                                    memory_mb=profile.memory_mb,
                                    gpu_memory_mb=profile.gpu_memory_mb,
                                    quality_score=profile.quality_score,
                                    reused_provider_instance_id=(
                                        reused_provider.provider_instance_id
                                        if reused_provider is not None
                                        else None
                                    ),
                                ),
                            )
                            outputs = dict(output_map)
                            for port_name, port in self.providers.provider_output_ports(provider.provider_id).items():
                                source_ref = f"{step.step_id}.{port_name}"
                                outputs[source_ref] = DataRef(
                                    source_ref=source_ref,
                                    data_type=port.data_type,
                                    node_id=node.node_id,
                                    bytes=estimated_size(port.data_type),
                                )
                            usage[node.node_id] = next_usage
                            next_states.append(
                                PlacementState(
                                    steps=placements,
                                    transfers=(*state.transfers, *transfers),
                                    outputs=tuple(sorted(outputs.items(), key=lambda item: item[0])),
                                    resource_usage=tuple(
                                        sorted(
                                            (
                                                node_id,
                                                values[0],
                                                values[1],
                                                values[2],
                                            )
                                            for node_id, values in usage.items()
                                        )
                                    ),
                                )
                            )
                            if len(next_states) >= self.config.max_placement_variants_per_assignment:
                                break
                        if len(next_states) >= self.config.max_placement_variants_per_assignment:
                            break
                    if len(next_states) >= self.config.max_placement_variants_per_assignment:
                        break
                states = deduplicate_states(next_states)[: self.config.max_placement_variants_per_assignment]
                if not states:
                    pruned.append(
                        PrunedAlternative(
                            candidate_id=deterministic_id(
                                "candidate",
                                {
                                    "demand": demand.demand_id,
                                    "chain": chain_id,
                                    "step": step.step_id,
                                    "assignment": assignment,
                                },
                            ),
                            demand_id=demand.demand_id,
                            chain_id=chain_id,
                            code="STEP_UNPLACEABLE",
                            reason=(
                                f"no compatible placement/data path for step {step.step_id}; "
                                f"diagnostic={json.dumps(rejection_diagnostics[:12], sort_keys=True)}"
                            ),
                        )
                    )
                    break
            return tuple(states), tuple(pruned)

    @staticmethod
    def _input_diagnostic(input_refs: Mapping[str, DataRef]) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "source_ref": ref.source_ref,
                "data_type": ref.data_type,
                "node": ref.node_id,
                "source_id": ref.source_id,
                "access_modes": sorted(str(mode) for mode in ref.access_modes),
            }
            for name, ref in sorted(input_refs.items())
        }

    def _candidate_rejection_diagnostic(
        self,
        *,
        demand: PredicateDemand,
        provider: ProviderContract,
        input_refs: Mapping[str, DataRef],
    ) -> dict[str, Any]:
        """Explain candidate filtering without changing placement decisions."""
        allowed_ids = set(demand.hard_constraints.allowed_node_ids)
        allowed_regions = set(demand.hard_constraints.allowed_regions)
        forced_raw_nodes = {
            ref.node_id
            for ref in input_refs.values()
            if self.providers.data_type(ref.data_type).kind == "raw_sensor"
            and demand.hard_constraints.raw_data_must_remain_local
        }
        nodes: dict[str, list[str]] = {}
        for node in sorted(self.deployment.nodes.values(), key=lambda item: item.node_id):
            reasons: list[str] = []
            if not node.available:
                reasons.append("unavailable")
            if allowed_ids and node.node_id not in allowed_ids:
                reasons.append("not_allowed_node")
            if allowed_regions and node.region not in allowed_regions:
                reasons.append("not_allowed_region")
            if len(forced_raw_nodes) == 1 and node.node_id not in forced_raw_nodes:
                reasons.append("raw_locality")
            if provider.eligible_node_classes and node.node_class not in provider.eligible_node_classes:
                reasons.append("ineligible_node_class")
            if self.placement_eligible is not None and not self.placement_eligible(
                node.node_id, provider.provider_id
            ):
                reasons.append("runtime_ineligible")
            nodes[node.node_id] = reasons or ["candidate"]
        return {
            "reason": "no_candidate_nodes",
            "provider": provider.provider_id,
            "forced_raw_nodes": sorted(forced_raw_nodes),
            "allowed_node_ids": sorted(allowed_ids),
            "allowed_regions": sorted(allowed_regions),
            "eligible_node_classes": sorted(provider.eligible_node_classes),
            "inputs": self._input_diagnostic(input_refs),
            "nodes": nodes,
        }

    def _active_provider_instance(
            self,
            *,
            provider_id: str,
            node_id: str,
        ) -> ActiveProviderInstance | None:
            for instance in self.active_providers:
                if instance.provider_id == provider_id and instance.node_id == node_id:
                    return instance
            return None

    def _candidate_nodes_for_step(
            self,
            demand: PredicateDemand,
            provider: ProviderContract,
            input_refs: Mapping[str, DataRef],
        ) -> tuple[Any, ...]:
            forced_raw_nodes = {
                ref.node_id
                for ref in input_refs.values()
                if self.providers.data_type(ref.data_type).kind == "raw_sensor"
                and demand.hard_constraints.raw_data_must_remain_local
            }
            if len(forced_raw_nodes) > 1:
                return ()
            nodes = self.deployment.candidate_nodes(
                allowed_node_ids=demand.hard_constraints.allowed_node_ids,
                allowed_regions=demand.hard_constraints.allowed_regions,
            )
            if forced_raw_nodes:
                forced = next(iter(forced_raw_nodes))
                nodes = tuple(node for node in nodes if node.node_id == forced)
            if provider.eligible_node_classes:
                nodes = tuple(node for node in nodes if node.node_class in provider.eligible_node_classes)
            if self.placement_eligible is not None:
                nodes = tuple(
                    node
                    for node in nodes
                    if self.placement_eligible(node.node_id, provider.provider_id)
                )

            locality_counts: dict[str, int] = defaultdict(int)
            for ref in input_refs.values():
                locality_counts[ref.node_id] += 1
            return tuple(
                sorted(
                    nodes,
                    key=lambda node: (
                        -locality_counts.get(node.node_id, 0),
                        # After the source-local placement, preserve trusted
                        # aggregation/offload candidates before unrelated
                        # sensor peers.  Otherwise a bounded candidate list
                        # can be filled by alphabetically earlier cameras and
                        # erase the only executable fallback when the source
                        # device loses capacity.  Locality remains the primary
                        # key, so nominal source-local behavior is unchanged.
                        0 if node.node_class in {"edge", "server"} else 1,
                        node.node_id,
                    ),
                )
            )

    def _transfer_options(
            self,
            *,
            demand: PredicateDemand,
            provider: ProviderContract,
            target_node_id: str,
            input_refs: Mapping[str, DataRef],
            target_step_id: str,
        ) -> tuple[tuple[DataTransfer, ...], ...] | None:
            per_port: list[tuple[DataTransfer, ...]] = []
            accepted = set(provider.execution_capabilities.accepted_input_access)
            for port_name, ref in sorted(input_refs.items()):
                if ref.node_id == target_node_id:
                    if ArtifactAccessMode.LOCAL not in accepted:
                        return None
                    per_port.append(
                        (
                            DataTransfer(
                                source_ref=ref.source_ref,
                                target_step_id=target_step_id,
                                target_port=port_name,
                                data_type=ref.data_type,
                                source_node_id=ref.node_id,
                                target_node_id=target_node_id,
                                mode=TransferMode.LOCAL,
                            ),
                        )
                    )
                    continue

                definition = self.providers.data_type(ref.data_type)
                if definition.kind == "raw_sensor" and demand.hard_constraints.raw_data_must_remain_local:
                    return None
                path = self.deployment.shortest_path(ref.node_id, target_node_id)
                if path is None:
                    return None
                modes: list[DataTransfer] = []
                if (
                    self.config.allow_transfer
                    and definition.transferable in (True, "policy_dependent")
                    and ArtifactAccessMode.TRANSFERRED in accepted
                    and (
                        not ref.access_modes
                        or ArtifactAccessMode.TRANSFERRED in ref.access_modes
                        or ref.artifact_id is None
                    )
                ):
                    modes.append(
                        DataTransfer(
                            source_ref=ref.source_ref,
                            target_step_id=target_step_id,
                            target_port=port_name,
                            data_type=ref.data_type,
                            source_node_id=ref.node_id,
                            target_node_id=target_node_id,
                            mode=TransferMode.TRANSFER,
                            bytes=ref.bytes,
                            estimated_ms=self.deployment.estimate_transfer_ms(path, ref.bytes),
                            path_node_ids=path.node_ids,
                        )
                    )
                if (
                    self.config.allow_remote_reference
                    and definition.remote_reference_allowed
                    and ArtifactAccessMode.REMOTE_REFERENCE in accepted
                    and (
                        not ref.access_modes
                        or ArtifactAccessMode.REMOTE_REFERENCE in ref.access_modes
                        or ref.artifact_id is None
                    )
                ):
                    modes.append(
                        DataTransfer(
                            source_ref=ref.source_ref,
                            target_step_id=target_step_id,
                            target_port=port_name,
                            data_type=ref.data_type,
                            source_node_id=ref.node_id,
                            target_node_id=target_node_id,
                            mode=TransferMode.REMOTE_REFERENCE,
                            bytes=0,
                            estimated_ms=path.latency_ms,
                            path_node_ids=path.node_ids,
                        )
                    )
                if not modes:
                    return None
                per_port.append(tuple(modes))

            combinations = []
            for combo in product(*per_port):
                combinations.append(tuple(combo))
                if len(combinations) >= self.config.max_placement_variants_per_assignment:
                    break
            return tuple(combinations)


__all__ = ["PlacementEnumerator"]
