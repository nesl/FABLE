"""Construct checkpoint-bounded physical alternative graphs.

This phase enumerates and annotates feasible alternatives; it deliberately does
not rank them.  Phase 4 consumes this graph with bounded label-driven search.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from math import ceil
from typing import Any
from uuid import UUID

from pydantic import Field

from fable.common.base import FableModel
from fable.common.enums import ArtifactAccessMode, ExecutionMode, ProviderPortKind
from fable.common.ids import deterministic_id
from fable.common.schemas import ArtifactRef, PredicateDemand, ProviderContract
from fable.common.time import ensure_utc, utc_now

from .artifact_catalog import ArtifactCatalog
from .deployment import DeploymentGraph
from .models import (
    ActiveProviderInstance,
    AlternativeEdgeKind,
    AlternativeGraphEdge,
    AlternativeGraphNode,
    AlternativeNodeKind,
    DataTransfer,
    ExternalInputKind,
    ExternalInputRealization,
    PhysicalAlternative,
    PhysicalAlternativeGraph,
    PrunedAlternative,
    StepPlacement,
    TransferMode,
)
from .provider_registry import ProviderRegistry, ProviderRegistryError


class AlternativeGraphError(ValueError):
    """Raised when physical alternatives cannot be constructed safely."""


class AlternativeBuildConfig(FableModel):
    max_external_assignments_per_chain: int = Field(default=32, ge=1)
    max_placement_variants_per_assignment: int = Field(default=24, ge=1)
    max_total_alternatives: int = Field(default=128, ge=1)
    max_alternatives_per_chain: int = Field(default=32, ge=1)
    max_candidate_nodes_per_step: int = Field(default=2, ge=1)
    allow_remote_reference: bool = True
    allow_transfer: bool = True
    default_queue_ms: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class _DataRef:
    source_ref: str
    data_type: str
    node_id: str
    bytes: int
    source_id: str | None = None
    artifact_id: UUID | None = None
    access_modes: tuple[ArtifactAccessMode, ...] = (
        ArtifactAccessMode.LOCAL,
        ArtifactAccessMode.TRANSFERRED,
        ArtifactAccessMode.REMOTE_REFERENCE,
    )


@dataclass(frozen=True)
class _PlacementState:
    steps: tuple[StepPlacement, ...]
    transfers: tuple[DataTransfer, ...]
    outputs: tuple[tuple[str, _DataRef], ...]
    resource_usage: tuple[tuple[str, float, int, int], ...]

    def output_map(self) -> dict[str, _DataRef]:
        return dict(self.outputs)

    def usage_map(self) -> dict[str, tuple[float, int, int]]:
        return {
            node_id: (cpu, memory, gpu)
            for node_id, cpu, memory, gpu in self.resource_usage
        }


_SIZE_DEFAULTS: dict[str, int] = {
    "raw_video_frames.v1": 20_000_000,
    "camera_calibration.v1": 16_000,
    "route_graph.v1": 128_000,
    "detection_set.v1": 500_000,
    "track_set.v1": 250_000,
    "projected_track_set.v1": 300_000,
    "image_crop_set.v1": 5_000_000,
    "vehicle_reid_embedding_set.v1": 16_000,
    "canonical_entity_map.v1": 16_000,
    "pair_trajectory.v1": 128_000,
    "track_summary.v1": 96_000,
    "audio_segment.v1": 2_000_000,
    "audio_event_set.v1": 32_000,
    "predicate_match.v1": 4_000,
}


class PhysicalAlternativeGraphBuilder:
    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry,
        artifact_catalog: ArtifactCatalog,
        deployment: DeploymentGraph,
        config: AlternativeBuildConfig | None = None,
        active_providers: Iterable[ActiveProviderInstance] = (),
    ) -> None:
        self.providers = provider_registry
        self.artifacts = artifact_catalog
        self.deployment = deployment
        self.config = config or AlternativeBuildConfig()
        self.active_providers = tuple(
            sorted(
                (item for item in active_providers if item.available),
                key=lambda item: item.provider_instance_id,
            )
        )

    def build(
        self,
        demands: Iterable[PredicateDemand],
        *,
        now: datetime | None = None,
    ) -> PhysicalAlternativeGraph:
        observed_now = ensure_utc(now or utc_now())
        ordered_demands = tuple(sorted(demands, key=lambda demand: str(demand.demand_id)))
        if not ordered_demands:
            raise AlternativeGraphError("at least one predicate demand is required")

        nodes: dict[str, AlternativeGraphNode] = {}
        edges: dict[str, AlternativeGraphEdge] = {}
        alternatives: list[PhysicalAlternative] = []
        pruned: list[PrunedAlternative] = []

        for demand in ordered_demands:
            if observed_now >= demand.deadline.latest_useful_completion:
                pruned.append(
                    PrunedAlternative(
                        candidate_id=deterministic_id("candidate", {"demand": demand.demand_id, "expired": True}),
                        demand_id=demand.demand_id,
                        chain_id="none",
                        code="DEADLINE_EXPIRED",
                        reason="demand latest useful completion has already passed",
                    )
                )
                continue
            chains = self.providers.candidate_chains(demand)
            if not chains:
                pruned.append(
                    PrunedAlternative(
                        candidate_id=deterministic_id("candidate", {"demand": demand.demand_id, "no_chain": True}),
                        demand_id=demand.demand_id,
                        chain_id="none",
                        code="NO_PROVIDER_CHAIN",
                        reason=f"no chain implements {demand.semantic_predicate.predicate_id}",
                    )
                )
                continue

            for chain in chains:
                chain_alternative_count = 0
                assignments, assignment_pruned = self._external_assignments(
                    demand, chain.chain_id, now=observed_now
                )
                pruned.extend(assignment_pruned)
                for assignment in assignments:
                    placement_states, placement_pruned = self._placement_states(
                        demand=demand,
                        chain_id=chain.chain_id,
                        assignment=assignment,
                    )
                    pruned.extend(placement_pruned)
                    for state in placement_states:
                        candidate_id = deterministic_id(
                            "candidate",
                            {
                                "demand_id": demand.demand_id,
                                "chain_id": chain.chain_id,
                                "external_inputs": assignment,
                                "placements": state.steps,
                                "transfers": state.transfers,
                            },
                            length=32,
                        )
                        required_types = {
                            requirement.artifact_type for requirement in demand.continuation_requirements
                        }
                        available_continuations = set(chain.continuation_output_types)
                        if not required_types.issubset(available_continuations):
                            pruned.append(
                                PrunedAlternative(
                                    candidate_id=candidate_id,
                                    demand_id=demand.demand_id,
                                    chain_id=chain.chain_id,
                                    code="CONTINUATION_UNAVAILABLE",
                                    reason=(
                                        f"chain does not produce required continuation types "
                                        f"{sorted(required_types - available_continuations)}"
                                    ),
                                )
                            )
                            continue

                        total_transfer = sum(transfer.bytes for transfer in state.transfers)
                        if (
                            demand.hard_constraints.maximum_transfer_bytes is not None
                            and total_transfer > demand.hard_constraints.maximum_transfer_bytes
                        ):
                            pruned.append(
                                PrunedAlternative(
                                    candidate_id=candidate_id,
                                    demand_id=demand.demand_id,
                                    chain_id=chain.chain_id,
                                    code="TRANSFER_BUDGET_EXCEEDED",
                                    reason="estimated transfer exceeds the demand hard limit",
                                )
                            )
                            continue

                        completion_ms = (
                            self.config.default_queue_ms
                            + sum(step.startup_ms + step.execution_ms for step in state.steps)
                            + sum(transfer.estimated_ms for transfer in state.transfers)
                        )
                        remaining_ms = int(
                            (demand.deadline.latest_useful_completion - observed_now).total_seconds() * 1000
                        )
                        if completion_ms > remaining_ms:
                            pruned.append(
                                PrunedAlternative(
                                    candidate_id=candidate_id,
                                    demand_id=demand.demand_id,
                                    chain_id=chain.chain_id,
                                    code="DEADLINE_INFEASIBLE",
                                    reason=f"estimated {completion_ms} ms exceeds remaining {remaining_ms} ms",
                                )
                            )
                            continue

                        alternative_id = deterministic_id(
                            "alt",
                            {
                                "candidate_id": candidate_id,
                                "checkpoint_id": demand.checkpoint_id,
                            },
                            length=32,
                        )
                        alt_nodes, alt_edges = self._materialize_graph(
                            alternative_id=alternative_id,
                            demand=demand,
                            chain_id=chain.chain_id,
                            assignment=assignment,
                            state=state,
                        )
                        nodes.update({node.node_id: node for node in alt_nodes})
                        edges.update({edge.edge_id: edge for edge in alt_edges})
                        execution_mode = (
                            ExecutionMode.LIVE
                            if any(item.kind == ExternalInputKind.LIVE_SOURCE for item in assignment)
                            else ExecutionMode.RETROSPECTIVE
                        )
                        result_type = chain.output_types["result"]
                        spatial_penalty, spatial_reason = _spatial_preference(
                            demand, assignment
                        )
                        alternatives.append(
                            PhysicalAlternative(
                                alternative_id=alternative_id,
                                demand_id=demand.demand_id,
                                checkpoint_id=demand.checkpoint_id,
                                chain_id=chain.chain_id,
                                execution_mode=execution_mode,
                                external_inputs=assignment,
                                step_placements=state.steps,
                                transfers=state.transfers,
                                result_output_type=result_type,
                                continuation_output_types=chain.continuation_output_types,
                                estimated_completion_ms=completion_ms,
                                estimated_transfer_bytes=total_transfer,
                                minimum_quality_score=min(step.quality_score for step in state.steps),
                                graph_node_ids=tuple(node.node_id for node in alt_nodes),
                                graph_edge_ids=tuple(edge.edge_id for edge in alt_edges),
                                spatial_preference_penalty=spatial_penalty,
                                spatial_preference_reason=spatial_reason,
                            )
                        )
                        chain_alternative_count += 1
                        if (
                            len(alternatives) >= self.config.max_total_alternatives
                            or chain_alternative_count >= self.config.max_alternatives_per_chain
                        ):
                            break
                    if (
                        len(alternatives) >= self.config.max_total_alternatives
                        or chain_alternative_count >= self.config.max_alternatives_per_chain
                    ):
                        break
                if len(alternatives) >= self.config.max_total_alternatives:
                    break

        graph_id = deterministic_id(
            "physical_graph",
            {
                "demand_ids": [demand.demand_id for demand in ordered_demands],
                "checkpoint_ids": sorted({str(demand.checkpoint_id) for demand in ordered_demands}),
                "alternative_ids": sorted(alt.alternative_id for alt in alternatives),
            },
            length=32,
        )
        return PhysicalAlternativeGraph(
            graph_id=graph_id,
            checkpoint_ids=tuple(
                sorted({demand.checkpoint_id for demand in ordered_demands}, key=str)
            ),
            demand_ids=tuple(demand.demand_id for demand in ordered_demands),
            nodes=tuple(sorted(nodes.values(), key=lambda node: node.node_id)),
            edges=tuple(sorted(edges.values(), key=lambda edge: edge.edge_id)),
            alternatives=tuple(sorted(alternatives, key=lambda alt: alt.alternative_id)),
            pruned=tuple(sorted(pruned, key=lambda item: (str(item.demand_id), item.chain_id, item.candidate_id))),
            built_at=observed_now,
        )

    def _external_assignments(
        self,
        demand: PredicateDemand,
        chain_id: str,
        *,
        now: datetime,
    ) -> tuple[tuple[tuple[ExternalInputRealization, ...], ...], tuple[PrunedAlternative, ...]]:
        chain = self.providers.chain(chain_id)
        choices: list[tuple[ExternalInputRealization, ...]] = []
        pruned: list[PrunedAlternative] = []
        for external in chain.external_inputs:
            candidates = self._input_candidates(
                demand, chain_id, external.name, external.data_type, now=now
            )
            if external.optional:
                candidates = (
                    *candidates,
                    ExternalInputRealization(
                        input_name=external.name,
                        data_type=external.data_type,
                        kind=ExternalInputKind.OMITTED_OPTIONAL,
                    ),
                )
            if not candidates:
                pruned.append(
                    PrunedAlternative(
                        candidate_id=deterministic_id(
                            "candidate",
                            {"demand": demand.demand_id, "chain": chain_id, "input": external.name},
                        ),
                        demand_id=demand.demand_id,
                        chain_id=chain_id,
                        code="MISSING_EXTERNAL_INPUT",
                        reason=f"no compatible source or artifact for {external.name}:{external.data_type}",
                    )
                )
                return (), tuple(pruned)
            choices.append(candidates)

        assignments: list[tuple[ExternalInputRealization, ...]] = []
        for combination in product(*choices):
            ordered = tuple(sorted(combination, key=lambda item: item.input_name))
            if not self._assignment_compatible(ordered):
                continue
            assignments.append(ordered)
            if len(assignments) >= self.config.max_external_assignments_per_chain:
                break
        if assignments:
            retrospective = [
                item
                for item in assignments
                if not any(part.kind == ExternalInputKind.LIVE_SOURCE for part in item)
            ]
            live = [
                item
                for item in assignments
                if any(part.kind == ExternalInputKind.LIVE_SOURCE for part in item)
            ]
            balanced: list[tuple[ExternalInputRealization, ...]] = []
            for index in range(max(len(retrospective), len(live))):
                if index < len(retrospective):
                    balanced.append(retrospective[index])
                if index < len(live):
                    balanced.append(live[index])
            assignments = balanced
        if not assignments:
            pruned.append(
                PrunedAlternative(
                    candidate_id=deterministic_id(
                        "candidate", {"demand": demand.demand_id, "chain": chain_id, "assignment": "none"}
                    ),
                    demand_id=demand.demand_id,
                    chain_id=chain_id,
                    code="INCOMPATIBLE_EXTERNAL_INPUTS",
                    reason="external inputs exist individually but cannot form one compatible assignment",
                )
            )
        return tuple(assignments), tuple(pruned)

    def _input_candidates(
        self,
        demand: PredicateDemand,
        chain_id: str,
        input_name: str,
        data_type: str,
        *,
        now: datetime,
    ) -> tuple[ExternalInputRealization, ...]:
        definition = self.providers.data_type(data_type)
        candidates: list[ExternalInputRealization] = []
        is_raw = definition.kind == "raw_sensor"
        if is_raw:
            for source in self.deployment.candidate_sources(
                data_type=data_type,
                interval=demand.event_time_interval,
                eligible_source_ids=demand.eligible_source_ids,
                eligible_regions=demand.eligible_regions,
                require_live=True,
            ):
                candidates.append(
                    ExternalInputRealization(
                        input_name=input_name,
                        data_type=data_type,
                        kind=ExternalInputKind.LIVE_SOURCE,
                        node_id=source.node_id,
                        source_id=source.source_id,
                        bytes=_estimated_size(data_type),
                        access_modes=(ArtifactAccessMode.LOCAL,),
                    )
                )

        artifacts = self.artifacts.query(
            artifact_type=data_type,
            event_time_interval=demand.event_time_interval,
            required_access_modes=demand.hard_constraints.required_access_modes,
            now=now,
            require_interval_containment=True,
        )
        for artifact in artifacts:
            source_id = artifact.bindings.get("source_id")
            if demand.eligible_source_ids and source_id:
                if source_id not in demand.eligible_source_ids:
                    continue
            kind = (
                ExternalInputKind.RETAINED_ARTIFACT
                if definition.kind in {"raw_sensor", "derived_evidence", "feature", "provider_state"}
                else ExternalInputKind.DEPLOYMENT_ARTIFACT
            )
            candidates.append(
                ExternalInputRealization(
                    input_name=input_name,
                    data_type=data_type,
                    kind=kind,
                    node_id=artifact.location.node_id,
                    source_id=source_id,
                    artifact_id=artifact.artifact_id,
                    bytes=artifact.bytes or _estimated_size(data_type),
                    access_modes=artifact.access_modes,
                    expires_at=artifact.expires_at,
                )
            )
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.kind,
                    item.source_id or "",
                    str(item.artifact_id or ""),
                ),
            )
        )

    @staticmethod
    def _assignment_compatible(assignment: tuple[ExternalInputRealization, ...]) -> bool:
        by_name = {item.input_name: item for item in assignment}
        for suffix in ("a", "b", "left", "right"):
            matched = [
                item
                for name, item in by_name.items()
                if name.endswith(f"_{suffix}") and item.kind != ExternalInputKind.OMITTED_OPTIONAL
            ]
            source_ids = {item.source_id for item in matched if item.source_id is not None}
            if len(source_ids) > 1:
                return False
        video = by_name.get("video")
        for companion_name in ("calibration", "tracker_checkpoint"):
            companion = by_name.get(companion_name)
            if video and companion and video.source_id and companion.source_id:
                if video.source_id != companion.source_id:
                    return False
        video_a = by_name.get("video_a")
        video_b = by_name.get("video_b")
        if video_a and video_b and video_a.source_id and video_b.source_id:
            if video_a.source_id == video_b.source_id:
                return False
        return True

    def _placement_states(
        self,
        *,
        demand: PredicateDemand,
        chain_id: str,
        assignment: tuple[ExternalInputRealization, ...],
    ) -> tuple[tuple[_PlacementState, ...], tuple[PrunedAlternative, ...]]:
        chain = self.providers.chain(chain_id)
        external_refs = {
            f"external.{item.input_name}": _DataRef(
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
        states: list[_PlacementState] = [
            _PlacementState(steps=(), transfers=(), outputs=(), resource_usage=())
        ]
        pruned: list[PrunedAlternative] = []

        for step in chain.steps:
            next_states: list[_PlacementState] = []
            provider = self.providers.provider(step.provider_id)
            for state in states:
                output_map = state.output_map()
                input_refs: dict[str, _DataRef] = {}
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
                    continue

                candidate_nodes = self._candidate_nodes_for_step(demand, provider, input_refs)
                for node in candidate_nodes[: self.config.max_candidate_nodes_per_step]:
                    try:
                        profile = self.providers.profile(provider.provider_id, node.node_class)
                    except ProviderRegistryError:
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
                        continue
                    transfer_options = self._transfer_options(
                        demand=demand,
                        provider=provider,
                        target_node_id=node.node_id,
                        input_refs=input_refs,
                        target_step_id=step.step_id,
                    )
                    if transfer_options is None:
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
                                execution_ms=profile.execution_ms,
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
                            outputs[source_ref] = _DataRef(
                                source_ref=source_ref,
                                data_type=port.data_type,
                                node_id=node.node_id,
                                bytes=_estimated_size(port.data_type),
                            )
                        usage[node.node_id] = next_usage
                        next_states.append(
                            _PlacementState(
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
            states = _deduplicate_states(next_states)[: self.config.max_placement_variants_per_assignment]
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
                        reason=f"no compatible placement/data path for step {step.step_id}",
                    )
                )
                break
        return tuple(states), tuple(pruned)

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
        input_refs: Mapping[str, _DataRef],
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

        locality_counts: dict[str, int] = defaultdict(int)
        for ref in input_refs.values():
            locality_counts[ref.node_id] += 1
        return tuple(
            sorted(
                nodes,
                key=lambda node: (
                    -locality_counts.get(node.node_id, 0),
                    0 if node.node_class == "edge" else 1,
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
        input_refs: Mapping[str, _DataRef],
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

    def _materialize_graph(
        self,
        *,
        alternative_id: str,
        demand: PredicateDemand,
        chain_id: str,
        assignment: tuple[ExternalInputRealization, ...],
        state: _PlacementState,
    ) -> tuple[tuple[AlternativeGraphNode, ...], tuple[AlternativeGraphEdge, ...]]:
        chain = self.providers.chain(chain_id)
        nodes: dict[str, AlternativeGraphNode] = {}
        edges: dict[str, AlternativeGraphEdge] = {}
        source_graph_nodes: dict[str, str] = {}

        for item in assignment:
            if item.kind == ExternalInputKind.OMITTED_OPTIONAL:
                continue
            node_id = deterministic_id(
                "pan",
                {"alternative": alternative_id, "external": item.input_name, "realization": item},
            )
            source_graph_nodes[f"external.{item.input_name}"] = node_id
            nodes[node_id] = AlternativeGraphNode(
                node_id=node_id,
                kind=(
                    AlternativeNodeKind.LIVE_SOURCE
                    if item.kind == ExternalInputKind.LIVE_SOURCE
                    else AlternativeNodeKind.RETAINED_ARTIFACT
                ),
                label=f"{item.input_name}: {item.data_type}",
                demand_id=demand.demand_id,
                chain_id=chain_id,
                execution_node_id=item.node_id,
                data_type=item.data_type,
                source_id=item.source_id,
                artifact_id=item.artifact_id,
                annotations={
                    "input_kind": item.kind.value,
                    "bytes": item.bytes,
                    "expires_at": (
                        item.expires_at.isoformat().replace("+00:00", "Z")
                        if item.expires_at is not None
                        else None
                    ),
                },
            )

        placements = {placement.step_id: placement for placement in state.steps}
        step_graph_nodes: dict[str, str] = {}
        for step in chain.steps:
            placement = placements[step.step_id]
            node_id = deterministic_id(
                "pan",
                {"alternative": alternative_id, "step": step.step_id, "placement": placement},
            )
            step_graph_nodes[step.step_id] = node_id
            nodes[node_id] = AlternativeGraphNode(
                node_id=node_id,
                kind=AlternativeNodeKind.PROVIDER_OPERATION,
                label=f"{step.provider_id} @ {placement.node_id}",
                demand_id=demand.demand_id,
                chain_id=chain_id,
                step_id=step.step_id,
                provider_id=step.provider_id,
                execution_node_id=placement.node_id,
                annotations={
                    "startup_ms": placement.startup_ms,
                    "execution_ms": placement.execution_ms,
                    "quality_score": placement.quality_score,
                    "sharing_key": demand.sharing_key,
                    "reused_provider_instance_id": placement.reused_provider_instance_id,
                },
            )

        transfer_lookup = {
            (transfer.target_step_id, transfer.target_port, transfer.source_ref): transfer
            for transfer in state.transfers
        }
        for step in chain.steps:
            target_node = step_graph_nodes[step.step_id]
            for port_name, source_ref in sorted(step.bindings.items()):
                source_node = source_graph_nodes.get(source_ref)
                if source_node is None and "." in source_ref:
                    producer_step = source_ref.split(".", 1)[0]
                    source_node = step_graph_nodes.get(producer_step)
                if source_node is None:
                    continue
                transfer = transfer_lookup.get((step.step_id, port_name, source_ref))
                if transfer is not None and transfer.mode != TransferMode.LOCAL:
                    transfer_node_id = deterministic_id(
                        "pan",
                        {
                            "alternative": alternative_id,
                            "transfer": transfer,
                        },
                    )
                    nodes[transfer_node_id] = AlternativeGraphNode(
                        node_id=transfer_node_id,
                        kind=AlternativeNodeKind.TRANSFER,
                        label=f"{transfer.mode.value}: {transfer.data_type}",
                        demand_id=demand.demand_id,
                        chain_id=chain_id,
                        execution_node_id=transfer.target_node_id,
                        data_type=transfer.data_type,
                        annotations={
                            "source_node_id": transfer.source_node_id,
                            "target_node_id": transfer.target_node_id,
                            "bytes": transfer.bytes,
                            "estimated_ms": transfer.estimated_ms,
                            "path": list(transfer.path_node_ids),
                        },
                    )
                    _add_edge(
                        edges,
                        alternative_id,
                        source_node,
                        transfer_node_id,
                        AlternativeEdgeKind.DATA,
                        transfer.data_type,
                    )
                    _add_edge(
                        edges,
                        alternative_id,
                        transfer_node_id,
                        target_node,
                        AlternativeEdgeKind.DATA,
                        transfer.data_type,
                    )
                else:
                    data_type = transfer.data_type if transfer is not None else None
                    _add_edge(
                        edges,
                        alternative_id,
                        source_node,
                        target_node,
                        AlternativeEdgeKind.DATA,
                        data_type,
                    )

        result_ref = chain.outputs["result"]
        result_step = result_ref.split(".", 1)[0]
        result_sink_id = deterministic_id(
            "pan",
            {"alternative": alternative_id, "checkpoint_sink": str(demand.checkpoint_id)},
        )
        nodes[result_sink_id] = AlternativeGraphNode(
            node_id=result_sink_id,
            kind=AlternativeNodeKind.CHECKPOINT_RESULT_SINK,
            label=f"checkpoint {demand.checkpoint_id}",
            demand_id=demand.demand_id,
            chain_id=chain_id,
            data_type=chain.output_types["result"],
            annotations={"graph_node_id": demand.graph_node_id},
        )
        _add_edge(
            edges,
            alternative_id,
            step_graph_nodes[result_step],
            result_sink_id,
            AlternativeEdgeKind.SATISFIES,
            chain.output_types["result"],
        )

        for output_name, output_ref in sorted(chain.outputs.items()):
            if output_name == "result":
                continue
            data_type = chain.output_types[output_name]
            if data_type not in chain.continuation_output_types:
                continue
            producer_step = output_ref.split(".", 1)[0]
            sink_id = deterministic_id(
                "pan",
                {"alternative": alternative_id, "continuation": output_name, "type": data_type},
            )
            nodes[sink_id] = AlternativeGraphNode(
                node_id=sink_id,
                kind=AlternativeNodeKind.CONTINUATION_SINK,
                label=f"retain {data_type}",
                demand_id=demand.demand_id,
                chain_id=chain_id,
                data_type=data_type,
                annotations={
                    "required": data_type
                    in {req.artifact_type for req in demand.continuation_requirements},
                    "desired": data_type in set(demand.desired_continuation_types),
                },
            )
            _add_edge(
                edges,
                alternative_id,
                step_graph_nodes[producer_step],
                sink_id,
                AlternativeEdgeKind.PRODUCES,
                data_type,
            )
        return (
            tuple(sorted(nodes.values(), key=lambda node: node.node_id)),
            tuple(sorted(edges.values(), key=lambda edge: edge.edge_id)),
        )



def _spatial_preference(
    demand: PredicateDemand,
    assignment: tuple[ExternalInputRealization, ...],
) -> tuple[int, str]:
    """Return an explainable soft penalty for source placement.

    Zero means that no spatial hint exists.  With a hint, lower values represent
    earlier/higher-confidence observation groups.  Unpredicted sources remain
    feasible as fallbacks rather than being silently removed.
    """

    if not demand.source_preferences:
        return 0, "no spatial source preference"
    preferences = {item.source_id: item for item in demand.source_preferences}
    source_ids = tuple(
        dict.fromkeys(
            item.source_id
            for item in assignment
            if item.source_id is not None
            and item.kind != ExternalInputKind.OMITTED_OPTIONAL
        )
    )
    matches = [preferences[source_id] for source_id in source_ids if source_id in preferences]
    if not matches:
        return 1000, "source is outside predicted observation groups; retained as fallback"
    best = min(
        matches,
        key=lambda item: (item.priority_rank, -item.confidence, item.source_id),
    )
    confidence_penalty = int(round((1.0 - best.confidence) * 20.0))
    penalty = (best.priority_rank - 1) * 100 + confidence_penalty
    return penalty, best.reason

def _estimated_size(data_type: str) -> int:
    return _SIZE_DEFAULTS.get(data_type, 64_000)


def _deduplicate_states(states: Iterable[_PlacementState]) -> list[_PlacementState]:
    unique: dict[str, _PlacementState] = {}
    for state in states:
        key = deterministic_id(
            "placement",
            {"steps": state.steps, "transfers": state.transfers},
            length=32,
        )
        unique.setdefault(key, state)
    return [unique[key] for key in sorted(unique)]


def _add_edge(
    edges: dict[str, AlternativeGraphEdge],
    alternative_id: str,
    source_node_id: str,
    target_node_id: str,
    kind: AlternativeEdgeKind,
    data_type: str | None,
) -> None:
    edge_id = deterministic_id(
        "pae",
        {
            "alternative": alternative_id,
            "source": source_node_id,
            "target": target_node_id,
            "kind": kind,
            "data_type": data_type,
        },
    )
    edges[edge_id] = AlternativeGraphEdge(
        edge_id=edge_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        kind=kind,
        data_type=data_type,
    )
