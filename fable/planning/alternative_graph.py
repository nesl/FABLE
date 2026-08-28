"""Construct checkpoint-bounded physical implementation alternatives.

Nodes describe external inputs, provider steps, artifacts/transfers, and result
outputs; edges describe typed data dependencies. The graph is not a copy of the
semantic Event Graph. It materializes provider-chain DAGs for diagnostics and
search, while ``alternatives`` is the directly selectable unit. This phase
enumerates and annotates alternatives but deliberately does not rank them.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from itertools import product
from math import ceil
from typing import Any, Callable
from uuid import UUID

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


from .alternatives.config import AlternativeBuildConfig
from .alternatives.inputs import AlternativeInputResolver
from .alternatives.materialize import AlternativeGraphMaterializer
from .alternatives.placement import PlacementEnumerator








class PhysicalAlternativeGraphBuilder:
    """Enumerate feasible provider-chain realizations without ranking them."""

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry,
        artifact_catalog: ArtifactCatalog,
        deployment: DeploymentGraph,
        config: AlternativeBuildConfig | None = None,
        active_providers: Iterable[ActiveProviderInstance] = (),
        placement_eligible: Callable[[str, str], bool] | None = None,
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
        # Split input resolution, placement enumeration, and graph projection
        # into collaborators so each constraint has one authoritative owner.
        self.placement_eligible = placement_eligible
        self.input_resolver = AlternativeInputResolver(
            provider_registry=self.providers,
            artifact_catalog=self.artifacts,
            deployment=self.deployment,
            config=self.config,
        )
        self.placement_enumerator = PlacementEnumerator(
            provider_registry=self.providers,
            deployment=self.deployment,
            config=self.config,
            active_providers=self.active_providers,
            placement_eligible=self.placement_eligible,
        )
        self.graph_materializer = AlternativeGraphMaterializer(
            provider_registry=self.providers
        )

    def build(
        self,
        demands: Iterable[PredicateDemand],
        *,
        now: datetime | None = None,
    ) -> PhysicalAlternativeGraph:
        """Materialize every bounded realization and its explicit prune reason."""

        observed_now = ensure_utc(now or utc_now())
        # Demand order affects artifact serialization only; alternatives for
        # different demands are independent at this construction phase.
        ordered_demands = tuple(sorted(demands, key=lambda demand: str(demand.demand_id)))
        if not ordered_demands:
            raise AlternativeGraphError("at least one predicate demand is required")

        nodes: dict[str, AlternativeGraphNode] = {}
        edges: dict[str, AlternativeGraphEdge] = {}
        alternatives: list[PhysicalAlternative] = []
        pruned: list[PrunedAlternative] = []

        for demand in ordered_demands:
            # Expired demands remain visible as typed pruning evidence rather
            # than disappearing from diagnostics.
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
                # Resolve live sources and retained/static artifacts before
                # placement because access mode and artifact location constrain
                # which nodes can legally host the first provider step.
                assignments, assignment_pruned = self.input_resolver._external_assignments(
                    demand, chain.chain_id, now=observed_now
                )
                pruned.extend(assignment_pruned)
                for assignment in assignments:
                    # Placement enumeration walks the typed chain and adds any
                    # required transfers between chosen nodes.
                    placement_states, placement_pruned = self.placement_enumerator._placement_states(
                        demand=demand,
                        chain_id=chain.chain_id,
                        assignment=assignment,
                    )
                    pruned.extend(placement_pruned)
                    for state in placement_states:
                        # Candidate identity includes all physical choices so
                        # equivalent realizations deduplicate deterministically.
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
                        alt_nodes, alt_edges = self.graph_materializer._materialize_graph(
                            alternative_id=alternative_id,
                            demand=demand,
                            chain_id=chain.chain_id,
                            assignment=assignment,
                            state=state,
                        )
                        nodes.update({node.node_id: node for node in alt_nodes})
                        edges.update({edge.edge_id: edge for edge in alt_edges})
                        # Execution mode describes *when* the demand operates,
                        # not merely how its recording is addressed. A node-
                        # local replay source is represented as LIVE_SOURCE in
                        # deployment metadata, but a retrospective semantic
                        # demand must still issue a bounded replay command.
                        execution_mode = (
                            ExecutionMode.RETROSPECTIVE
                            if demand.retrospective_context
                            else (
                                ExecutionMode.LIVE
                                if any(
                                    item.kind == ExternalInputKind.LIVE_SOURCE
                                    for item in assignment
                                )
                                else ExecutionMode.RETROSPECTIVE
                            )
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
