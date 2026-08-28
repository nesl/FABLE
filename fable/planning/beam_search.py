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

from fable.common.enums import (
    ArtifactAccessMode,
    BindingCapability,
    PlanStatus,
    ProviderPortKind,
)
from fable.common.ids import deterministic_id
from fable.common.schemas import (
    ExecutionInput,
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


from .search.config import BeamSearchConfig
from .search.feasibility import PlanFeasibilityChecker
from .search.label_builder import LabelBuilder
from .search.ranking import LabelRanker
from .search.projection import ExecutionPlanProjector


class BoundedLabelPlanner:
    """Beam plan search with hard filtering, deduplication, and dominance.

    At each demand boundary it enumerates extensions, rejects infeasible ones,
    creates immutable partial-plan labels, deduplicates equivalent labels,
    removes dominated states, ranks the survivors lexicographically, and keeps
    ``beam_width`` candidates. Ranking is part of beam search—not a second
    planner run afterward. The winning label is projected to ``ExecutionPlan``.
    """

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
        self.feasibility = PlanFeasibilityChecker(
            provider_registry=self.providers,
            artifact_catalog=self.artifacts,
            deployment=self.deployment,
            config=self.config,
            representation_compatibility=self.representations,
        )
        self.label_builder = LabelBuilder(
            provider_registry=self.providers,
            representation_compatibility=self.representations,
            near_expiry_horizon_ms=self.config.near_expiry_horizon_ms,
        )
        self.ranker = LabelRanker()
        self.projector = ExecutionPlanProjector()

    def search(
        self,
        graph: PhysicalAlternativeGraph,
        demands: Iterable[PredicateDemand],
        *,
        now: datetime | None = None,
        required_checkpoint_consumers: Iterable[str] = (),
    ) -> PlanSearchResult:
        """Search one checkpoint with bounded nondominated labels.

        Each boundary adds exactly one demand realization. Feasibility is
        checked before extension; duplicate, dominated, and over-width labels
        are retained as traceable pruning records rather than silently lost.
        """
        observed_now = ensure_utc(now or utc_now())
        # Materialize once because callers may pass generators and because all
        # later consistency checks need identity-based lookup.
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
        # Earliest useful completion first makes tight obligations cross the
        # beam boundary before more permissive ones.
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
                    # Feasibility checks resources, bindings, representations,
                    # deadlines, and transfer constraints without mutating the
                    # parent label.
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
            # Beam truncation is the sole approximation in the normal search;
            # all feasibility and dominance decisions before it are exact.
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
                # No later demand can repair an infeasible partial plan.
                break

        complete = tuple(item for item in retained if item is not None)
        if required_consumers:
            # Continuation compatibility is a checkpoint-level constraint and
            # can only be evaluated after every demand has been covered.
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
        # Selection and fallback order use the same deterministic rank tuple;
        # dominated final labels remain eligible as bounded recovery options.
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
        execution_plan = (
            self.to_execution_plan(selected, now=observed_now)
            if selected is not None
            else None
        )

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
        return self.feasibility.check_extension(
            parent,
            alternative,
            demand,
            demand_map=demand_map,
            now=now,
        )

    def extend_label(
        self,
        parent: LabelSearchState | None,
        alternative: PhysicalAlternative,
        demand: PredicateDemand,
        *,
        demand_map: Mapping[UUID, PredicateDemand],
        now: datetime,
    ) -> LabelSearchState:
        return self.label_builder.extend_label(
            parent,
            alternative,
            demand,
            demand_map=demand_map,
            now=now,
        )

    def rank_key(self, state: LabelSearchState | None) -> tuple:
        return self.ranker.rank_key(state)

    def dominates(self, left: LabelSearchState, right: LabelSearchState) -> bool:
        return self.ranker.dominates(left, right)

    def to_execution_plan(
        self,
        state: LabelSearchState | None,
        *,
        now: datetime | None = None,
    ) -> ExecutionPlan | None:
        return self.projector.project(state, now=now)

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


    def _deduplicate(
        self,
        states: Iterable[LabelSearchState],
        *,
        boundary_index: int,
        demand_id: UUID,
    ) -> tuple[tuple[LabelSearchState, ...], tuple[PruningRecord, ...]]:
        return self.ranker._deduplicate(
            states, boundary_index=boundary_index, demand_id=demand_id
        )

    def _dominance_prune(
        self,
        states: Iterable[LabelSearchState],
        *,
        boundary_index: int,
        demand_id: UUID,
    ) -> tuple[tuple[LabelSearchState, ...], tuple[PruningRecord, ...]]:
        return self.ranker._dominance_prune(
            states, boundary_index=boundary_index, demand_id=demand_id
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


def _external_roots_for_step(chain, step_id: str) -> set[str]:
    """Return external input names that transitively feed one chain step."""

    step_by_id = {step.step_id: step for step in chain.steps}
    cache: dict[str, set[str]] = {}

    def visit(current: str, stack: set[str]) -> set[str]:
        if current in cache:
            return cache[current]
        if current in stack:
            raise PlanSearchError(f"cycle in provider chain {chain.chain_id}")
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
