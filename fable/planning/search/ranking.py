"""Ranking, deduplication, and dominance pruning for search labels."""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID

from fable.planning.search_models import LabelSearchState, PruneCode, PruningRecord


class LabelRanker:
    """Deterministic ordering and Pareto-style dominance for search labels."""

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


__all__ = ["LabelRanker"]
