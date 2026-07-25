from __future__ import annotations

from statistics import median

from pydantic import Field

from fable.common.base import FableModel

from evaluation.schemas import PlanDecision


class PlanningMetrics(FableModel):
    decisions: int
    feasible_decisions: int
    feasible_plan_rate: float
    median_planning_latency_ms: float | None = None
    p95_planning_latency_ms: float | None = None
    total_labels_generated: int = 0
    total_labels_pruned: int = 0
    total_labels_retained: int = 0
    mean_oracle_gap_ms: float | None = None
    mean_predicted_transfer_bytes: float | None = None
    sensor_change_fraction: float = 0.0
    provider_change_fraction: float = 0.0
    representation_change_fraction: float = 0.0


def summarize_planning(records: tuple[PlanDecision, ...]) -> PlanningMetrics:
    if not records:
        return PlanningMetrics(decisions=0, feasible_decisions=0, feasible_plan_rate=0.0)
    latencies = sorted(item.planning_latency_ms for item in records)
    gaps = [item.oracle_gap_ms for item in records if item.oracle_gap_ms is not None]
    transfers = [item.predicted_transfer_bytes for item in records if item.predicted_transfer_bytes is not None]
    comparisons = max(0, len(records) - 1)
    sensor_changes = provider_changes = representation_changes = 0
    for previous, current in zip(records, records[1:]):
        sensor_changes += set(previous.selected_source_ids) != set(current.selected_source_ids)
        provider_changes += set(previous.activated_provider_keys) != set(current.activated_provider_keys)
        representation_changes += set(previous.continuation_types) != set(current.continuation_types)
    feasible = sum(bool(item.selected_alternative_ids) for item in records)
    return PlanningMetrics(
        decisions=len(records),
        feasible_decisions=feasible,
        feasible_plan_rate=feasible / len(records),
        median_planning_latency_ms=median(latencies),
        p95_planning_latency_ms=_percentile(latencies, 0.95),
        total_labels_generated=sum(item.labels_generated for item in records),
        total_labels_pruned=sum(item.labels_pruned for item in records),
        total_labels_retained=sum(item.labels_retained for item in records),
        mean_oracle_gap_ms=(sum(gaps) / len(gaps) if gaps else None),
        mean_predicted_transfer_bytes=(sum(transfers) / len(transfers) if transfers else None),
        sensor_change_fraction=(sensor_changes / comparisons if comparisons else 0.0),
        provider_change_fraction=(provider_changes / comparisons if comparisons else 0.0),
        representation_change_fraction=(representation_changes / comparisons if comparisons else 0.0),
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return values[int(round((len(values) - 1) * fraction))]
