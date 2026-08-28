from __future__ import annotations

from fable.common.base import FableModel

from evaluation.schemas import HypothesisTransition, PlanDecision, ResourceSample


class ScalingMetrics(FableModel):
    planning_decisions: int = 0
    p95_planning_latency_ms: float | None = None
    p99_planning_latency_ms: float | None = None
    mean_orchestrator_cpu_utilization: float | None = None
    peak_orchestrator_memory_bytes: int = 0
    seeds: int = 0
    forks: int = 0
    merges: int = 0
    expirations: int = 0
    duplicates_suppressed: int = 0
    labels_per_alternative: float = 0.0


def summarize_scaling(
    plans: tuple[PlanDecision, ...],
    resources: tuple[ResourceSample, ...],
    transitions: tuple[HypothesisTransition, ...],
    *,
    orchestrator_node_id: str = "x86server",
) -> ScalingMetrics:
    latencies = sorted(item.planning_latency_ms for item in plans)
    orchestrator = [
        item for item in resources if item.node_id == orchestrator_node_id
    ]
    kinds = [item.transition_kind.upper() for item in transitions]
    alternatives = sum(len(item.selected_alternative_ids) for item in plans)
    labels = sum(item.labels_generated for item in plans)
    return ScalingMetrics(
        planning_decisions=len(plans),
        p95_planning_latency_ms=_percentile(latencies, 0.95),
        p99_planning_latency_ms=_percentile(latencies, 0.99),
        mean_orchestrator_cpu_utilization=(
            sum(item.cpu_utilization for item in orchestrator) / len(orchestrator)
            if orchestrator
            else None
        ),
        peak_orchestrator_memory_bytes=max(
            (item.memory_bytes for item in orchestrator),
            default=0,
        ),
        seeds=sum(item in {"CREATED", "SEED"} for item in kinds),
        forks=sum(item == "FORKED" for item in kinds),
        merges=sum(item == "MERGED" for item in kinds),
        expirations=sum(item in {"EXPIRED", "WINDOW_EXPIRED"} for item in kinds),
        duplicates_suppressed=sum(item == "DUPLICATE" for item in kinds),
        labels_per_alternative=labels / alternatives if alternatives else 0.0,
    )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return values[int(round((len(values) - 1) * fraction))]
