from __future__ import annotations

from statistics import median

from fable.common.base import FableModel
from evaluation.schemas import NetworkCondition, PlanDecision, ProviderLifecycleEvent


class AdaptationMetrics(FableModel):
    disturbance_epochs: int = 0
    replans_after_disturbance: int = 0
    median_condition_to_plan_seconds: float | None = None
    p95_condition_to_plan_seconds: float | None = None
    provider_relocations: int = 0
    provider_substitutions: int = 0
    processing_blackout_seconds: float = 0.0


def summarize_adaptation(
    conditions: tuple[NetworkCondition, ...],
    plans: tuple[PlanDecision, ...],
    lifecycle: tuple[ProviderLifecycleEvent, ...] = (),
) -> AdaptationMetrics:
    epochs = sorted(
        {item.condition_epoch for item in conditions if item.condition_epoch > 0}
    )
    latencies: list[float] = []
    replans = 0
    for epoch in epochs:
        disturbance = min(
            item.wall_timestamp for item in conditions if item.condition_epoch == epoch
        )
        later = [
            item for item in plans
            if item.resource_epoch >= epoch and item.wall_timestamp >= disturbance
        ]
        if later:
            replans += 1
            latencies.append((min(item.wall_timestamp for item in later) - disturbance).total_seconds())
    relocations = substitutions = 0
    ordered = sorted(plans, key=lambda item: item.wall_timestamp)
    for previous, current in zip(ordered, ordered[1:]):
        if set(previous.selected_node_ids) != set(current.selected_node_ids):
            relocations += 1
        previous_providers = {item.split("@", 1)[0] for item in previous.activated_provider_keys}
        current_providers = {item.split("@", 1)[0] for item in current.activated_provider_keys}
        if previous_providers != current_providers:
            substitutions += 1
    ordered_latencies = sorted(latencies)
    return AdaptationMetrics(
        disturbance_epochs=len(epochs),
        replans_after_disturbance=replans,
        median_condition_to_plan_seconds=(median(latencies) if latencies else None),
        p95_condition_to_plan_seconds=(
            ordered_latencies[int(round((len(ordered_latencies) - 1) * 0.95))]
            if ordered_latencies else None
        ),
        provider_relocations=relocations,
        provider_substitutions=substitutions,
        processing_blackout_seconds=_blackout_seconds(lifecycle),
    )


def _blackout_seconds(events: tuple[ProviderLifecycleEvent, ...]) -> float:
    unavailable_at = None
    blackout = 0.0
    for item in sorted(events, key=lambda value: value.wall_timestamp):
        event = item.lifecycle_event.upper()
        if event in {"FAILED", "UNAVAILABLE"} and unavailable_at is None:
            unavailable_at = item.wall_timestamp
        elif event in {"READY", "ACTIVE", "AVAILABLE"} and unavailable_at is not None:
            blackout += max(0.0, (item.wall_timestamp - unavailable_at).total_seconds())
            unavailable_at = None
    return blackout
