"""Spatial-coordination metrics with campaign and replay-scope guards."""

from __future__ import annotations

from statistics import median

from fable.common.base import FableModel

from evaluation.schemas import CoordinationEpisode


class SpatialCoordinationMetrics(FableModel):
    total_episodes: int
    eligible_episodes: int
    evaluated_episodes: int
    excluded_unknown_topology: int
    excluded_unavailable_mobile_target: int
    timely_handoff_recall: float
    sensor_targeting_precision: float
    sensor_targeting_recall: float
    false_wakeup_rate: float
    weighted_activation_fanout_reduction: float
    median_lookout_lead_time_seconds: float | None = None


def evaluate_spatial_coordination(
    episodes: tuple[CoordinationEpisode, ...],
) -> SpatialCoordinationMetrics:
    unknown = sum(not item.spatial_evaluation_eligible for item in episodes)
    eligible = [item for item in episodes if item.spatial_evaluation_eligible]
    mobile_excluded = 0
    evaluated: list[CoordinationEpisode] = []
    for item in eligible:
        actual = item.actual_downstream_sensor_id
        if actual and actual in set(item.unavailable_mobile_sensor_ids):
            mobile_excluded += 1
            continue
        evaluated.append(item)

    timely_hits = 0
    target_hits = 0
    activated_total = 0
    false_wakeups = 0
    fanout_values: list[float] = []
    lead_times: list[float] = []
    for item in evaluated:
        actual = item.actual_downstream_sensor_id
        activated = set(item.activated_sensor_ids)
        activated_total += len(activated)
        if actual is not None and actual in activated:
            target_hits += 1
            if (
                item.downstream_observation_time is not None
                and (item.deadline is None or item.downstream_observation_time <= item.deadline)
            ):
                timely_hits += 1
            if item.downstream_observation_time is not None:
                lead_times.append(
                    (item.downstream_observation_time - item.prediction_time).total_seconds()
                )
        false_wakeups += len(activated - ({actual} if actual else set()))
        broadcast = len(set(item.replay_supported_sensor_ids))
        if broadcast:
            fanout_values.append(max(0.0, 1.0 - len(activated) / broadcast))

    denominator = len(evaluated)
    precision = target_hits / activated_total if activated_total else 0.0
    recall = target_hits / denominator if denominator else 0.0
    return SpatialCoordinationMetrics(
        total_episodes=len(episodes),
        eligible_episodes=len(eligible),
        evaluated_episodes=denominator,
        excluded_unknown_topology=unknown,
        excluded_unavailable_mobile_target=mobile_excluded,
        timely_handoff_recall=timely_hits / denominator if denominator else 0.0,
        sensor_targeting_precision=precision,
        sensor_targeting_recall=recall,
        false_wakeup_rate=false_wakeups / activated_total if activated_total else 0.0,
        weighted_activation_fanout_reduction=(
            sum(fanout_values) / len(fanout_values) if fanout_values else 0.0
        ),
        median_lookout_lead_time_seconds=(median(lead_times) if lead_times else None),
    )
