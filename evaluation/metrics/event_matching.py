"""Common CE matching metrics for native and common-perception runs."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from fable.common.base import FableModel

from evaluation.schemas import ComplexEventResult


class GroundTruthEvent(FableModel):
    event_id: str
    event_family: str
    start_time: datetime
    end_time: datetime
    deadline: datetime | None = None
    bindings: dict[str, str] = Field(default_factory=dict)


class EventMatchMetrics(FableModel):
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    timely_recall: float
    median_detection_delay_seconds: float | None = None
    p95_detection_delay_seconds: float | None = None
    role_binding_accuracy: float | None = None


def evaluate_event_results(
    ground_truth: tuple[GroundTruthEvent, ...],
    results: tuple[ComplexEventResult, ...],
    *,
    minimum_temporal_iou: float = 0.1,
    temporal_boundary_tolerance_seconds: float = 0.0,
) -> EventMatchMetrics:
    if temporal_boundary_tolerance_seconds < 0:
        raise ValueError("temporal_boundary_tolerance_seconds must be non-negative")
    candidates: list[tuple[float, int, int]] = []
    for gi, truth in enumerate(ground_truth):
        for ri, result in enumerate(results):
            if _norm(truth.event_family) != _norm(result.event_family):
                continue
            score = _temporal_iou(
                truth.start_time,
                truth.end_time,
                result.event_start_time,
                result.event_end_time,
            )
            boundary_distance = _interval_distance_seconds(
                truth.start_time,
                truth.end_time,
                result.event_start_time,
                result.event_end_time,
            )
            within_tolerance = bool(
                temporal_boundary_tolerance_seconds > 0
                and boundary_distance <= temporal_boundary_tolerance_seconds
            )
            if score >= minimum_temporal_iou or within_tolerance:
                candidates.append((score, gi, ri))
    matched_truth: set[int] = set()
    matched_results: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, gi, ri in sorted(candidates, reverse=True):
        if gi in matched_truth or ri in matched_results:
            continue
        matched_truth.add(gi)
        matched_results.add(ri)
        matches.append((gi, ri))
    tp = len(matches)
    fp = len(results) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    timely = 0
    delays: list[float] = []
    role_total = 0
    role_correct = 0
    for gi, ri in matches:
        truth = ground_truth[gi]
        result = results[ri]
        delays.append((result.emitted_at - truth.end_time).total_seconds())
        if truth.deadline is None or result.emitted_at <= truth.deadline:
            timely += 1
        for role, identity in truth.bindings.items():
            role_total += 1
            role_correct += result.bindings.get(role) == identity
    timely_recall = timely / len(ground_truth) if ground_truth else 0.0
    ordered_delays = sorted(delays)
    return EventMatchMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        timely_recall=timely_recall,
        median_detection_delay_seconds=_percentile(ordered_delays, 0.5),
        p95_detection_delay_seconds=_percentile(ordered_delays, 0.95),
        role_binding_accuracy=(role_correct / role_total if role_total else None),
    )


def _temporal_iou(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> float:
    intersection = max(0.0, (min(a1, b1) - max(a0, b0)).total_seconds())
    union = max(a1, b1) - min(a0, b0)
    seconds = union.total_seconds()
    if seconds == 0:
        return 1.0 if a0 == b0 else 0.0
    return intersection / seconds


def _interval_distance_seconds(
    a0: datetime, a1: datetime, b0: datetime, b1: datetime
) -> float:
    """Return zero for overlapping intervals, otherwise their nearest gap."""

    if a1 < b0:
        return (b0 - a1).total_seconds()
    if b1 < a0:
        return (a0 - b1).total_seconds()
    return 0.0


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    index = int(round((len(values) - 1) * fraction))
    return values[index]


def _norm(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")
