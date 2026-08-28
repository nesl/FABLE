"""Deterministic confidence intervals, paired comparisons, and load bounds."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt

from pydantic import Field

from fable.common.base import FrozenFableModel


class ConfidenceInterval(FrozenFableModel):
    count: int = Field(ge=0)
    mean: float
    lower_95: float
    upper_95: float


class PairedComparison(FrozenFableModel):
    pair_count: int = Field(ge=0)
    treatment_mean: float
    control_mean: float
    difference: ConfidenceInterval
    treatment_better_fraction: float = Field(ge=0, le=1)


class LoadSample(FrozenFableModel):
    workload: float = Field(ge=0)
    timely_recall: float = Field(ge=0, le=1)
    p95_latency_ms: float = Field(ge=0)
    completed: bool = True


class SustainableLoadResult(FrozenFableModel):
    target_timely_recall: float = Field(ge=0, le=1)
    maximum_p95_latency_ms: float = Field(gt=0)
    maximum_sustainable_workload: float | None = None
    evaluated_workloads: tuple[float, ...] = ()
    rejected_workloads: tuple[float, ...] = ()


def confidence_interval(values: tuple[float, ...]) -> ConfidenceInterval:
    if not values:
        return ConfidenceInterval(count=0, mean=0, lower_95=0, upper_95=0)
    mean = sum(values) / len(values)
    if len(values) == 1:
        return ConfidenceInterval(
            count=1,
            mean=mean,
            lower_95=mean,
            upper_95=mean,
        )
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = 1.96 * sqrt(variance / len(values))
    return ConfidenceInterval(
        count=len(values),
        mean=mean,
        lower_95=mean - half_width,
        upper_95=mean + half_width,
    )


def paired_comparison(
    treatment: dict[str, float],
    control: dict[str, float],
    *,
    lower_is_better: bool = False,
) -> PairedComparison:
    keys = sorted(set(treatment) & set(control))
    treatment_values = tuple(treatment[key] for key in keys)
    control_values = tuple(control[key] for key in keys)
    differences = tuple(
        treatment[key] - control[key] for key in keys
    )
    better = sum(
        (difference < 0 if lower_is_better else difference > 0)
        for difference in differences
    )
    return PairedComparison(
        pair_count=len(keys),
        treatment_mean=(
            sum(treatment_values) / len(treatment_values)
            if treatment_values
            else 0
        ),
        control_mean=(
            sum(control_values) / len(control_values) if control_values else 0
        ),
        difference=confidence_interval(differences),
        treatment_better_fraction=better / len(keys) if keys else 0,
    )


def maximum_sustainable_load(
    samples: tuple[LoadSample, ...],
    *,
    target_timely_recall: float,
    maximum_p95_latency_ms: float,
) -> SustainableLoadResult:
    grouped: dict[float, list[LoadSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.workload].append(sample)
    accepted = []
    rejected = []
    for workload, rows in sorted(grouped.items()):
        recall = sum(row.timely_recall for row in rows) / len(rows)
        latency = sorted(row.p95_latency_ms for row in rows)[
            int(round((len(rows) - 1) * 0.95))
        ]
        valid = (
            all(row.completed for row in rows)
            and recall >= target_timely_recall
            and latency <= maximum_p95_latency_ms
        )
        (accepted if valid else rejected).append(workload)
    return SustainableLoadResult(
        target_timely_recall=target_timely_recall,
        maximum_p95_latency_ms=maximum_p95_latency_ms,
        maximum_sustainable_workload=max(accepted) if accepted else None,
        evaluated_workloads=tuple(sorted(grouped)),
        rejected_workloads=tuple(rejected),
    )
