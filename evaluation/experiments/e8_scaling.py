"""E8 bounded one-factor-at-a-time scaling manifests."""

from __future__ import annotations

from itertools import product

from pydantic import Field

from evaluation.schemas import BaselineId, EvaluationMode
from fable.common.base import FrozenFableModel
from fable.common.ids import deterministic_id


class ScalingPoint(FrozenFableModel):
    sensors: int = Field(ge=1)
    concurrent_requests: int = Field(ge=1)
    hypotheses_per_request: int = Field(ge=1)
    branch_factor: int = Field(ge=1)
    providers_per_predicate: int = Field(ge=1)
    shared_demand_fraction: float = Field(ge=0, le=1)
    event_rate: str = Field(pattern=r"^(low|medium|high)$")


class PlannedScalingRun(FrozenFableModel):
    schema_version: str = "fable.planned_scaling_run.v1"
    run_id: str
    point: ScalingPoint
    baseline_id: BaselineId
    mode: EvaluationMode = EvaluationMode.COMMON_PERCEPTION
    network_profile_id: str
    repetition: int = Field(ge=1)
    random_seed: int = Field(ge=0)
    target_relative_timely_recall: float = Field(default=0.95, gt=0, le=1)
    maximum_p95_control_latency_ms: float = Field(default=250.0, gt=0)


def build(
    _catalog=None,
    *,
    repetitions: int = 10,
    seed: int = 0,
    network_profiles: tuple[str, ...] = ("good_network", "cloud_degraded"),
    target_relative_timely_recall: float = 0.95,
    maximum_p95_control_latency_ms: float = 250.0,
) -> tuple[PlannedScalingRun, ...]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    standard = ScalingPoint(
        sensors=10,
        concurrent_requests=4,
        hypotheses_per_request=4,
        branch_factor=2,
        providers_per_predicate=2,
        shared_demand_fraction=0.25,
        event_rate="medium",
    )
    points = {standard.model_dump_json(): standard}
    sweeps = {
        "sensors": (5, 10, 15, 20),
        "concurrent_requests": (1, 4, 8, 16),
        "hypotheses_per_request": (1, 4, 8, 16, 32),
        "branch_factor": (1, 2, 4, 8),
        "providers_per_predicate": (1, 2, 4, 8),
        "shared_demand_fraction": (0.0, 0.25, 0.5, 0.75),
        "event_rate": ("low", "medium", "high"),
    }
    for field, values in sweeps.items():
        for value in values:
            point = standard.model_copy(update={field: value})
            points[point.model_dump_json()] = point
    stress = ScalingPoint(
        sensors=20,
        concurrent_requests=16,
        hypotheses_per_request=16,
        branch_factor=4,
        providers_per_predicate=4,
        shared_demand_fraction=0.5,
        event_rate="high",
    )
    points[stress.model_dump_json()] = stress

    rows = []
    baselines = (
        BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
        BaselineId.FABLE,
        BaselineId.FABLE_NO_SHARING,
    )
    for point, baseline, profile, repetition in product(
        sorted(points.values(), key=lambda item: item.model_dump_json()),
        baselines,
        network_profiles,
        range(1, repetitions + 1),
    ):
        identity = {
            "point": point,
            "baseline": baseline,
            "profile": profile,
            "repetition": repetition,
            "seed": seed + repetition - 1,
        }
        rows.append(
            PlannedScalingRun(
                run_id=deterministic_id("scaling_run", identity, length=32),
                point=point,
                baseline_id=baseline,
                network_profile_id=profile,
                repetition=repetition,
                random_seed=seed + repetition - 1,
                target_relative_timely_recall=target_relative_timely_recall,
                maximum_p95_control_latency_ms=maximum_p95_control_latency_ms,
            )
        )
    return tuple(sorted(rows, key=lambda item: item.run_id))


def build_saturation_refinement(
    *,
    repetitions: int = 10,
    seed: int = 0,
    network_profiles: tuple[str, ...] = ("good_network", "cloud_degraded"),
    target_relative_timely_recall: float = 0.95,
    maximum_p95_control_latency_ms: float = 250.0,
) -> tuple[PlannedScalingRun, ...]:
    """Refine the gap immediately below the combined E8 stress point.

    Sensors, hypotheses, branching, providers, sharing and event rate remain
    fixed at the combined-stress values. Only concurrent requests changes,
    yielding 1,024, 2,048, 4,096, 6,144 and 8,192 generated labels.
    """

    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    points = tuple(
        ScalingPoint(
            sensors=20,
            concurrent_requests=concurrent_requests,
            hypotheses_per_request=16,
            branch_factor=4,
            providers_per_predicate=4,
            shared_demand_fraction=0.5,
            event_rate="high",
        )
        for concurrent_requests in (2, 4, 8, 12, 16)
    )
    baselines = (
        BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
        BaselineId.FABLE,
        BaselineId.FABLE_NO_SHARING,
    )
    rows = []
    for point, baseline, profile, repetition in product(
        points,
        baselines,
        network_profiles,
        range(1, repetitions + 1),
    ):
        identity = {
            "experiment": "E8_SATURATION_REFINEMENT",
            "point": point,
            "baseline": baseline,
            "profile": profile,
            "repetition": repetition,
            "seed": seed + repetition - 1,
        }
        rows.append(
            PlannedScalingRun(
                run_id=deterministic_id("scaling_refinement", identity, length=32),
                point=point,
                baseline_id=baseline,
                network_profile_id=profile,
                repetition=repetition,
                random_seed=seed + repetition - 1,
                target_relative_timely_recall=target_relative_timely_recall,
                maximum_p95_control_latency_ms=maximum_p95_control_latency_ms,
            )
        )
    return tuple(sorted(rows, key=lambda item: item.run_id))
