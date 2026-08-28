"""Deterministic profile-driven execution for the largest E8 scaling points."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import Field

from evaluation.experiments.e8_scaling import PlannedScalingRun
from evaluation.runner import JsonlEventStore
from evaluation.schemas import PlanDecision, ResourceSample
from fable.common.base import FrozenFableModel
from fable.common.ids import deterministic_id


class ScalingExecutionProfile(FrozenFableModel):
    schema_version: str = "fable.scaling_execution_profile.v1"
    profile_id: str = Field(min_length=1)
    calibrated: bool
    calibration_source: str = Field(min_length=1)
    calibration_metadata: dict[str, object] = Field(default_factory=dict)
    base_planning_latency_ms: float = Field(gt=0)
    latency_per_label_ms: float = Field(gt=0)
    base_cpu_seconds_per_request: float = Field(gt=0)
    cpu_seconds_per_label: float = Field(gt=0)
    base_memory_bytes: int = Field(gt=0)
    memory_bytes_per_live_hypothesis: int = Field(gt=0)
    network_bytes_per_provider: int = Field(gt=0)
    nominal_timely_recall: float = Field(gt=0, le=1)
    overload_label_threshold: int = Field(gt=0)
    overload_recall_penalty_per_threshold: float = Field(gt=0, le=1)
    event_rate_multipliers: dict[str, float] = Field(
        default_factory=lambda: {"low": 0.5, "medium": 1.0, "high": 2.0}
    )
    network_profiles: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "good_network": {
                "round_trip_latency_ms": 50.0,
                "bandwidth_mbps": 1000.0,
                "loss_fraction": 0.0,
            },
            "cloud_degraded": {
                "round_trip_latency_ms": 256.0,
                "bandwidth_mbps": 5.0,
                "loss_fraction": 0.03,
            },
        }
    )


class ProfiledScalingResult(FrozenFableModel):
    schema_version: str = "fable.profiled_scaling_result.v1"
    run_id: str
    baseline_id: str
    profile_id: str
    calibrated: bool
    execution_mode: str = "PROFILE_DRIVEN"
    completed: bool
    logical_demands: int = Field(ge=1)
    generated_labels: int = Field(ge=1)
    effective_provider_invocations: int = Field(ge=1)
    sharing_savings: int = Field(ge=0)
    p95_control_latency_ms: float = Field(ge=0)
    timely_recall: float = Field(ge=0, le=1)
    cpu_time_seconds: float = Field(ge=0)
    peak_memory_bytes: int = Field(ge=0)
    network_bytes: int = Field(ge=0)
    network_profile_id: str
    network_round_trip_latency_ms: float = Field(ge=0)
    target_timely_recall: float = Field(ge=0, le=1)
    maximum_p95_control_latency_ms: float = Field(gt=0)
    slo_satisfied: bool


def execute_profiled_scaling_run(
    run: PlannedScalingRun,
    profile: ScalingExecutionProfile,
    output_root: str | Path,
    *,
    allow_unmeasured_profile: bool = False,
) -> ProfiledScalingResult:
    if not profile.calibrated and not allow_unmeasured_profile:
        raise ValueError(
            "unmeasured scaling profiles are implementation fixtures only; "
            "pass allow_unmeasured_profile=True explicitly"
        )
    point = run.point
    try:
        network = profile.network_profiles[run.network_profile_id]
    except KeyError as exc:
        raise ValueError(
            f"scaling profile has no network calibration for {run.network_profile_id}"
        ) from exc
    rate = profile.event_rate_multipliers[point.event_rate]
    logical_demands = max(
        1,
        round(
            point.concurrent_requests
            * point.hypotheses_per_request
            * point.branch_factor
            * rate
        ),
    )
    generated_labels = max(
        1,
        logical_demands * point.providers_per_predicate,
    )
    shareable = round(generated_labels * point.shared_demand_fraction)
    if run.baseline_id.value == "FABLE":
        sharing_savings = shareable
    elif run.baseline_id.value == "B3_TASK_RESOURCE_ADAPTIVE":
        sharing_savings = shareable // 2
    else:
        sharing_savings = 0
    effective_invocations = max(1, generated_labels - sharing_savings)
    sensor_parallelism = max(1.0, point.sensors**0.5)
    jitter = _deterministic_jitter(run.run_id, run.random_seed)
    planner_latency = (
        profile.base_planning_latency_ms
        + profile.latency_per_label_ms * generated_labels / sensor_parallelism
    ) * jitter
    overload_ratio = max(
        0.0,
        (effective_invocations - profile.overload_label_threshold)
        / profile.overload_label_threshold,
    )
    recall = max(
        0.0,
        profile.nominal_timely_recall
        - overload_ratio * profile.overload_recall_penalty_per_threshold,
    )
    cpu_seconds = (
        profile.base_cpu_seconds_per_request * point.concurrent_requests
        + profile.cpu_seconds_per_label * effective_invocations
    )
    memory_bytes = (
        profile.base_memory_bytes
        + profile.memory_bytes_per_live_hypothesis
        * point.concurrent_requests
        * point.hypotheses_per_request
    )
    network_bytes = (
        effective_invocations * profile.network_bytes_per_provider
    )
    serialization_ms = (
        network_bytes * 8.0 / (network["bandwidth_mbps"] * 1_000_000.0) * 1000.0
    ) / sensor_parallelism
    latency = (
        planner_latency
        + network["round_trip_latency_ms"]
        + serialization_ms
    ) / max(0.01, 1.0 - network["loss_fraction"])
    target_recall = (
        run.target_relative_timely_recall * profile.nominal_timely_recall
    )
    completed = latency <= run.maximum_p95_control_latency_ms * 4
    result = ProfiledScalingResult(
        run_id=run.run_id,
        baseline_id=run.baseline_id.value,
        profile_id=profile.profile_id,
        calibrated=profile.calibrated,
        completed=completed,
        logical_demands=logical_demands,
        generated_labels=generated_labels,
        effective_provider_invocations=effective_invocations,
        sharing_savings=sharing_savings,
        p95_control_latency_ms=round(latency, 6),
        timely_recall=round(recall, 6),
        cpu_time_seconds=round(cpu_seconds, 6),
        peak_memory_bytes=memory_bytes,
        network_bytes=network_bytes,
        network_profile_id=run.network_profile_id,
        network_round_trip_latency_ms=network["round_trip_latency_ms"],
        target_timely_recall=round(target_recall, 6),
        maximum_p95_control_latency_ms=run.maximum_p95_control_latency_ms,
        slo_satisfied=(
            completed
            and recall >= target_recall
            and latency <= run.maximum_p95_control_latency_ms
        ),
    )
    _write_common_records(run, result, output_root)
    return result


def _write_common_records(
    run: PlannedScalingRun,
    result: ProfiledScalingResult,
    output_root: str | Path,
) -> None:
    run_dir = Path(output_root) / run.run_id
    store = JsonlEventStore(run_dir)
    observed_at = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
        seconds=run.random_seed
    )
    request_id = f"{run.run_id}:aggregate"
    common = {
        "run_id": run.run_id,
        "baseline_id": run.baseline_id,
        "trace_id": f"e8:{run.point.model_dump_json()}",
        "request_id": request_id,
        "event_time": observed_at,
        "monotonic_timestamp_ns": run.random_seed,
    }
    store.append(
        PlanDecision(
            **common,
            decision_id=deterministic_id("e8_decision", run.run_id, length=32),
            checkpoint_id="profiled-scaling",
            planning_scope="PROFILE_DRIVEN_AGGREGATE",
            planning_latency_ms=result.p95_control_latency_ms,
            labels_generated=result.generated_labels,
            labels_retained=result.effective_provider_invocations,
            selected_alternative_ids=tuple(
                f"aggregate-alternative-{index}"
                for index in range(min(result.effective_provider_invocations, 32))
            ),
            reason="E8 deterministic profile-driven execution",
            metadata={
                "profile_id": result.profile_id,
                "calibrated": result.calibrated,
                "logical_demands": result.logical_demands,
                "sharing_savings": result.sharing_savings,
            },
        )
    )
    store.append(
        ResourceSample(
            **common,
            node_id="x86server",
            cpu_utilization=min(1.0, result.cpu_time_seconds / 60),
            cpu_time_seconds=result.cpu_time_seconds,
            memory_bytes=result.peak_memory_bytes,
            network_tx_bytes=result.network_bytes,
            metadata={
                "measurement_kind": "profile_driven",
                "profile_id": result.profile_id,
            },
        )
    )
    (run_dir / "scaling_result.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _deterministic_jitter(run_id: str, seed: int) -> float:
    token = deterministic_id("e8_jitter", {"run_id": run_id, "seed": seed})
    bucket = int(token[-8:], 16) % 101
    return 0.95 + bucket / 1000
