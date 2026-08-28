"""Planner-visible compute-capacity disturbance profiles."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

from pydantic import Field

from evaluation.disturbance_schedule import (
    DisturbanceKind,
    ScheduledDisturbanceAction,
)
from evaluation.schemas import BaselineId, ResourceSample
from fable.common.base import FrozenFableModel
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import ComputeCapacity


class ComputeCapacityProfile(FrozenFableModel):
    profile_id: str = Field(min_length=1)
    cpu_capacity_fraction: float = Field(gt=0, le=1)
    memory_capacity_fraction: float = Field(gt=0, le=1)
    gpu_capacity_fraction: float = Field(gt=0, le=1)
    execution_time_multiplier: float = Field(default=1, ge=1)
    queue_delay_ms: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class AppliedComputeCapacityProfile:
    profile_id: str
    resource_epoch: int
    target_node_id: str
    deployment: DeploymentGraph
    execution_time_multiplier: float
    queue_delay_ms: int


def apply_compute_capacity_profile(
    deployment: DeploymentGraph,
    *,
    target_node_id: str,
    profile: ComputeCapacityProfile,
    resource_epoch: int,
) -> AppliedComputeCapacityProfile:
    if target_node_id not in deployment.nodes:
        raise ValueError(f"unknown capacity target node: {target_node_id}")
    pool_id, capacity = deployment.resource_pool(target_node_id)
    reservation = ComputeCapacity(
        cpu_cores=capacity.cpu_cores * (1 - profile.cpu_capacity_fraction),
        memory_mb=round(capacity.memory_mb * (1 - profile.memory_capacity_fraction)),
        gpu_memory_mb=round(
            capacity.gpu_memory_mb * (1 - profile.gpu_capacity_fraction)
        ),
    )
    adjusted = deployment.with_resource_reservations({pool_id: reservation})
    return AppliedComputeCapacityProfile(
        profile_id=profile.profile_id,
        resource_epoch=resource_epoch,
        target_node_id=target_node_id,
        deployment=adjusted,
        execution_time_multiplier=profile.execution_time_multiplier,
        queue_delay_ms=profile.queue_delay_ms,
    )


class ProfiledCapacityActionApplier:
    """Update planner capacity and emit the exact synthetic capacity sample."""

    def __init__(
        self,
        *,
        deployment: DeploymentGraph,
        profiles: dict[str, ComputeCapacityProfile],
        run_id: str,
        baseline_id: BaselineId,
        trace_id: str,
        request_id: str,
        record_sink,
    ) -> None:
        self.deployment = deployment
        self.profiles = dict(profiles)
        self.run_id = run_id
        self.baseline_id = baseline_id
        self.trace_id = trace_id
        self.request_id = request_id
        self.record_sink = record_sink
        self.latest: AppliedComputeCapacityProfile | None = None

    def __call__(
        self,
        action: ScheduledDisturbanceAction,
        condition_epoch: int,
    ) -> dict[str, int | float | str | bool]:
        if action.kind != DisturbanceKind.CAPACITY_PROFILE:
            raise ValueError("capacity applier accepts only CAPACITY_PROFILE actions")
        try:
            profile = self.profiles[action.condition_id]
        except KeyError as exc:
            raise ValueError(
                f"no compute-capacity profile for {action.condition_id}"
            ) from exc
        applied = apply_compute_capacity_profile(
            self.deployment,
            target_node_id=action.target_id,
            profile=profile,
            resource_epoch=condition_epoch,
        )
        _, capacity = applied.deployment.resource_pool(action.target_id)
        self.record_sink(
            ResourceSample(
                run_id=self.run_id,
                baseline_id=self.baseline_id,
                trace_id=self.trace_id,
                request_id=self.request_id,
                event_time=action.due_at,
                monotonic_timestamp_ns=perf_counter_ns(),
                node_id=action.target_id,
                memory_bytes=capacity.memory_mb * 1024 * 1024,
                gpu_memory_bytes=capacity.gpu_memory_mb * 1024 * 1024,
                metadata={
                    "measurement_kind": "profiled_capacity",
                    "capacity_profile": profile.profile_id,
                    "condition_epoch": condition_epoch,
                    "cpu_cores": capacity.cpu_cores,
                    "execution_time_multiplier": profile.execution_time_multiplier,
                    "queue_delay_ms": profile.queue_delay_ms,
                },
            )
        )
        self.latest = applied
        return {
            "profile_id": profile.profile_id,
            "cpu_cores": capacity.cpu_cores,
            "memory_mb": capacity.memory_mb,
            "gpu_memory_mb": capacity.gpu_memory_mb,
            "execution_time_multiplier": profile.execution_time_multiplier,
            "queue_delay_ms": profile.queue_delay_ms,
        }
