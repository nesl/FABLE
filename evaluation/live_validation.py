"""Acceptance checks for live disturbance and concurrent logging milestones."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Callable, Literal

from pydantic import Field

from evaluation.adaptation_controller import (
    AdaptationActionResult,
    AdaptationController,
)
from evaluation.disturbance_schedule import ScheduledDisturbanceAction
from evaluation.schemas import (
    ArtifactEvent,
    ProviderLeaseEvent,
    ResourceSample,
    RetrospectiveAttempt,
)
from fable.common.base import FrozenFableModel
from fable.common.time import ensure_utc


class LiveConditionValidation(FrozenFableModel):
    schema_version: Literal["fable.live_condition_validation.v1"] = "fable.live_condition_validation.v1"
    apply_result: AdaptationActionResult
    restore_result: AdaptationActionResult
    condition_measurements: dict[str, int | float | str | bool]
    restored_measurements: dict[str, int | float | str | bool]
    successful: bool


ConditionProbe = Callable[
    [ScheduledDisturbanceAction], dict[str, int | float | str | bool]
]


def validate_apply_restore(
    controller: AdaptationController,
    *,
    apply_action: ScheduledDisturbanceAction,
    restore_action: ScheduledDisturbanceAction,
    condition_probe: ConditionProbe,
    restored_probe: ConditionProbe,
    observed_at: datetime,
) -> LiveConditionValidation:
    """Apply, probe, and restore a typed condition with restoration in ``finally``."""

    observed_at = ensure_utc(observed_at)
    apply_result = controller.execute(apply_action, observed_at=observed_at)
    if not apply_result.validated:
        raise RuntimeError("condition apply was not validated")
    condition_measurements: dict[str, int | float | str | bool] = {}
    restore_result: AdaptationActionResult | None = None
    restored_measurements: dict[str, int | float | str | bool] = {}
    probe_error: Exception | None = None
    try:
        condition_measurements = condition_probe(apply_action)
    except Exception as exc:  # restoration must still execute
        probe_error = exc
    finally:
        restore_result = controller.execute(
            restore_action,
            observed_at=observed_at,
        )
        if restore_result.validated:
            restored_measurements = restored_probe(restore_action)
    if restore_result is None or not restore_result.validated:
        raise RuntimeError("condition restoration was not validated")
    if probe_error is not None:
        raise RuntimeError("condition probe failed after successful apply") from probe_error
    return LiveConditionValidation(
        apply_result=apply_result,
        restore_result=restore_result,
        condition_measurements=condition_measurements,
        restored_measurements=restored_measurements,
        successful=True,
    )


class ConcurrentLoggingValidation(FrozenFableModel):
    schema_version: Literal["fable.concurrent_logging_validation.v1"] = "fable.concurrent_logging_validation.v1"
    shared_provider_instances: tuple[str, ...]
    attributed_heartbeat_windows: int = Field(ge=0)
    retrospective_attempt_count: int = Field(ge=0)
    derived_artifact_failure_count: int = Field(ge=0)
    successful: bool


def validate_concurrent_logging(
    *,
    leases: tuple[ProviderLeaseEvent, ...],
    resources: tuple[ResourceSample, ...],
    retrospective_attempts: tuple[RetrospectiveAttempt, ...],
    artifacts: tuple[ArtifactEvent, ...],
) -> ConcurrentLoggingValidation:
    """Require real many-request sharing and joined retrospective failures."""

    requests_by_provider: dict[str, set[str]] = defaultdict(set)
    for lease in leases:
        requests_by_provider[lease.provider_instance_id].add(lease.request_id)
    shared = tuple(
        sorted(
            provider
            for provider, requests in requests_by_provider.items()
            if len(requests) >= 2
        )
    )

    heartbeat_requests: dict[tuple[str, str], set[str]] = defaultdict(set)
    heartbeat_fractions: dict[tuple[str, str], float] = defaultdict(float)
    for sample in resources:
        if sample.metadata.get("attribution") != "shared_active_demand_allocation":
            continue
        key = (
            str(sample.metadata.get("session_id", "")),
            str(sample.metadata.get("heartbeat_sequence", "")),
        )
        heartbeat_requests[key].add(sample.request_id)
        heartbeat_fractions[key] += float(
            sample.metadata.get("allocation_fraction", 0)
        )
    attributed = sum(
        len(requests) >= 2 and abs(heartbeat_fractions[key] - 1.0) <= 1e-6
        for key, requests in heartbeat_requests.items()
    )

    attempts = {item.attempt_id: item for item in retrospective_attempts}
    joined_failures = 0
    for artifact in artifacts:
        attempt_id = str(artifact.metadata.get("derived_from_attempt_id", ""))
        if (
            artifact.action.upper()
            in {"BUFFER_EXPIRED", "COMPATIBILITY_FAILURE"}
            and attempt_id in attempts
            and attempts[attempt_id].request_id == artifact.request_id
        ):
            joined_failures += 1
    successful = bool(shared) and attributed > 0 and (
        not retrospective_attempts or joined_failures > 0
    )
    return ConcurrentLoggingValidation(
        shared_provider_instances=shared,
        attributed_heartbeat_windows=attributed,
        retrospective_attempt_count=len(retrospective_attempts),
        derived_artifact_failure_count=joined_failures,
        successful=successful,
    )
