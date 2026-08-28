"""Fail-closed promotion of E0 observations into desktop-x86 profiles."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from pydantic import Field

from evaluation.experiments.e0_calibration import (
    CalibrationObservation,
    CalibrationTarget,
    CalibratedProviderTierProfile,
    summarize_observations,
)
from fable.common.base import FrozenFableModel
from fable.common.time import ensure_utc


class CalibrationPromotionPolicy(FrozenFableModel):
    schema_version: str = "fable.calibration_promotion_policy.v1"
    host_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    physical_host_class: str = "desktop_x86"
    minimum_warm_samples: int = Field(default=30, ge=1)
    minimum_cold_samples: int = Field(default=10, ge=1)
    minimum_success_rate: float = Field(default=0.95, ge=0, le=1)


class CalibrationPromotionReport(FrozenFableModel):
    schema_version: str = "fable.calibration_promotion_report.v1"
    host_id: str
    physical_host_class: str
    generated_at: datetime
    promotable: bool
    required_target_count: int
    promoted_profile_count: int
    missing_targets: tuple[str, ...] = ()
    insufficient_targets: tuple[str, ...] = ()
    non_measured_providers: tuple[str, ...] = ()
    profiles: tuple[CalibratedProviderTierProfile, ...] = ()


def promote_desktop_observations(
    observations: tuple[CalibrationObservation, ...],
    required_targets: tuple[CalibrationTarget, ...],
    *,
    worker_operations: dict[str, dict[str, Any]],
    policy: CalibrationPromotionPolicy,
    generated_at: datetime,
) -> CalibrationPromotionReport:
    """Promote only complete, production-measured target groups."""

    generated_at = ensure_utc(generated_at)
    required = {
        (item.provider_id, item.tier, item.input_class): item
        for item in required_targets
    }
    grouped: dict[tuple[str, str, str], list[CalibrationObservation]] = {}
    for item in observations:
        key = (
            item.target.provider_id,
            item.target.tier,
            item.target.input_class,
        )
        if key in required:
            grouped.setdefault(key, []).append(item)

    missing = []
    insufficient = []
    non_measured = []
    accepted: list[CalibrationObservation] = []
    for key, target in sorted(required.items()):
        label = "/".join(key)
        capability = worker_operations.get(target.provider_id, {})
        if (
            capability.get("measurement_status") != "MEASURED_PROVIDER"
            or target.input_class not in capability.get("input_classes", ())
        ):
            non_measured.append(label)
            continue
        rows = grouped.get(key, [])
        if not rows:
            missing.append(label)
            continue
        kinds = Counter(item.invocation_kind for item in rows)
        success_rate = sum(item.successful for item in rows) / len(rows)
        if (
            kinds["warm"] < policy.minimum_warm_samples
            or kinds["cold"] < policy.minimum_cold_samples
            or success_rate < policy.minimum_success_rate
        ):
            insufficient.append(label)
            continue
        accepted.extend(rows)

    promotable = not (missing or insufficient or non_measured)
    profiles = summarize_observations(tuple(accepted)) if promotable else ()
    return CalibrationPromotionReport(
        host_id=policy.host_id,
        physical_host_class=policy.physical_host_class,
        generated_at=generated_at,
        promotable=promotable,
        required_target_count=len(required),
        promoted_profile_count=len(profiles),
        missing_targets=tuple(missing),
        insufficient_targets=tuple(insufficient),
        non_measured_providers=tuple(non_measured),
        profiles=profiles,
    )
