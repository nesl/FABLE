"""Synthetic, replay-free validation of network-sensitive physical plans.

The timing multipliers in this module are experimental hypotheses, not E0
measurements.  They let a campaign prove that the planner can observe a cost
crossover before the corresponding hardware tiers have been calibrated.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from fable.common.base import FrozenFableModel
from fable.planning.models import PhysicalAlternative, ProviderResourceProfile
from providers.vehicle.profiling import load_profile_records


class TimingMultiplier(FrozenFableModel):
    node_class: str = Field(min_length=1)
    provider_ids: tuple[str, ...] = ()
    startup_multiplier: float = Field(default=1.0, gt=0)
    execution_multiplier: float = Field(default=1.0, gt=0)


class ExpectedPlan(FrozenFableModel):
    profile_id: str = Field(min_length=1)
    detector_node_class: str = Field(pattern=r"^(sensor|server)$")
    maximum_transfer_bytes: int | None = Field(default=None, ge=0)
    provider_node_classes: dict[str, str] = Field(default_factory=dict)


class PlanDiscriminationManifest(FrozenFableModel):
    schema_version: str = "fable.plan_discrimination.v1"
    trace_id: str = Field(min_length=1)
    variant_override: str | None = None
    profiles: tuple[str, ...]
    policies: tuple[str, ...]
    repetitions: int = Field(default=1, ge=1)
    permit_synthetic_raw_transfer: bool = False
    semantic_frontier_index: int = Field(default=0, ge=0)
    minimum_b3_resource_reduction_fraction: float = Field(default=0.0, ge=0, le=1)
    target_chain_id: str | None = None
    two_camera_fixture: bool = False
    include_chain_ids: tuple[str, ...] = ()
    timing_multipliers: tuple[TimingMultiplier, ...] = ()
    expected_plans: tuple[ExpectedPlan, ...]

    @model_validator(mode="after")
    def _validate_profile_contract(self) -> "PlanDiscriminationManifest":
        expected = [item.profile_id for item in self.expected_plans]
        if len(expected) != len(set(expected)):
            raise ValueError("expected plan profile IDs must be unique")
        if set(expected) != {Path(item).stem for item in self.profiles}:
            raise ValueError("every network profile needs exactly one expected plan")
        return self


def load_plan_discrimination_manifest(path: str | Path) -> PlanDiscriminationManifest:
    return PlanDiscriminationManifest.model_validate(
        yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    )


def load_synthetic_profiles(
    profile_path: str | Path,
    multipliers: tuple[TimingMultiplier, ...],
) -> tuple[ProviderResourceProfile, ...]:
    """Apply declared tier assumptions without altering measured E0 profiles."""

    result = []
    for record in load_profile_records(profile_path):
        profile = record.to_planner_profile()
        startup_factor = execution_factor = 1.0
        for rule in multipliers:
            if rule.node_class != profile.node_class:
                continue
            if rule.provider_ids and profile.provider_id not in rule.provider_ids:
                continue
            startup_factor *= rule.startup_multiplier
            execution_factor *= rule.execution_multiplier
        result.append(
            profile.model_copy(
                update={
                    "startup_ms": round(profile.startup_ms * startup_factor),
                    "execution_ms": round(profile.execution_ms * execution_factor),
                }
            )
        )
    return tuple(result)


def alternative_signature(alternative: PhysicalAlternative) -> dict[str, Any]:
    placements = [
        {
            "step_id": step.step_id,
            "provider_id": step.provider_id,
            "node_id": step.node_id,
            "node_class": step.node_class,
        }
        for step in alternative.step_placements
    ]
    detector = next(
        (item for item in placements if item["provider_id"].startswith("yolo_")),
        None,
    )
    return {
        "alternative_id": alternative.alternative_id,
        "chain_id": alternative.chain_id,
        "placements": placements,
        "detector_node_class": detector["node_class"] if detector else None,
        "detector_node_id": detector["node_id"] if detector else None,
        "completion_ms": alternative.estimated_completion_ms,
        "transfer_bytes": alternative.estimated_transfer_bytes,
        "path_node_ids": sorted(
            {node for transfer in alternative.transfers for node in transfer.path_node_ids}
        ),
    }


def validate_expected_plans(
    rows: list[dict[str, Any]],
    manifest: PlanDiscriminationManifest,
) -> dict[str, Any]:
    expected = {item.profile_id: item for item in manifest.expected_plans}
    failures = []
    by_policy: dict[str, set[tuple[str | None, str | None]]] = defaultdict(set)
    for row in rows:
        contract = expected[str(row["profile_id"])]
        by_policy[str(row["policy_id"])].add(
            tuple(
                sorted(
                    (str(item.get("provider_id")), str(item.get("node_id")))
                    for item in (row.get("all_placements") or row.get("placements") or ())
                    if isinstance(item, dict)
                )
            )
        )
        if row.get("detector_node_class") != contract.detector_node_class:
            failures.append(
                f"{row['profile_id']}/{row['policy_id']}: expected detector on "
                f"{contract.detector_node_class}, got {row.get('detector_node_class')}"
            )
        placements = row.get("all_placements") or row.get("placements") or ()
        actual_provider_classes: dict[str, set[str]] = defaultdict(set)
        for placement in placements:
            if isinstance(placement, dict):
                actual_provider_classes[str(placement.get("provider_id"))].add(
                    str(placement.get("node_class"))
                )
        for provider_id, expected_class in contract.provider_node_classes.items():
            if expected_class not in actual_provider_classes.get(provider_id, set()):
                failures.append(
                    f"{row['profile_id']}/{row['policy_id']}: expected {provider_id} "
                    f"on {expected_class}, got {sorted(actual_provider_classes.get(provider_id, set()))}"
                )
        if (
            contract.maximum_transfer_bytes is not None
            and int(row.get("transfer_bytes", 0)) > contract.maximum_transfer_bytes
        ):
            failures.append(
                f"{row['profile_id']}/{row['policy_id']}: transfer budget exceeded"
            )
    insensitive = sorted(
        policy for policy, signatures in by_policy.items() if len(signatures) < 2
    )
    if insensitive:
        failures.append("placement did not change for: " + ", ".join(insensitive))
    return {
        "schema_version": "fable.plan_discrimination_result.v1",
        "valid": not failures,
        "failures": failures,
        "profile_sensitive_policies": sorted(set(by_policy) - set(insensitive)),
        "sensor_recordings_replayed": False,
        "timings_are_synthetic": bool(manifest.timing_multipliers),
    }


def validate_transition_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the intended distinction between fixed and adaptive policies."""

    failures = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy_id"])].append(row)
    for policy, items in grouped.items():
        ordered = sorted(items, key=lambda item: int(item["resource_epoch"]))
        placements = [
            (
                item.get("chain_id"),
                (
                    tuple(
                        (step.get("provider_id"), step.get("node_id"))
                        for step in (item.get("placements") or ())
                        if isinstance(step, dict)
                    )
                    or (("detector", item.get("detector_node_id")),)
                ),
            )
            for item in ordered
        ]
        if policy == "B2_FRONTIER_FIXED_REALIZATION":
            if len(set(placements)) != 1:
                failures.append("B2 changed its admission-time realization")
        elif policy in {"B3_TASK_RESOURCE_ADAPTIVE", "FABLE"}:
            if len(set(placements)) < 2:
                failures.append(f"{policy} did not adapt its realization")
    return {
        "valid": not failures,
        "failures": failures,
    }
