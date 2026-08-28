"""Immutable E0 provider/network calibration manifests."""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path

from pydantic import Field

from fable.common.base import FrozenFableModel
from fable.common.ids import deterministic_id
from fable.common.enums import ProviderPortKind
from fable.planning.deployment import DeploymentGraph
from fable.planning.provider_registry import ProviderRegistry


class CalibrationTarget(FrozenFableModel):
    target_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    tier: str = Field(min_length=1)
    input_class: str = Field(min_length=1)


class PlannedCalibrationRun(FrozenFableModel):
    schema_version: str = "fable.planned_calibration_run.v1"
    run_id: str
    target: CalibrationTarget
    invocation_kind: str
    repetition: int = Field(ge=1)
    random_seed: int = Field(ge=0)
    network_profile_id: str


class CalibrationObservation(FrozenFableModel):
    run_id: str = Field(min_length=1)
    target: CalibrationTarget
    invocation_kind: str = Field(pattern=r"^(warm|cold)$")
    startup_ms: float = Field(ge=0)
    execution_ms: float = Field(ge=0)
    quality_score: float = Field(ge=0, le=1)
    ambiguity_score: float = Field(ge=0, le=2)
    successful: bool = True


class CalibratedProviderTierProfile(FrozenFableModel):
    schema_version: str = "fable.calibrated_provider_tier_profile.v1"
    provider_id: str
    tier: str
    input_class: str
    sample_count: int = Field(ge=1)
    success_rate: float = Field(ge=0, le=1)
    cold_startup_p95_ms: float = Field(ge=0)
    warm_execution_p50_ms: float = Field(ge=0)
    warm_execution_p95_ms: float = Field(ge=0)
    quality_mean: float = Field(ge=0, le=1)
    ambiguity_p95: float = Field(ge=0, le=2)


def summarize_observations(
    observations: tuple[CalibrationObservation, ...],
) -> tuple[CalibratedProviderTierProfile, ...]:
    grouped: dict[tuple[str, str, str], list[CalibrationObservation]] = {}
    for item in observations:
        key = (item.target.provider_id, item.target.tier, item.target.input_class)
        grouped.setdefault(key, []).append(item)
    profiles = []
    for (provider_id, tier, input_class), rows in sorted(grouped.items()):
        cold = sorted(
            item.startup_ms for item in rows if item.invocation_kind == "cold"
        )
        warm = sorted(
            item.execution_ms for item in rows if item.invocation_kind == "warm"
        )
        if not cold or not warm:
            raise ValueError(
                f"calibration group requires warm and cold observations: "
                f"{provider_id}/{tier}/{input_class}"
            )
        profiles.append(
            CalibratedProviderTierProfile(
                provider_id=provider_id,
                tier=tier,
                input_class=input_class,
                sample_count=len(rows),
                success_rate=sum(item.successful for item in rows) / len(rows),
                cold_startup_p95_ms=_percentile(cold, 0.95),
                warm_execution_p50_ms=_percentile(warm, 0.50),
                warm_execution_p95_ms=_percentile(warm, 0.95),
                quality_mean=sum(item.quality_score for item in rows) / len(rows),
                ambiguity_p95=_percentile(
                    sorted(item.ambiguity_score for item in rows),
                    0.95,
                ),
            )
        )
    return tuple(profiles)


def _percentile(values: list[float], fraction: float) -> float:
    return values[int(round((len(values) - 1) * fraction))]


def targets_from_inventory(
    registry: ProviderRegistry,
    deployment: DeploymentGraph,
) -> tuple[CalibrationTarget, ...]:
    """Enumerate every feasible provider/tier/input combination structurally."""

    targets = []
    for provider in registry.providers.values():
        nodes = deployment.candidate_nodes(
            required_capabilities=provider.required_node_capabilities
        )
        if provider.eligible_node_classes:
            allowed = set(provider.eligible_node_classes)
            nodes = tuple(node for node in nodes if node.node_class in allowed)
        tiers = sorted({node.node_class for node in nodes})
        input_ports = tuple(
            port
            for port in provider.ports
            if port.kind in (ProviderPortKind.INPUT, ProviderPortKind.STATE_INPUT)
        )
        required = tuple(
            sorted({port.data_type for port in input_ports if port.required})
        )
        optional = tuple(
            sorted({port.data_type for port in input_ports if not port.required})
        )
        base_signature = "+".join(required) or "no_external_input"
        inputs = [base_signature]
        # Optional inputs are calibrated one-at-a-time in addition to the
        # required bundle. This isolates their incremental cost without an
        # unbounded optional-input Cartesian product.
        inputs.extend(
            "+".join((*required, item)) if required else item
            for item in optional
        )
        for tier, input_class in product(tiers, inputs):
            targets.append(
                CalibrationTarget(
                    target_id=deterministic_id(
                        "calibration_target",
                        {
                            "provider": provider.provider_id,
                            "tier": tier,
                            "input": input_class,
                        },
                        length=24,
                    ),
                    provider_id=provider.provider_id,
                    tier=tier,
                    input_class=input_class,
                )
            )
    return tuple(
        sorted(
            targets,
            key=lambda item: (item.provider_id, item.tier, item.input_class),
        )
    )


def write_manifest(
    runs: tuple[PlannedCalibrationRun, ...],
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "schema_version": "fable.calibration_manifest.v1",
                "run_count": len(runs),
                "runs": [
                    item.model_dump(mode="json")
                    for item in sorted(runs, key=lambda row: row.run_id)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def write_profiles(
    profiles: tuple[CalibratedProviderTierProfile, ...],
    path: str | Path,
) -> Path:
    """Persist measured profiles separately from the immutable run plan."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        profiles,
        key=lambda item: (item.provider_id, item.tier, item.input_class),
    )
    destination.write_text(
        json.dumps(
            {
                "schema_version": "fable.calibrated_provider_profiles.v1",
                "profile_count": len(ordered),
                "profiles": [item.model_dump(mode="json") for item in ordered],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def build(
    targets: tuple[CalibrationTarget, ...],
    *,
    network_profiles: tuple[str, ...] = ("good_network",),
    warm_repetitions: int = 30,
    cold_repetitions: int = 10,
    seed: int = 0,
) -> tuple[PlannedCalibrationRun, ...]:
    rows: list[PlannedCalibrationRun] = []
    for target, profile, kind in product(
        targets, network_profiles, ("warm", "cold")
    ):
        count = warm_repetitions if kind == "warm" else cold_repetitions
        for repetition in range(1, count + 1):
            identity = {
                "target": target,
                "profile": profile,
                "kind": kind,
                "repetition": repetition,
                "seed": seed + repetition - 1,
            }
            rows.append(
                PlannedCalibrationRun(
                    run_id=deterministic_id(
                        "calibration_run", identity, length=32
                    ),
                    target=target,
                    invocation_kind=kind,
                    repetition=repetition,
                    random_seed=seed + repetition - 1,
                    network_profile_id=profile,
                )
            )
    return tuple(sorted(rows, key=lambda item: item.run_id))
