"""Typed qualitative spatial-knowledge records used by FABLE planning.

The site model is deliberately treated as qualitative topology, not calibrated
camera geometry.  It produces ranked candidate observation groups and never
asserts that an object must appear at a predicted sensor.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from fable.common.base import FableModel


class SpatialMatchKind(StrEnum):
    DIRECTIONAL_RULE = "DIRECTIONAL_RULE"
    CORRIDOR = "CORRIDOR"
    NONE = "NONE"


class SpatialFilterMode(StrEnum):
    """How a spatial prediction affects source eligibility."""

    PREFER = "PREFER"
    LIMIT_TO_PREDICTED = "LIMIT_TO_PREDICTED"


class SpatialSensor(FableModel):
    sensor_id: str = Field(min_length=1)
    position: tuple[float, float]
    coverage_zones: tuple[str, ...] = ()
    camera_facing_approx: str | None = None
    confidence: str = "medium"
    microphone: bool = False
    fixed: bool = True
    deployment_ids: tuple[str, ...] = ()


class SpatialObservationGroup(FableModel):
    sensor_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _nonempty_unique(self) -> "SpatialObservationGroup":
        if not self.sensor_ids:
            raise ValueError("spatial observation group cannot be empty")
        if len(self.sensor_ids) != len(set(self.sensor_ids)):
            raise ValueError("spatial observation group sensor IDs must be unique")
        return self


class SpatialCorridor(FableModel):
    corridor_id: str = Field(min_length=1)
    from_zone: str = Field(min_length=1)
    to_zone: str = Field(min_length=1)
    reverse_supported: bool = False
    motion_forward: str = ""
    forward_groups: tuple[SpatialObservationGroup, ...]
    reverse_groups: tuple[SpatialObservationGroup, ...] = ()
    mobile_augmentations: dict[str, tuple[SpatialObservationGroup, ...]] = Field(
        default_factory=dict
    )
    confidence: str = "medium"
    note: str = ""


class SpatialNextSensorCandidate(FableModel):
    sensor_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    confidence: str = "medium"
    deployment_ids: tuple[str, ...] = ()


class SpatialDirectionalRule(FableModel):
    current_sensor_id: str = Field(min_length=1)
    observed_headings: tuple[str, ...]
    likely_next: tuple[SpatialNextSensorCandidate, ...]
    corridor_id: str | None = None

    @model_validator(mode="after")
    def _require_rule(self) -> "SpatialDirectionalRule":
        if not self.observed_headings:
            raise ValueError("directional rule requires at least one heading")
        if not self.likely_next:
            raise ValueError("directional rule requires at least one next sensor")
        return self


class SpatialTransitionModel(FableModel):
    schema_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_type: str = Field(min_length=1)
    zones: dict[str, tuple[float, float]] = Field(default_factory=dict)
    sensors: dict[str, SpatialSensor]
    mobile_deployments: tuple[str, ...] = ()
    corridors: dict[str, SpatialCorridor]
    directional_rules: tuple[SpatialDirectionalRule, ...]
    assumptions: tuple[str, ...] = ()
    known_issues: tuple[str, ...] = ()


class SpatialSensorBindings(FableModel):
    """Maps topology sensor IDs to runtime source and node IDs.

    The topology uses stable sensor IDs such as ``orin_6`` while a deployment
    may expose a camera as ``orin6_camera`` on node ``dvpg_gq_orin_6``.
    """

    source_ids_by_sensor: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    node_ids_by_sensor: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    source_ids_by_deployment: dict[str, dict[str, tuple[str, ...]]] = Field(
        default_factory=dict
    )
    node_ids_by_deployment: dict[str, dict[str, tuple[str, ...]]] = Field(
        default_factory=dict
    )

    def sources(
        self, sensor_id: str, deployment_id: str | None = None
    ) -> tuple[str, ...]:
        deployment_values = (
            self.source_ids_by_deployment.get(deployment_id, {}).get(sensor_id, ())
            if deployment_id is not None
            else ()
        )
        return tuple(
            dict.fromkeys((*self.source_ids_by_sensor.get(sensor_id, ()), *deployment_values))
        )

    def nodes(
        self, sensor_id: str, deployment_id: str | None = None
    ) -> tuple[str, ...]:
        deployment_values = (
            self.node_ids_by_deployment.get(deployment_id, {}).get(sensor_id, ())
            if deployment_id is not None
            else ()
        )
        return tuple(
            dict.fromkeys((*self.node_ids_by_sensor.get(sensor_id, ()), *deployment_values))
        )

    def sensor_for_source(
        self, source_id: str, deployment_id: str | None = None
    ) -> str | None:
        candidates = dict(self.source_ids_by_sensor)
        if deployment_id is not None:
            candidates.update(self.source_ids_by_deployment.get(deployment_id, {}))
        matches = [
            sensor_id
            for sensor_id, source_ids in candidates.items()
            if source_id in source_ids
        ]
        if len(matches) > 1:
            raise ValueError(f"runtime source {source_id} maps to multiple topology sensors")
        return matches[0] if matches else None


class SpatialObservation(FableModel):
    current_sensor_id: str = Field(min_length=1)
    observed_heading: str | None = None
    active_deployment_id: str | None = None
    corridor_id: str | None = None
    branch_unresolved: bool = False
    maximum_observation_groups: int = Field(default=1, ge=1, le=8)
    filter_mode: SpatialFilterMode = SpatialFilterMode.PREFER
    object_binding_id: str | None = None


class PredictedObservationGroup(FableModel):
    group_rank: int = Field(ge=1)
    sensor_ids: tuple[str, ...]
    source_ids: tuple[str, ...] = ()
    node_ids: tuple[str, ...] = ()
    confidence: str = "medium"
    confidence_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


class SpatialPrediction(FableModel):
    prediction_id: str = Field(min_length=1)
    match_kind: SpatialMatchKind
    current_sensor_id: str = Field(min_length=1)
    normalized_heading: str | None = None
    active_deployment_id: str | None = None
    corridor_id: str | None = None
    groups: tuple[PredictedObservationGroup, ...] = ()
    recommended_source_ids: tuple[str, ...] = ()
    wake_node_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class SpatialCheckpointGuidance(FableModel):
    """Spatial output consumed at a semantic checkpoint."""

    prediction: SpatialPrediction
    graph_node_ids: tuple[str, ...]
    reason: Literal["FRONTIER_CHANGED", "MANUAL_LOOKUP"] = "FRONTIER_CHANGED"
