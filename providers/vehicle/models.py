"""Typed vehicle-provider records.

These payloads are deliberately independent of one concrete model. They are the
serializable runtime form behind the catalog's ``detection_set.v1``,
``track_set.v1``, projected tracks, descriptor sets, associations, and
predicate observations.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from fable.common.base import FableModel, JSONValue
from fable.common.ids import deterministic_id
from fable.common.time import EventTimeInterval, ensure_utc


class CoordinateSpace(StrEnum):
    IMAGE = "IMAGE"
    WORLD = "WORLD"
    ROUTE = "ROUTE"


class BoundingBox(FableModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bounding-box maximum coordinates must not precede minima")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @classmethod
    def from_xywh(cls, xywh: list[float] | tuple[float, float, float, float]) -> "BoundingBox":
        cx, cy, width, height = (float(value) for value in xywh)
        return cls(
            x1=cx - width / 2.0,
            y1=cy - height / 2.0,
            x2=cx + width / 2.0,
            y2=cy + height / 2.0,
        )


class Point2D(FableModel):
    x: float
    y: float
    coordinate_frame_id: str = Field(min_length=1)


class Detection(FableModel):
    detection_id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    world_point: Point2D | None = None
    attributes: dict[str, JSONValue] = Field(default_factory=dict)


class DetectionFrame(FableModel):
    schema_version: Literal["detection_set.v1"] = "detection_set.v1"
    source_id: str = Field(min_length=1)
    event_time: datetime
    frame_id: str = Field(min_length=1)
    image_width: int | None = Field(default=None, ge=1)
    image_height: int | None = Field(default=None, ge=1)
    detector_id: str = Field(min_length=1)
    detector_version: str = Field(min_length=1)
    detections: tuple[Detection, ...] = ()
    source_sequence: int | None = Field(default=None, ge=0)

    @field_validator("event_time")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _unique_detection_ids(self) -> Self:
        ids = [item.detection_id for item in self.detections]
        if len(ids) != len(set(ids)):
            raise ValueError("detection IDs must be unique within a frame")
        return self


class TrackObservation(FableModel):
    local_track_id: int = Field(ge=0)
    scoped_track_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    tracker_session_id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    event_time: datetime
    world_point: Point2D | None = None
    route_id: str | None = None
    route_progress_m: float | None = None
    velocity_mps: float | None = None
    age_frames: int = Field(default=1, ge=1)
    attributes: dict[str, JSONValue] = Field(default_factory=dict)

    @field_validator("event_time")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _scope_identity(self) -> Self:
        expected = scoped_track_identity(
            self.source_id,
            self.tracker_session_id,
            self.local_track_id,
        )
        if self.scoped_track_id != expected:
            raise ValueError("scoped_track_id must use source/session/local identity")
        return self


class TrackSet(FableModel):
    schema_version: Literal["track_set.v1"] = "track_set.v1"
    source_id: str = Field(min_length=1)
    tracker_family: str = Field(min_length=1)
    tracker_version: str = Field(min_length=1)
    tracker_session_id: str = Field(min_length=1)
    event_time: datetime
    tracks: tuple[TrackObservation, ...] = ()
    reconstructed_from_detection_replay: bool = False
    replay_interval: EventTimeInterval | None = None

    @field_validator("event_time")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _source_alignment(self) -> Self:
        for track in self.tracks:
            if track.source_id != self.source_id:
                raise ValueError("all tracks must belong to the TrackSet source")
            if track.tracker_session_id != self.tracker_session_id:
                raise ValueError("all tracks must belong to the TrackSet session")
        if self.reconstructed_from_detection_replay and self.replay_interval is None:
            raise ValueError("reconstructed track sets require replay_interval")
        return self


class VehicleZone(FableModel):
    zone_id: str = Field(min_length=1)
    coordinate_frame_id: str = Field(min_length=1)
    polygon: tuple[Point2D, ...]

    @model_validator(mode="after")
    def _polygon_valid(self) -> Self:
        if len(self.polygon) < 3:
            raise ValueError("zone polygon requires at least three vertices")
        if any(point.coordinate_frame_id != self.coordinate_frame_id for point in self.polygon):
            raise ValueError("zone vertices must use the zone coordinate frame")
        return self


class PredicateObservation(FableModel):
    schema_version: Literal["vehicle_predicate_observation.v1"] = (
        "vehicle_predicate_observation.v1"
    )
    occurrence_id: str = Field(min_length=1)
    predicate_id: str = Field(min_length=1)
    truth: bool
    confidence: float = Field(ge=0.0, le=1.0)
    event_time_interval: EventTimeInterval
    bindings: dict[str, str] = Field(default_factory=dict)
    source_ids: tuple[str, ...] = ()
    provider_id: str = Field(min_length=1)
    provider_version: str = Field(min_length=1)
    supporting_artifact_types: tuple[str, ...] = ()
    measurements: dict[str, JSONValue] = Field(default_factory=dict)


class DescriptorRecord(FableModel):
    local_entity_id: str = Field(min_length=1)
    vector: tuple[float, ...]
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    source_crop_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _nonempty_vector(self) -> Self:
        if not self.vector:
            raise ValueError("descriptor vector cannot be empty")
        return self


class DescriptorSet(FableModel):
    schema_version: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    event_time_interval: EventTimeInterval
    descriptor_kind: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    preprocessing_id: str = Field(min_length=1)
    dimension: int = Field(ge=1)
    normalization: str = Field(min_length=1)
    distance_metric: str = Field(min_length=1)
    records: tuple[DescriptorRecord, ...]
    calibrated_for_identity: bool = False

    @model_validator(mode="after")
    def _dimensions(self) -> Self:
        for record in self.records:
            if len(record.vector) != self.dimension:
                raise ValueError("descriptor dimension does not match metadata")
        return self

    @property
    def compatibility_key(self) -> tuple[str, str, str, int, str, str]:
        return (
            self.model_id,
            self.model_version,
            self.preprocessing_id,
            self.dimension,
            self.normalization,
            self.distance_metric,
        )


class EntityAssociation(FableModel):
    left_local_entity_id: str = Field(min_length=1)
    right_local_entity_id: str = Field(min_length=1)
    canonical_entity_id: str = Field(min_length=1)
    distance: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    route_time_compatible: bool = True


class EntityAssociationSet(FableModel):
    schema_version: Literal["canonical_entity_map.v1"] = "canonical_entity_map.v1"
    left_source_id: str = Field(min_length=1)
    right_source_id: str = Field(min_length=1)
    event_time_interval: EventTimeInterval
    feature_space_key: tuple[str, str, str, int, str, str]
    associations: tuple[EntityAssociation, ...]
    unmatched_left: tuple[str, ...] = ()
    unmatched_right: tuple[str, ...] = ()


def scoped_track_identity(source_id: str, tracker_session_id: str, local_track_id: int) -> str:
    """Return the only globally safe identity for a sensor-local tracker ID."""

    return f"{source_id}:{tracker_session_id}:{int(local_track_id)}"


def occurrence_id(
    predicate_id: str,
    bindings: dict[str, str],
    interval: EventTimeInterval,
    provider_id: str,
) -> str:
    return deterministic_id(
        "vehicle_occurrence",
        {
            "predicate_id": predicate_id,
            "bindings": bindings,
            "interval": interval,
            "provider_id": provider_id,
        },
        length=40,
    )
