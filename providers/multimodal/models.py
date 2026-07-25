"""Typed records for Phase-8 audio and interaction providers.

The records in this module are provider-side evidence contracts.  They never
advance a complex-event graph directly; the node agent validates them against
an active :class:`~fable.common.schemas.PredicateDemand` before forwarding a
semantic result to the orchestrator.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from fable.common.base import FableModel, JSONValue
from fable.common.ids import deterministic_id
from fable.common.time import EventTimeInterval, ensure_utc


class AudioSampleEncoding(StrEnum):
    FLOAT32_NORMALIZED = "FLOAT32_NORMALIZED"
    PCM16 = "PCM16"


class AudioWindow(FableModel):
    """Bounded channels-first audio in event time.

    The replay adapter normalizes PCM16 into ``[-1, 1]`` before constructing
    this record.  Keeping the pure Python schema independent of NumPy lets the
    contract be serialized and unit tested without the optional audio extra.
    """

    schema_version: Literal["audio_segment.v1"] = "audio_segment.v1"
    source_id: str = Field(min_length=1)
    event_time_interval: EventTimeInterval
    sample_rate_hz: int = Field(ge=1)
    channel_ids: tuple[str, ...]
    waveform: tuple[tuple[float, ...], ...]
    encoding: AudioSampleEncoding = AudioSampleEncoding.FLOAT32_NORMALIZED
    source_sequence: int | None = Field(default=None, ge=0)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_shape(self) -> Self:
        if not self.channel_ids:
            raise ValueError("audio window requires at least one channel")
        if len(self.channel_ids) != len(set(self.channel_ids)):
            raise ValueError("audio channel identifiers must be unique")
        if len(self.waveform) != len(self.channel_ids):
            raise ValueError("waveform channel count must match channel_ids")
        lengths = {len(channel) for channel in self.waveform}
        if not lengths or 0 in lengths or len(lengths) != 1:
            raise ValueError("all audio channels must contain the same non-zero sample count")
        if any(not isfinite(float(value)) for channel in self.waveform for value in channel):
            raise ValueError("audio samples must be finite")
        return self

    @property
    def sample_count(self) -> int:
        return len(self.waveform[0])

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / float(self.sample_rate_hz)


class AudioEventObservation(FableModel):
    schema_version: Literal["audio_event_observation.v1"] = "audio_event_observation.v1"
    occurrence_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    event_time_interval: EventTimeInterval
    source_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    provider_version: str = Field(min_length=1)
    source_labels: tuple[str, ...] = ()
    class_scores: dict[str, float] = Field(default_factory=dict)
    localized_zone_id: str | None = None
    attributes: dict[str, JSONValue] = Field(default_factory=dict)

    @field_validator("class_scores")
    @classmethod
    def _scores(cls, value: dict[str, float]) -> dict[str, float]:
        for label, score in value.items():
            if not label or not 0.0 <= score <= 1.0:
                raise ValueError("audio class scores require non-empty labels and values in [0, 1]")
        return value


class MicrophonePosition(FableModel):
    microphone_id: str = Field(min_length=1)
    x_m: float
    y_m: float
    z_m: float = 0.0


class MicrophoneArrayGeometry(FableModel):
    array_id: str = Field(min_length=1)
    coordinate_frame_id: str = Field(min_length=1)
    microphones: tuple[MicrophonePosition, ...]
    reference_microphone_id: str = Field(min_length=1)
    speed_of_sound_mps: float = Field(default=343.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_array(self) -> Self:
        ids = [item.microphone_id for item in self.microphones]
        if len(ids) < 2:
            raise ValueError("audio localization requires at least two microphones")
        if len(ids) != len(set(ids)):
            raise ValueError("microphone identifiers must be unique")
        if self.reference_microphone_id not in set(ids):
            raise ValueError("reference microphone is not present in the array")
        return self


class BearingZone(FableModel):
    zone_id: str = Field(min_length=1)
    minimum_azimuth_deg: float = Field(ge=-180.0, le=180.0)
    maximum_azimuth_deg: float = Field(ge=-180.0, le=180.0)

    def contains(self, azimuth_deg: float) -> bool:
        value = ((azimuth_deg + 180.0) % 360.0) - 180.0
        if self.minimum_azimuth_deg <= self.maximum_azimuth_deg:
            return self.minimum_azimuth_deg <= value <= self.maximum_azimuth_deg
        return value >= self.minimum_azimuth_deg or value <= self.maximum_azimuth_deg


class AudioLocalization(FableModel):
    schema_version: Literal["audio_localization.v1"] = "audio_localization.v1"
    localization_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    array_id: str = Field(min_length=1)
    event_time_interval: EventTimeInterval
    azimuth_deg: float = Field(ge=-180.0, le=180.0)
    elevation_deg: float | None = Field(default=None, ge=-90.0, le=90.0)
    confidence: float = Field(ge=0.0, le=1.0)
    zone_id: str | None = None
    pair_delays_seconds: dict[str, float] = Field(default_factory=dict)
    residual_seconds: float = Field(default=0.0, ge=0.0)
    provider_id: str = Field(default="gcc_phat_audio_localizer", min_length=1)
    provider_version: str = Field(default="1", min_length=1)


class VisualBearingCandidate(FableModel):
    local_entity_id: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    event_time_interval: EventTimeInterval
    azimuth_deg: float = Field(ge=-180.0, le=180.0)
    zone_id: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class AudioVisualAssociation(FableModel):
    audio_occurrence_id: str = Field(min_length=1)
    local_entity_id: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    angular_error_deg: float = Field(ge=0.0, le=180.0)
    zone_compatible: bool = True


class AudioVisualAssociationSet(FableModel):
    schema_version: Literal["audio_visual_association_set.v1"] = (
        "audio_visual_association_set.v1"
    )
    source_id: str = Field(min_length=1)
    event_time_interval: EventTimeInterval
    audio_occurrence_id: str = Field(min_length=1)
    associations: tuple[AudioVisualAssociation, ...]
    provider_id: str = Field(default="audio_visual_association", min_length=1)
    provider_version: str = Field(default="1", min_length=1)


class SpeechSegment(FableModel):
    segment_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    event_time_interval: EventTimeInterval
    speech_probability: float = Field(ge=0.0, le=1.0)
    embedding: tuple[float, ...] = ()
    transcript: str | None = None


class DiarizationTurn(FableModel):
    turn_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    event_time_interval: EventTimeInterval
    speech_probability: float = Field(ge=0.0, le=1.0)
    transcript: str | None = None
    source_segment_ids: tuple[str, ...] = ()


class SpeakerTurnSet(FableModel):
    schema_version: Literal["speaker_turn_set.v1"] = "speaker_turn_set.v1"
    source_id: str = Field(min_length=1)
    event_time_interval: EventTimeInterval
    turns: tuple[DiarizationTurn, ...]
    speaker_count: int = Field(ge=0)
    diarization_model_id: str = Field(min_length=1)
    diarization_model_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _speaker_count(self) -> Self:
        observed = len({turn.speaker_id for turn in self.turns})
        if observed != self.speaker_count:
            raise ValueError("speaker_count must equal the number of distinct turn speakers")
        return self


class InteractionPredicateObservation(FableModel):
    """Generic typed evidence for Phase-8 semantic predicates."""

    schema_version: Literal["interaction_predicate_observation.v1"] = (
        "interaction_predicate_observation.v1"
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


class CustodyRecord(FableModel):
    package_id: str = Field(min_length=1)
    holder_id: str | None = None
    holder_type: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    established_at: datetime
    last_seen_at: datetime
    previous_holder_id: str | None = None
    transfer_occurrence_id: str | None = None

    @field_validator("established_at", "last_seen_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _time_order(self) -> Self:
        if self.last_seen_at < self.established_at:
            raise ValueError("custody last_seen_at cannot precede established_at")
        if self.holder_id is None and self.holder_type is not None:
            raise ValueError("holder_type requires holder_id")
        return self


class CustodyState(FableModel):
    schema_version: Literal["custody_state.v1"] = "custody_state.v1"
    source_id: str = Field(min_length=1)
    event_time_interval: EventTimeInterval
    records: tuple[CustodyRecord, ...]
    provider_id: str = Field(default="object_transfer_reasoner", min_length=1)
    provider_version: str = Field(default="1", min_length=1)


def phase8_occurrence_id(
    predicate_id: str,
    bindings: dict[str, str],
    interval: EventTimeInterval,
    provider_id: str,
) -> str:
    return deterministic_id(
        "phase8_occurrence",
        {
            "predicate_id": predicate_id,
            "bindings": bindings,
            "interval": interval,
            "provider_id": provider_id,
        },
        length=40,
    )
