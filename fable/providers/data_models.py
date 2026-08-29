"""Small provider-internal data models.

These classes describe data exchanged *between provider implementations*.
They are intentionally separate from ``PredicateMatch``, which is the only
provider result consumed by the CE runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import hypot
from typing import Iterable


def _require_aware(value: datetime, field: str = "event_time") -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bounding-box maximum coordinates must not precede minima")

    @property
    def width(self) -> float: return self.x2 - self.x1
    @property
    def height(self) -> float: return self.y2 - self.y1
    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def iou(self, other: "BoundingBox") -> float:
        left, top = max(self.x1, other.x1), max(self.y1, other.y1)
        right, bottom = min(self.x2, other.x2), min(self.y2, other.y2)
        intersection = max(0.0, right-left) * max(0.0, bottom-top)
        union = self.area + other.area - intersection
        return 0.0 if union <= 0 else intersection / union


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One timestamped video frame from a configured source.

    ``image`` is intentionally opaque to the FABLE runtime.  In a real vision
    deployment it is typically a NumPy/OpenCV array; tests and replay adapters
    may use any object accepted by the configured detector.
    """

    source_id: str
    event_time: datetime
    image: object
    frame_id: str = ""

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        _require_aware(self.event_time)


@dataclass(frozen=True, slots=True)
class Detection:
    class_name: str
    confidence: float
    bbox: BoundingBox
    detection_id: str = ""
    world_xy: tuple[float, float] | None = None
    world_frame: str | None = None

    def __post_init__(self) -> None:
        if not self.class_name:
            raise ValueError("class_name must be non-empty")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if self.world_xy is not None and not self.world_frame:
            raise ValueError("world_frame is required when world_xy is present")


@dataclass(frozen=True, slots=True)
class DetectionFrame:
    source_id: str
    event_time: datetime
    detections: tuple[Detection, ...]
    frame_id: str = ""
    image_width: int | None = None
    image_height: int | None = None

    def __post_init__(self) -> None:
        if not self.source_id: raise ValueError("source_id must be non-empty")
        _require_aware(self.event_time)
        object.__setattr__(self, "detections", tuple(self.detections))


@dataclass(frozen=True, slots=True)
class Track:
    object_id: str
    source_id: str
    class_name: str
    confidence: float
    bbox: BoundingBox
    event_time: datetime
    world_xy: tuple[float, float] | None = None
    world_frame: str | None = None
    velocity_xy_per_s: tuple[float, float] | None = None
    age_frames: int = 1

    def __post_init__(self) -> None:
        if not self.object_id or not self.source_id or not self.class_name:
            raise ValueError("object_id, source_id, and class_name must be non-empty")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("confidence must be in [0, 1]")
        _require_aware(self.event_time)
        if self.world_xy is not None and not self.world_frame:
            raise ValueError("world_frame is required when world_xy is present")

    @property
    def position(self) -> tuple[float, float]:
        return self.world_xy if self.world_xy is not None else self.bbox.center

    @property
    def speed(self) -> float | None:
        return None if self.velocity_xy_per_s is None else hypot(*self.velocity_xy_per_s)


@dataclass(frozen=True, slots=True)
class TrackFrame:
    source_id: str
    event_time: datetime
    tracks: tuple[Track, ...]

    def __post_init__(self) -> None:
        if not self.source_id: raise ValueError("source_id must be non-empty")
        _require_aware(self.event_time)
        object.__setattr__(self, "tracks", tuple(self.tracks))
        ids: set[str] = set()
        for track in self.tracks:
            if track.source_id != self.source_id:
                raise ValueError("all tracks in a TrackFrame must use its source_id")
            if track.event_time != self.event_time:
                raise ValueError("all tracks in a TrackFrame must use its event_time")
            if track.object_id in ids:
                raise ValueError("track IDs must be unique within a frame")
            ids.add(track.object_id)

    def by_id(self) -> dict[str, Track]:
        return {track.object_id: track for track in self.tracks}


@dataclass(frozen=True, slots=True)
class AudioWindow:
    source_id: str
    event_time: datetime
    samples: tuple[float, ...]
    sample_rate_hz: int = 16_000

    def __post_init__(self) -> None:
        if not self.source_id: raise ValueError("source_id must be non-empty")
        _require_aware(self.event_time)
        if self.sample_rate_hz <= 0: raise ValueError("sample_rate_hz must be positive")
        object.__setattr__(self, "samples", tuple(float(v) for v in self.samples))


@dataclass(frozen=True, slots=True)
class MultichannelAudioWindow:
    source_id: str
    event_time: datetime
    channels: tuple[tuple[float, ...], ...]
    sample_rate_hz: int = 16_000

    def __post_init__(self) -> None:
        if not self.source_id: raise ValueError("source_id must be non-empty")
        _require_aware(self.event_time)
        if len(self.channels) < 2: raise ValueError("multichannel audio needs at least two channels")
        lengths = {len(channel) for channel in self.channels}
        if len(lengths) > 1: raise ValueError("all audio channels must have equal length")


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """A bounded interval where voice activity was detected."""
    start_time: datetime
    end_time: datetime
    confidence: float = 1.0
    def __post_init__(self) -> None:
        _require_aware(self.start_time, "start_time"); _require_aware(self.end_time, "end_time")
        if self.end_time < self.start_time: raise ValueError("segment end precedes start")


@dataclass(frozen=True, slots=True)
class SpeakerEmbedding:
    """Embedding computed for one detected speech segment."""
    segment: SpeechSegment
    vector: tuple[float, ...]
    model_id: str


@dataclass(frozen=True, slots=True)
class DiarizedSpeechSegment:
    """One speech interval assigned to a speaker by diarization.

    This replaces the less obvious old name ``SpeakerTurn``.
    """
    speaker_id: str
    start_time: datetime
    end_time: datetime
    confidence: float = 1.0
    transcript: str | None = None

    def __post_init__(self) -> None:
        if not self.speaker_id: raise ValueError("speaker_id must be non-empty")
        _require_aware(self.start_time, "start_time"); _require_aware(self.end_time, "end_time")
        if self.end_time < self.start_time: raise ValueError("speech segment end precedes start")
        if not 0 <= float(self.confidence) <= 1: raise ValueError("confidence must be in [0,1]")


@dataclass(frozen=True, slots=True)
class DiarizedSpeechWindow:
    source_id: str
    segments: tuple[DiarizedSpeechSegment, ...]
    def __post_init__(self) -> None:
        if not self.source_id: raise ValueError("source_id must be non-empty")
        object.__setattr__(self, "segments", tuple(self.segments))
    @property
    def speaker_count(self) -> int:
        return len({segment.speaker_id for segment in self.segments})


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    speaker_id: str | None
    start_time: datetime
    end_time: datetime
    text: str


@dataclass(frozen=True, slots=True)
class ImageCrop:
    object_id: str
    source_id: str
    event_time: datetime
    image: object
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    object_id: str
    source_id: str
    event_time: datetime
    vector: tuple[float, ...]
    model_id: str
    model_version: str = "1"


@dataclass(frozen=True, slots=True)
class AudioLocalization:
    source_id: str
    event_time: datetime
    bearing_degrees: float
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class VisualBearing:
    object_id: str
    source_id: str
    event_time: datetime
    bearing_degrees: float
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class InteractionEvidence:
    item_id: str
    participant_a_id: str
    participant_b_id: str
    event_time: datetime
    score: float


def track_frame(source_id: str, event_time: datetime, tracks: Iterable[Track]) -> TrackFrame:
    return TrackFrame(source_id=source_id, event_time=event_time, tracks=tuple(tracks))
