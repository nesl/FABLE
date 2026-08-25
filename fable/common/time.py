"""Event-time intervals, deadlines, lateness, and watermark helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from pydantic import Field, field_validator, model_validator

from .base import FableModel

UTC = timezone.utc

# Catalog/scenario timestamps are often rounded or include a small capture
# tail beyond the final decodable media timestamp. Raw-buffer availability
# checks may tolerate up to five seconds of that boundary mismatch; semantic
# event comparisons remain exact by default.
RAW_BUFFER_ALIGNMENT_TOLERANCE = timedelta(seconds=5)


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("FABLE timestamps must be timezone-aware")
    return value.astimezone(UTC)


class EventTimeInterval(FableModel):
    """Closed event-time interval.

    Instantaneous observations use ``start == end``. All timestamps are
    normalized to UTC at validation time.
    """

    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _validate_order(self) -> "EventTimeInterval":
        if self.end < self.start:
            raise ValueError("event-time interval end must not precede start")
        return self

    @property
    def is_instant(self) -> bool:
        return self.start == self.end

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def overlaps(self, other: "EventTimeInterval") -> bool:
        return self.start <= other.end and other.start <= self.end

    def contains(self, timestamp: datetime) -> bool:
        timestamp = ensure_utc(timestamp)
        return self.start <= timestamp <= self.end

    def contains_interval(
        self,
        other: "EventTimeInterval",
        *,
        tolerance: timedelta = timedelta(0),
    ) -> bool:
        if tolerance < timedelta(0):
            raise ValueError("interval containment tolerance cannot be negative")
        return self.start - tolerance <= other.start and other.end <= self.end + tolerance

    def intersection(self, other: "EventTimeInterval") -> "EventTimeInterval | None":
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return None if end < start else EventTimeInterval(start=start, end=end)


class DeadlineSpec(FableModel):
    """Processing-time usefulness boundary for one demand or hypothesis."""

    latest_useful_completion: datetime
    latest_start: datetime | None = None

    @field_validator("latest_useful_completion", "latest_start")
    @classmethod
    def _normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    @model_validator(mode="after")
    def _validate_order(self) -> "DeadlineSpec":
        if self.latest_start and self.latest_start > self.latest_useful_completion:
            raise ValueError("latest_start cannot be after latest_useful_completion")
        return self

    def slack(self, now: datetime | None = None) -> timedelta:
        return self.latest_useful_completion - ensure_utc(now or utc_now())


class LatenessPolicy(FableModel):
    allowed_lateness_ms: int = Field(default=0, ge=0)

    @property
    def allowed_lateness(self) -> timedelta:
        return timedelta(milliseconds=self.allowed_lateness_ms)


class SourceWatermark(FableModel):
    source_id: str = Field(min_length=1)
    event_time: datetime
    observed_at: datetime = Field(default_factory=utc_now)
    sequence: int | None = Field(default=None, ge=0)
    operational_coverage: bool = True

    @field_validator("event_time", "observed_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class WatermarkSnapshot(FableModel):
    generated_at: datetime = Field(default_factory=utc_now)
    sources: dict[str, SourceWatermark]

    @field_validator("generated_at")
    @classmethod
    def _normalize_timestamp(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _keys_match_sources(self) -> "WatermarkSnapshot":
        mismatches = [key for key, value in self.sources.items() if key != value.source_id]
        if mismatches:
            raise ValueError(f"watermark map keys must match source_id: {mismatches}")
        return self

    def minimum_event_time(self, source_ids: Iterable[str]) -> datetime | None:
        selected = [self.sources[source_id].event_time for source_id in source_ids if source_id in self.sources]
        return min(selected) if selected else None


def interval_closed_by_watermarks(
    interval: EventTimeInterval,
    watermarks: WatermarkSnapshot,
    required_source_ids: Iterable[str],
    lateness: LatenessPolicy,
    *,
    require_operational_coverage: bool = True,
) -> bool:
    """Return whether all required sources have passed an interval's close time."""

    close_after = interval.end + lateness.allowed_lateness
    required = tuple(required_source_ids)
    if not required:
        return False
    for source_id in required:
        source = watermarks.sources.get(source_id)
        if source is None:
            return False
        if require_operational_coverage and not source.operational_coverage:
            return False
        if source.event_time < close_after:
            return False
    return True
