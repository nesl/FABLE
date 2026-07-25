"""Package detection adaptation and transfer/custody reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import hypot
from typing import Iterable

from fable.common.time import EventTimeInterval
from providers.vehicle.models import DetectionFrame, TrackObservation, TrackSet

from .errors import InteractionStateError
from .models import (
    CustodyRecord,
    CustodyState,
    InteractionPredicateObservation,
    phase8_occurrence_id,
)


DEFAULT_PACKAGE_CLASSES = frozenset(
    {
        "backpack",
        "handbag",
        "suitcase",
        "package",
        "parcel",
        "box",
        "bag",
    }
)
HOLDER_CLASSES = frozenset({"person", "car", "truck", "bus", "motorcycle", "vehicle"})


class PackageDetectionAdapter:
    """Select package-like detections from a full-context detector output."""

    def __init__(
        self,
        *,
        class_allowlist: Iterable[str] = DEFAULT_PACKAGE_CLASSES,
        confidence_threshold: float = 0.25,
        provider_id: str = "package_detector",
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("package confidence threshold must be in [0, 1]")
        self.class_allowlist = frozenset(item.lower() for item in class_allowlist)
        self.confidence_threshold = confidence_threshold
        self.provider_id = provider_id

    def filter(self, detections: DetectionFrame) -> DetectionFrame:
        selected = tuple(
            item
            for item in detections.detections
            if item.class_name.lower() in self.class_allowlist
            and item.confidence >= self.confidence_threshold
        )
        return detections.model_copy(
            update={
                "detector_id": self.provider_id,
                "detections": selected,
            }
        )


@dataclass
class _PendingHolder:
    holder_id: str
    holder_type: str
    since: datetime
    distance_m: float


class TransferCustodyReasoner:
    """Infer stable custody and emit transfer only after a confirmed owner change.

    The compact ``custody_state.v1`` artifact is suitable for continuation after
    high-resolution interaction analysis stops.  Raw frames/crops are not
    required once a transfer has been resolved unless a later graph branch asks
    for reinterpretation.
    """

    def __init__(
        self,
        *,
        maximum_holder_distance_m: float = 1.5,
        minimum_stable_seconds: float = 1.0,
        missing_timeout_seconds: float = 5.0,
        provider_id: str = "object_transfer_reasoner",
        provider_version: str = "1",
    ) -> None:
        if maximum_holder_distance_m <= 0:
            raise ValueError("maximum holder distance must be positive")
        if minimum_stable_seconds < 0 or missing_timeout_seconds < 0:
            raise ValueError("custody timing thresholds cannot be negative")
        self.maximum_holder_distance_m = maximum_holder_distance_m
        self.minimum_stable_seconds = minimum_stable_seconds
        self.missing_timeout_seconds = missing_timeout_seconds
        self.provider_id = provider_id
        self.provider_version = provider_version
        self._records: dict[str, CustodyRecord] = {}
        self._pending: dict[str, _PendingHolder] = {}
        self._last_event_time: datetime | None = None
        self._last_source_id: str | None = None

    def update(
        self,
        tracks: TrackSet,
    ) -> tuple[CustodyState, tuple[InteractionPredicateObservation, ...]]:
        now = tracks.event_time
        if self._last_event_time is not None and now < self._last_event_time:
            raise InteractionStateError("custody input must be ordered by event time")
        packages = sorted(
            (
                item
                for item in tracks.tracks
                if item.class_name.lower() in DEFAULT_PACKAGE_CLASSES
            ),
            key=lambda item: item.scoped_track_id,
        )
        holders = sorted(
            (
                item
                for item in tracks.tracks
                if item.class_name.lower() in HOLDER_CLASSES
            ),
            key=lambda item: item.scoped_track_id,
        )
        emitted: list[InteractionPredicateObservation] = []
        visible_packages = {item.scoped_track_id for item in packages}
        for package in packages:
            nearest = _nearest(package, holders)
            if nearest is None or nearest[1] > self.maximum_holder_distance_m:
                self._pending.pop(package.scoped_track_id, None)
                existing = self._records.get(package.scoped_track_id)
                if existing is not None:
                    self._records[package.scoped_track_id] = existing.model_copy(
                        update={"last_seen_at": now}
                    )
                continue
            holder, distance = nearest
            holder_type = _holder_type(holder)
            pending = self._pending.get(package.scoped_track_id)
            if pending is None or pending.holder_id != holder.scoped_track_id:
                self._pending[package.scoped_track_id] = _PendingHolder(
                    holder_id=holder.scoped_track_id,
                    holder_type=holder_type,
                    since=now,
                    distance_m=distance,
                )
                continue
            if (now - pending.since).total_seconds() < self.minimum_stable_seconds:
                continue
            existing = self._records.get(package.scoped_track_id)
            if existing is None:
                self._records[package.scoped_track_id] = CustodyRecord(
                    package_id=package.scoped_track_id,
                    holder_id=pending.holder_id,
                    holder_type=pending.holder_type,
                    confidence=_custody_confidence(distance, self.maximum_holder_distance_m),
                    established_at=pending.since,
                    last_seen_at=now,
                )
                continue
            if existing.holder_id == pending.holder_id:
                self._records[package.scoped_track_id] = existing.model_copy(
                    update={
                        "confidence": _custody_confidence(
                            distance, self.maximum_holder_distance_m
                        ),
                        "last_seen_at": now,
                    }
                )
                continue
            interval = EventTimeInterval(start=existing.established_at, end=now)
            bindings = {
                "object": package.scoped_track_id,
                "source": existing.holder_id or "unbound",
                "destination": pending.holder_id,
            }
            occurrence_id = phase8_occurrence_id(
                "TRANSFER",
                bindings,
                interval,
                self.provider_id,
            )
            confidence = min(
                existing.confidence,
                _custody_confidence(distance, self.maximum_holder_distance_m),
            )
            emitted.append(
                InteractionPredicateObservation(
                    occurrence_id=occurrence_id,
                    predicate_id="TRANSFER",
                    truth=True,
                    confidence=confidence,
                    event_time_interval=interval,
                    bindings=bindings,
                    source_ids=(tracks.source_id,),
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                    supporting_artifact_types=(
                        "package_track_set.v1",
                        "custody_state.v1",
                    ),
                    measurements={
                        "source_holder_type": existing.holder_type or "unknown",
                        "destination_holder_type": pending.holder_type,
                        "destination_distance_m": distance,
                    },
                )
            )
            self._records[package.scoped_track_id] = CustodyRecord(
                package_id=package.scoped_track_id,
                holder_id=pending.holder_id,
                holder_type=pending.holder_type,
                confidence=confidence,
                established_at=pending.since,
                last_seen_at=now,
                previous_holder_id=existing.holder_id,
                transfer_occurrence_id=occurrence_id,
            )

        # Expire only custody state whose package has been absent beyond the
        # bounded continuation horizon. A brief detector miss does not erase it.
        for package_id, record in tuple(self._records.items()):
            if package_id in visible_packages:
                continue
            if (now - record.last_seen_at).total_seconds() > self.missing_timeout_seconds:
                self._records.pop(package_id, None)
                self._pending.pop(package_id, None)

        interval_start = min(
            (record.established_at for record in self._records.values()),
            default=now,
        )
        state = CustodyState(
            source_id=tracks.source_id,
            event_time_interval=EventTimeInterval(start=interval_start, end=now),
            records=tuple(self._records[key] for key in sorted(self._records)),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )
        self._last_event_time = now
        self._last_source_id = tracks.source_id
        return state, tuple(emitted)


def _holder_type(track: TrackObservation) -> str:
    return "person" if track.class_name.lower() == "person" else "vehicle"


def _point(track: TrackObservation) -> tuple[float, float]:
    if track.world_point is not None:
        return track.world_point.x, track.world_point.y
    return track.bbox.center


def _distance(left: TrackObservation, right: TrackObservation) -> float:
    if left.world_point is not None and right.world_point is not None:
        if left.world_point.coordinate_frame_id != right.world_point.coordinate_frame_id:
            raise InteractionStateError("custody tracks use incompatible coordinate frames")
    left_point = _point(left)
    right_point = _point(right)
    return hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])


def _nearest(
    package: TrackObservation,
    holders: list[TrackObservation],
) -> tuple[TrackObservation, float] | None:
    if not holders:
        return None
    ranked = sorted(
        ((_distance(package, holder), holder) for holder in holders),
        key=lambda item: (item[0], item[1].scoped_track_id),
    )
    distance, holder = ranked[0]
    return holder, distance


def _custody_confidence(distance: float, maximum: float) -> float:
    return max(0.0, min(1.0, 0.55 + 0.45 * (1.0 - distance / maximum)))
