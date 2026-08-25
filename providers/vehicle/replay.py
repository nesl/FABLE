"""Retained-detection storage and replay for Phase-7 vehicle providers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from fable.common.time import EventTimeInterval

from .association import CrossSensorIdentityAssociator
from .errors import InvalidProviderInput
from .models import DescriptorSet, DetectionFrame, EntityAssociationSet, TrackSet
from .tracker import DetectionReplayTracker, RoboflowTrackerAdapter


@dataclass(frozen=True)
class HistoricalVehicleMatch:
    """Compact vehicle candidate recovered from a bounded track interval."""

    scoped_track_id: str
    source_id: str
    event_time_interval: EventTimeInterval
    confidence: float


class HistoricalVehicleIntervalMatcher:
    """Recover concrete vehicle identities from retained/replayed track sets.

    This provider intentionally performs presence recovery, not cross-camera
    identity association.  Each candidate remains scoped to its source and
    tracker session; later SAME_ENTITY processing is responsible for relating
    it to evidence from another interval or camera.
    """

    def match_many(
        self,
        track_sets: tuple[TrackSet, ...],
        *,
        entity_kind: str = "vehicle",
        bound_entity_id: str | None = None,
        maximum_candidates: int = 8,
    ) -> tuple[HistoricalVehicleMatch, ...]:
        if entity_kind != "vehicle":
            return ()
        grouped: dict[str, list] = {}
        for track_set in sorted(track_sets, key=lambda item: item.event_time):
            for track in track_set.tracks:
                if bound_entity_id and track.scoped_track_id != bound_entity_id:
                    continue
                grouped.setdefault(track.scoped_track_id, []).append(track)
        candidates = []
        for scoped_track_id, observations in grouped.items():
            first, last = observations[0], observations[-1]
            candidates.append(
                HistoricalVehicleMatch(
                    scoped_track_id=scoped_track_id,
                    source_id=last.source_id,
                    event_time_interval=EventTimeInterval(
                        start=first.event_time,
                        end=last.event_time,
                    ),
                    confidence=max(item.confidence for item in observations),
                )
            )
        candidates.sort(
            key=lambda item: (-item.confidence, item.event_time_interval.start, item.scoped_track_id)
        )
        return tuple(candidates[: max(0, maximum_candidates)])


class JsonlDetectionStore:
    """Append-only detection artifact store with event-time interval lookup."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, frame: DetectionFrame) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(frame.model_dump_json() + "\n")

    def query(self, *, source_id: str, interval: EventTimeInterval) -> tuple[DetectionFrame, ...]:
        if not self.path.exists():
            return ()
        frames: list[DetectionFrame] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            frame = DetectionFrame.model_validate_json(line)
            if frame.source_id != source_id:
                continue
            instant = EventTimeInterval(start=frame.event_time, end=frame.event_time)
            if interval.overlaps(instant):
                frames.append(frame)
        frames.sort(key=lambda frame: (frame.event_time, frame.frame_id))
        return tuple(frames)


class RetrospectiveVehicleExecutor:
    """Replay registered detector-derived artifacts using original event time."""

    def __init__(
        self,
        *,
        detection_store: JsonlDetectionStore,
        tracker_factory: Callable[[], RoboflowTrackerAdapter],
    ) -> None:
        self.detection_store = detection_store
        self.tracker_replay = DetectionReplayTracker(tracker_factory)

    def replay_tracks(
        self,
        *,
        source_id: str,
        interval: EventTimeInterval,
    ) -> tuple[TrackSet, ...]:
        frames = self.detection_store.query(source_id=source_id, interval=interval)
        if not frames:
            raise InvalidProviderInput(
                f"no retained detection_set.v1 frames for {source_id} in {interval}"
            )
        return self.tracker_replay.replay(frames)

    @staticmethod
    def replay_association(
        left: DescriptorSet,
        right: DescriptorSet,
        *,
        associator: CrossSensorIdentityAssociator | None = None,
        route_time_compatibility: dict[tuple[str, str], bool] | None = None,
    ) -> EntityAssociationSet:
        return (associator or CrossSensorIdentityAssociator()).associate(
            left,
            right,
            route_time_compatibility=route_time_compatibility,
        )


def write_detection_fixture(path: str | Path, frames: Iterable[DetectionFrame]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(frame.model_dump_json() + "\n" for frame in frames),
        encoding="utf-8",
    )
