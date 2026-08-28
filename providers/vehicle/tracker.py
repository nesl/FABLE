"""Library-backed tracking and deterministic retained-detection replay."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Callable, Iterable, Protocol
from uuid import uuid4

from .errors import InvalidProviderInput, OptionalDependencyError
from .models import (
    BoundingBox,
    Detection,
    DetectionFrame,
    TrackObservation,
    TrackSet,
    scoped_track_identity,
)


class TrackerProtocol(Protocol):
    def update(self, detections: Any, frame: Any | None = None, timestamp: float | None = None) -> Any: ...
    def reset(self) -> None: ...


class RoboflowTrackerAdapter:
    """Adapter for Roboflow ``trackers`` ByteTrack or SORT.

    Provider-local IDs are never treated as global identity. Every output uses
    ``(source_id, tracker_session_id, local_track_id)``. The tracker does not
    advertise arbitrary state migration: retrospective recovery is performed by
    replaying retained ``detection_set.v1`` frames into a fresh instance.
    """

    def __init__(
        self,
        *,
        algorithm: str = "bytetrack",
        frame_rate: float = 30.0,
        tracker: TrackerProtocol | None = None,
        detections_factory: Callable[[DetectionFrame], Any] | None = None,
        tracker_version: str = "trackers-2.5",
        session_id: str | None = None,
        tracker_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.algorithm = algorithm.lower()
        self.frame_rate = float(frame_rate)
        self.tracker_version = tracker_version
        self.session_id = session_id or uuid4().hex
        self._tracker_kwargs = dict(tracker_kwargs or {})
        self._tracker = tracker
        self._detections_factory = detections_factory
        self._ages: dict[int, int] = defaultdict(int)
        # ByteTrack can continue returning a live track on frames where its
        # predicted box has no corresponding detector row.  Preserve the last
        # detector-aligned metadata for that tracker-local identity so a later
        # lifecycle/EXITS observation can still request the exact YOLO crop.
        # This is deliberately tracker-session local and is cleared on reset.
        self._last_detection_by_track: dict[int, Detection] = {}
        # Trackers may withhold a tentative identity until a later frame. Keep
        # a short detector-only history so that first publication can still be
        # tied to its originating YOLO crop. Entries never cross replay/session
        # resets and matching requires compatible class plus spatial overlap.
        self._recent_detections: deque[tuple[datetime, Detection]] = deque(
            maxlen=max(8, int(round(self.frame_rate)))
        )
        self._last_event_time: datetime | None = None
        self._last_frame_id: str | None = None
        self._last_source_sequence: int | None = None
        self._replay_id: str | None = None

    @property
    def family(self) -> str:
        return f"roboflow_{self.algorithm}"

    def _ensure_tracker(self) -> TrackerProtocol:
        if self._tracker is not None:
            return self._tracker
        try:
            from trackers import ByteTrackTracker, SORTTracker
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OptionalDependencyError(
                "Roboflow trackers is required; install trackers==2.5.0.post0 and supervision"
            ) from exc
        kwargs = {"frame_rate": self.frame_rate, **self._tracker_kwargs}
        if self.algorithm in {"bytetrack", "byte_track", "byte-track"}:
            self._tracker = ByteTrackTracker(**kwargs)
        elif self.algorithm == "sort":
            self._tracker = SORTTracker(**kwargs)
        else:
            raise ValueError(f"unsupported tracker algorithm: {self.algorithm}")
        return self._tracker

    def _to_supervision(self, frame: DetectionFrame) -> Any:
        if self._detections_factory is not None:
            return self._detections_factory(frame)
        try:
            import numpy as np
            import supervision as sv
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OptionalDependencyError(
                "Roboflow Trackers uses supervision.Detections; install the vehicle-tracking extra"
            ) from exc
        xyxy = np.asarray(
            [[det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2] for det in frame.detections],
            dtype=np.float32,
        ).reshape((-1, 4))
        confidence = np.asarray([det.confidence for det in frame.detections], dtype=np.float32)
        # Tracker class IDs describe semantic classes, not a row's transient
        # position in one detector frame.  Using ``arange`` made the same car
        # change class whenever YOLO reordered boxes, preventing association
        # on busy/mobile views.
        stable_classes = {
            "car": 0,
            "truck": 1,
            "bus": 2,
            "motorcycle": 3,
            "person": 4,
            "object": 5,
        }
        class_ids = np.asarray(
            [stable_classes.get(detection.class_name, 6) for detection in frame.detections],
            dtype=np.int32,
        )
        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids,
            data={"fable_detection_id": [det.detection_id for det in frame.detections]},
        )

    def update(self, frame: DetectionFrame, *, image: Any | None = None) -> TrackSet:
        if frame.replay_id and frame.replay_id != self._replay_id:
            # A bounded retrospective replay is a new event-time generation.
            # Reset tracker-local state before accepting its earlier timestamps
            # rather than mixing them into the live generation.
            self.reset(new_session=True)
            self._replay_id = frame.replay_id
        if self._last_event_time is not None and frame.event_time < self._last_event_time:
            raise InvalidProviderInput(
                "tracker input event time must be monotonic: "
                f"source_id={frame.source_id!r} replay_id={frame.replay_id!r} "
                f"previous_event_time={self._last_event_time.isoformat()} "
                f"current_event_time={frame.event_time.isoformat()} "
                f"previous_frame_id={self._last_frame_id!r} current_frame_id={frame.frame_id!r} "
                f"previous_source_sequence={self._last_source_sequence!r} "
                f"current_source_sequence={frame.source_sequence!r}"
            )
        self._last_event_time = frame.event_time
        self._last_frame_id = frame.frame_id
        self._last_source_sequence = frame.source_sequence
        tracker = self._ensure_tracker()
        tracked = tracker.update(
            self._to_supervision(frame),
            frame=image,
        )
        self._recent_detections.extend(
            (frame.event_time, detection) for detection in frame.detections
        )
        output = self._from_tracked(frame, tracked)
        return TrackSet(
            source_id=frame.source_id,
            tracker_family=self.family,
            tracker_version=self.tracker_version,
            tracker_session_id=self.session_id,
            event_time=frame.event_time,
            replay_id=frame.replay_id,
            tracks=tuple(output),
        )

    def reset(self, *, new_session: bool = True) -> None:
        if self._tracker is not None:
            self._tracker.reset()
        self._ages.clear()
        self._last_detection_by_track.clear()
        self._recent_detections.clear()
        self._last_event_time = None
        self._last_frame_id = None
        self._last_source_sequence = None
        if new_session:
            self._replay_id = None
        if new_session:
            self.session_id = uuid4().hex

    def _from_tracked(self, source: DetectionFrame, tracked: Any) -> list[TrackObservation]:
        xyxy = _array(getattr(tracked, "xyxy", ()), columns=4)
        tracker_ids = _vector(getattr(tracked, "tracker_id", ()), integer=True)
        confidences = _vector(getattr(tracked, "confidence", ()))
        class_ids = _vector(getattr(tracked, "class_id", ()), integer=True)
        if len(confidences) < len(xyxy):
            confidences += [1.0] * (len(xyxy) - len(confidences))
        if len(class_ids) < len(xyxy):
            class_ids += [-1] * (len(xyxy) - len(class_ids))
        rows: list[TrackObservation] = []
        for index, coords in enumerate(xyxy):
            track_id = tracker_ids[index] if index < len(tracker_ids) else -1
            if track_id < 0:
                continue
            bbox = BoundingBox(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])
            current_match = _match_input_detection(
                source.detections, bbox, class_ids[index]
            )
            matched = current_match
            if current_match is not None:
                self._last_detection_by_track[track_id] = current_match
            else:
                matched = self._last_detection_by_track.get(track_id)
                if matched is None:
                    matched = _match_recent_detection(
                        self._recent_detections,
                        bbox,
                        class_ids[index],
                        event_time=source.event_time,
                    )
                    if matched is not None:
                        self._last_detection_by_track[track_id] = matched
            self._ages[track_id] += 1
            rows.append(
                TrackObservation(
                    local_track_id=track_id,
                    scoped_track_id=scoped_track_identity(source.source_id, self.session_id, track_id),
                    source_id=source.source_id,
                    tracker_session_id=self.session_id,
                    class_name=matched.class_name if matched else "object",
                    confidence=max(0.0, min(1.0, confidences[index])),
                    bbox=bbox,
                    event_time=source.event_time,
                    world_point=matched.world_point if matched else None,
                    age_frames=self._ages[track_id],
                    attributes={
                        "matched_detection_id": matched.detection_id if matched else "",
                        "detector_matched_current_frame": current_match is not None,
                        **(
                            {"reid": matched.attributes.get("reid")}
                            if matched and matched.attributes.get("reid")
                            else {}
                        ),
                    },
                )
            )
        return rows


class DetectionReplayTracker:
    """Reconstruct tracker state by replaying retained detection frames."""

    def __init__(self, factory: Callable[[], RoboflowTrackerAdapter]) -> None:
        self.factory = factory

    def replay(self, frames: Iterable[DetectionFrame]) -> tuple[TrackSet, ...]:
        ordered = tuple(sorted(frames, key=lambda frame: (frame.event_time, frame.frame_id)))
        if not ordered:
            return ()
        sources = {frame.source_id for frame in ordered}
        if len(sources) != 1:
            raise InvalidProviderInput("one replay tracker invocation must use one source")
        tracker = self.factory()
        outputs = [tracker.update(frame) for frame in ordered]
        start = ordered[0].event_time
        end = ordered[-1].event_time
        from fable.common.time import EventTimeInterval

        interval = EventTimeInterval(start=start, end=end)
        return tuple(
            output.model_copy(
                update={
                    "reconstructed_from_detection_replay": True,
                    "replay_interval": interval,
                }
            )
            for output in outputs
        )


def _array(value: Any, *, columns: int) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    rows = list(value)
    if not rows:
        return []
    if rows and not isinstance(rows[0], (list, tuple)):
        rows = [rows]
    return [[float(item) for item in row[:columns]] for row in rows]


def _vector(value: Any, *, integer: bool = False) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    result = list(value)
    return [int(item) if integer else float(item) for item in result]


def _match_input_detection(
    detections: tuple[Detection, ...],
    bbox: BoundingBox,
    class_index: int,
) -> Detection | None:
    if not detections:
        return None
    # ``class_index`` is the detector's semantic class ID (for example every
    # car is class 0), not the row index in this frame. Treating it as a row
    # index silently attached detection 0's metadata/crop to every tracked car
    # whenever multiple vehicles were present. ByteTrack already returns the
    # tracked box, so IoU is the correct way to recover its corresponding YOLO
    # detection and detector-aligned ReID crop.
    return max(detections, key=lambda item: _iou(item.bbox, bbox))


def _match_recent_detection(
    detections: Iterable[tuple[datetime, Detection]],
    bbox: BoundingBox,
    class_index: int,
    *,
    event_time: datetime,
    maximum_age_seconds: float = 2.0,
    minimum_iou: float = 0.20,
) -> Detection | None:
    """Recover provenance for a tracker identity published after detection.

    This is not appearance inference and does not create evidence. It only
    attaches a recent typed YOLO observation of the same semantic class whose
    detector box overlaps the tracker prediction.
    """
    compatible = [
        detection
        for detected_at, detection in detections
        if 0.0 <= (event_time - detected_at).total_seconds() <= maximum_age_seconds
        and _semantic_class_id(detection.class_name) == class_index
        and _iou(detection.bbox, bbox) >= minimum_iou
    ]
    if not compatible:
        return None
    return max(compatible, key=lambda item: _iou(item.bbox, bbox))


def _semantic_class_id(class_name: str) -> int:
    return {
        "car": 0,
        "truck": 1,
        "bus": 2,
        "motorcycle": 3,
        "person": 4,
        "object": 5,
    }.get(class_name, 6)


def _iou(left: BoundingBox, right: BoundingBox) -> float:
    x1 = max(left.x1, right.x1)
    y1 = max(left.y1, right.y1)
    x2 = min(left.x2, right.x2)
    y2 = min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0
