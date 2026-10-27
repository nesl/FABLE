"""Source-adapter construction with one shared replay start barrier."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Event
import threading
import time
from typing import Callable

from fable.execution import OpenCVVideoSourceAdapter, SourceAdapter, StreamKey, WaveAudioSourceAdapter
from fable.providers.data_models import VideoFrame

from .manifest import ReplayManifest


@dataclass(frozen=True)
class SynchronizedReplay:
    replay_id: str
    adapters: dict[str, SourceAdapter]
    start_barrier: Event

    def start(self) -> None:
        self.start_barrier.set()


class _BarrierAdapter:
    def __init__(self, delegate: SourceAdapter, barrier: Event, delay_seconds: float = 0.0) -> None:
        self.delegate = delegate
        self.barrier = barrier
        self.source_id = delegate.source_id
        self.data_type = delegate.data_type
        self.delay_seconds = max(0.0, float(delay_seconds))
        self._start_lock = threading.Lock()
        self._started = False

    def start(self, bus) -> None:
        # Waiting occurs in a tiny daemon-side launcher so NodeAgent startup is
        # not blocked while all sources are being registered.
        with self._start_lock:
            if self._started:
                return
            self._started = True
        threading.Thread(target=self._start_after_barrier, args=(bus,), daemon=True).start()

    def _start_after_barrier(self, bus) -> None:
        self.barrier.wait()
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        self.delegate.start(bus)

    def stop(self) -> None:
        self.delegate.stop()
        with self._start_lock:
            self._started = False


class _SegmentedVideoAdapter:
    """Replay consecutive files as one camera on a shared event-time clock."""
    data_type = "video_frame"

    def __init__(self, row, barrier: Event, origin: datetime, speed: float) -> None:
        self.source_id, self.row = row.source_id, row
        self.barrier, self.origin, self.speed = barrier, origin, speed
        self._stop = Event(); self._thread: threading.Thread | None = None

    def start(self, bus) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(bus,), daemon=True)
        self._thread.start()

    def _run(self, bus) -> None:
        import cv2
        self.barrier.wait(); wall_start = time.monotonic(); frame_number = 0
        for segment in self.row.segments:
            if self._stop.is_set(): break
            capture = cv2.VideoCapture(str(segment.path))
            if not capture.isOpened(): continue
            if segment.offset_seconds:
                capture.set(cv2.CAP_PROP_POS_MSEC, segment.offset_seconds * 1000.0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0); period = 1.0 / fps if fps > 0 else 0.0; index = 0
            try:
                while not self._stop.is_set():
                    media_seconds = index * period
                    if segment.duration_seconds is not None and media_seconds >= segment.duration_seconds: break
                    ok, image = capture.read()
                    if not ok: break
                    if self.row.rotation_degrees:
                        rotation = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                    270: cv2.ROTATE_90_COUNTERCLOCKWISE}[self.row.rotation_degrees]
                        image = cv2.rotate(image, rotation)
                    event_time = segment.event_time_start + timedelta(seconds=segment.offset_seconds + media_seconds)
                    due = max(0.0, (event_time - self.origin).total_seconds() / self.speed)
                    self._stop.wait(max(0.0, wall_start + due - time.monotonic()))
                    if self._stop.is_set(): break
                    bus.publish(StreamKey("video_frame", (self.source_id,)), VideoFrame(
                        self.source_id, event_time, image, str(frame_number)
                    ))
                    frame_number += 1; index += 1
            finally:
                capture.release()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2); self._thread = None


def build_synchronized_replay(
    manifest: ReplayManifest,
    *,
    video_factory: Callable[..., SourceAdapter] = OpenCVVideoSourceAdapter,
    audio_factory: Callable[..., SourceAdapter] = WaveAudioSourceAdapter,
) -> SynchronizedReplay:
    """Build source adapters whose activation shares one explicit barrier.

    Adapters use the manifest's authoritative event-time origin, recording
    offset, and playback speed. Their worker threads share one release barrier.
    """

    barrier = Event()
    adapters: dict[str, SourceAdapter] = {}
    starts = [
        row.segments[0].event_time_start + timedelta(seconds=row.segments[0].offset_seconds)
        if row.segments else row.event_time_start + timedelta(seconds=row.offset_seconds)
        for row in manifest.sources
    ]
    origin = min(starts)
    for row in manifest.sources:
        if row.modality == "video":
            if row.segments:
                adapters[row.source_id] = _SegmentedVideoAdapter(row, barrier, origin, manifest.speed)
                continue
            delegate = video_factory(
                row.source_id, str(row.path), realtime=True,
                event_time_start=row.event_time_start,
                offset_seconds=row.offset_seconds,
                playback_speed=manifest.speed,
                rotation_degrees=row.rotation_degrees,
            )
        else:
            delegate = audio_factory(
                row.source_id, row.path, realtime=True,
                event_time_start=row.event_time_start,
                offset_seconds=row.offset_seconds,
                playback_speed=manifest.speed,
            )
        effective_start = row.event_time_start + timedelta(seconds=row.offset_seconds)
        adapters[row.source_id] = _BarrierAdapter(
            delegate, barrier, (effective_start - origin).total_seconds() / manifest.speed
        )
    return SynchronizedReplay(manifest.replay_id, adapters, barrier)
