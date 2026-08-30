"""Sensor/replay source adapters for the live same-node dataflow runtime."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import time
from typing import Iterable, Protocol
import wave

from fable.providers.data_models import AudioWindow, VideoFrame

from .stream_bus import StreamBus, StreamKey
from .input_gate import HysteresisInputGate, InputGateConfig


class SourceAdapter(Protocol):
    source_id: str
    data_type: str
    def start(self, bus: StreamBus) -> None: ...
    def stop(self) -> None: ...


class GatedSourceAdapter:
    """Apply a NodeAgent-owned activity gate before publishing replay input."""

    def __init__(self, adapter: SourceAdapter, config: InputGateConfig) -> None:
        self.adapter = adapter
        self.source_id = adapter.source_id
        self.data_type = adapter.data_type
        self.gate = HysteresisInputGate(config)

    def start(self, bus: StreamBus) -> None:
        outer = self

        class GateBus:
            def publish(self, key: StreamKey, value: object) -> int:
                if not outer.gate.accept(value):
                    return 0
                return bus.publish(key, value)

        self.adapter.start(GateBus())  # type: ignore[arg-type]

    def stop(self) -> None:
        self.adapter.stop()


class ManualSourceAdapter:
    """Manually-pushed source used by tests, replay harnesses, and adapters."""

    def __init__(self, source_id: str, data_type: str) -> None:
        self.source_id = source_id
        self.data_type = data_type
        self._bus: StreamBus | None = None

    def start(self, bus: StreamBus) -> None:
        self._bus = bus

    def stop(self) -> None:
        self._bus = None

    def emit(self, value: object) -> None:
        if self._bus is None:
            raise RuntimeError(f"source {self.source_id!r} is not active")
        self._bus.publish(StreamKey(self.data_type, (self.source_id,)), value)


class IterableSourceAdapter:
    """Emit a finite iterable of already-constructed source values in a thread."""

    def __init__(
        self,
        source_id: str,
        data_type: str,
        values: Iterable[object],
        *,
        interval_s: float = 0.0,
    ) -> None:
        self.source_id = source_id
        self.data_type = data_type
        self.values = values
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, bus: StreamBus) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def run() -> None:
            for value in self.values:
                if self._stop.is_set():
                    break
                bus.publish(StreamKey(self.data_type, (self.source_id,)), value)
                if self.interval_s > 0:
                    self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


class OpenCVVideoSourceAdapter:
    """Read a camera/RTSP/video-file URI and publish ``VideoFrame`` values."""

    data_type = "video_frame"

    def __init__(
        self,
        source_id: str,
        uri: str | int,
        *,
        realtime: bool = True,
    ) -> None:
        self.source_id = source_id
        self.uri = uri
        self.realtime = realtime
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, bus: StreamBus) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional vision stack
            raise RuntimeError("OpenCV is required for video source adapters") from exc
        self._stop.clear()

        def run() -> None:
            capture = cv2.VideoCapture(self.uri)
            if not capture.isOpened():
                return
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_period = 1.0 / fps if fps > 0 else 0.0
            index = 0
            start = datetime.now(timezone.utc)
            try:
                while not self._stop.is_set():
                    ok, image = capture.read()
                    if not ok:
                        break
                    event_time = (
                        start + timedelta(seconds=index * frame_period)
                        if frame_period > 0
                        else datetime.now(timezone.utc)
                    )
                    bus.publish(
                        StreamKey(self.data_type, (self.source_id,)),
                        VideoFrame(self.source_id, event_time, image, str(index)),
                    )
                    index += 1
                    if self.realtime and frame_period > 0:
                        self._stop.wait(frame_period)
            finally:
                capture.release()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


class WaveAudioSourceAdapter:
    """Replay PCM WAV audio as fixed-size ``AudioWindow`` values."""

    data_type = "audio_window"

    def __init__(
        self,
        source_id: str,
        path: str | Path,
        *,
        window_ms: int = 1000,
        realtime: bool = True,
    ) -> None:
        self.source_id = source_id
        self.path = Path(path)
        self.window_ms = int(window_ms)
        self.realtime = realtime
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, bus: StreamBus) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()

        def run() -> None:
            import struct
            with wave.open(str(self.path), "rb") as handle:
                channels = handle.getnchannels()
                sample_width = handle.getsampwidth()
                rate = handle.getframerate()
                if channels != 1 or sample_width != 2:
                    raise RuntimeError("minimal WAV adapter expects mono 16-bit PCM")
                count = max(1, int(rate * self.window_ms / 1000.0))
                start = datetime.now(timezone.utc)
                index = 0
                while not self._stop.is_set():
                    raw = handle.readframes(count)
                    if not raw:
                        break
                    ints = struct.unpack("<" + "h" * (len(raw) // 2), raw)
                    samples = tuple(value / 32768.0 for value in ints)
                    event_time = start + timedelta(seconds=index * count / rate)
                    bus.publish(
                        StreamKey(self.data_type, (self.source_id,)),
                        AudioWindow(self.source_id, event_time, samples, rate),
                    )
                    index += 1
                    if self.realtime:
                        self._stop.wait(count / rate)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
