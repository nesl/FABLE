"""Source-adapter construction with one shared replay start barrier."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from threading import Event
from typing import Callable

from fable.execution import OpenCVVideoSourceAdapter, SourceAdapter, WaveAudioSourceAdapter

from .manifest import ReplayManifest


@dataclass(frozen=True)
class SynchronizedReplay:
    replay_id: str
    adapters: dict[str, SourceAdapter]
    start_barrier: Event

    def start(self) -> None:
        self.start_barrier.set()


class _BarrierAdapter:
    def __init__(self, delegate: SourceAdapter, barrier: Event) -> None:
        self.delegate = delegate
        self.barrier = barrier
        self.source_id = delegate.source_id
        self.data_type = delegate.data_type

    def start(self, bus) -> None:
        # Waiting occurs in a tiny daemon-side launcher so NodeAgent startup is
        # not blocked while all sources are being registered.
        import threading

        threading.Thread(target=self._start_after_barrier, args=(bus,), daemon=True).start()

    def _start_after_barrier(self, bus) -> None:
        self.barrier.wait()
        self.delegate.start(bus)

    def stop(self) -> None:
        self.delegate.stop()


def build_synchronized_replay(
    manifest: ReplayManifest,
    *,
    video_factory: Callable[..., SourceAdapter] = OpenCVVideoSourceAdapter,
    audio_factory: Callable[..., SourceAdapter] = WaveAudioSourceAdapter,
) -> SynchronizedReplay:
    """Build source adapters whose activation shares one explicit barrier.

    Existing core source adapters timestamp from their local start. The
    manifest retains authoritative recording event time for evaluation joins;
    a future timestamp-aware adapter may consume it without changing this
    manifest contract.
    """

    barrier = Event()
    adapters: dict[str, SourceAdapter] = {}
    for row in manifest.sources:
        realtime = manifest.speed == 1.0
        if row.modality == "video":
            delegate = video_factory(row.source_id, str(row.path), realtime=realtime)
        else:
            delegate = audio_factory(row.source_id, row.path, realtime=realtime)
        adapters[row.source_id] = _BarrierAdapter(delegate, barrier)
    return SynchronizedReplay(manifest.replay_id, adapters, barrier)
