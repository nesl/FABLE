"""Gunshot/alarm audio classification with a pluggable real-model backend.

The orchestration contract distinguishes a raw ``audio_segment.v1`` from the
thresholded/debounced ``audio_event_observation.v1``.  ``YamNetBackend`` is a
lazy adapter for a local or TensorFlow-Hub YAMNet model.  Unit tests use the
small deterministic backend so the core package remains runnable without a
large ML stack or network access.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from math import sqrt
from typing import Any, Protocol

from fable.common.time import EventTimeInterval

from .errors import InvalidAudioInput, OptionalDependencyError
from .models import AudioEventObservation, AudioWindow, phase8_occurrence_id


class AudioEventBackend(Protocol):
    backend_id: str
    backend_version: str

    def score(self, window: AudioWindow) -> Mapping[str, float]: ...


DEFAULT_LABEL_ALIASES: dict[str, tuple[str, ...]] = {
    "gunshot": (
        "Gunshot, gunfire",
        "Machine gun",
        "Fusillade",
        "Cap gun",
        "gunshot",
    ),
    "alarm": (
        "Alarm",
        "Fire alarm",
        "Smoke detector, smoke alarm",
        "Buzzer",
        "Siren",
        "alarm",
    ),
}


class DeterministicAudioEventBackend:
    """Small backend for fixtures, replay smoke tests, and policy tests.

    ``scores`` may be a fixed mapping or a callback over the typed audio window.
    This backend is explicit test/reference infrastructure, not a trained audio
    classifier.
    """

    backend_id = "deterministic_audio_backend"
    backend_version = "1"

    def __init__(
        self,
        scores: Mapping[str, float] | Callable[[AudioWindow], Mapping[str, float]],
    ) -> None:
        self._scores = scores

    def score(self, window: AudioWindow) -> Mapping[str, float]:
        values = self._scores(window) if callable(self._scores) else self._scores
        return {str(label): max(0.0, min(1.0, float(score))) for label, score in values.items()}


class SpectralRuleAudioBackend:
    """Dependency-light signal baseline for integration testing.

    The baseline measures crest factor, broadband high-frequency energy, and
    tonal persistence.  It is useful when validating a deployment without a
    model checkpoint, but it should not be reported as the final gunshot/alarm
    classifier in an evaluation.
    """

    backend_id = "spectral_rule_audio_backend"
    backend_version = "1"

    def score(self, window: AudioWindow) -> Mapping[str, float]:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OptionalDependencyError(
                "SpectralRuleAudioBackend requires NumPy; install the multimodal-core extra"
            ) from exc
        waveform = np.asarray(window.waveform, dtype=np.float32)
        mono = waveform.mean(axis=0)
        if mono.size < 8:
            raise InvalidAudioInput("audio window is too short for spectral scoring")
        rms = float(np.sqrt(np.mean(mono * mono)) + 1e-12)
        peak = float(np.max(np.abs(mono)) + 1e-12)
        crest = peak / rms
        spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
        total = float(np.sum(spectrum) + 1e-12)
        freqs = np.fft.rfftfreq(mono.size, d=1.0 / window.sample_rate_hz)
        high = float(np.sum(spectrum[freqs >= 2000.0]) / total)
        tonal = float(np.max(spectrum) / total)
        gunshot = _sigmoid((crest - 5.0) * 0.8 + (high - 0.25) * 4.0)
        alarm = _sigmoid((tonal - 0.08) * 18.0 - (crest - 3.0) * 0.15)
        return {"gunshot": gunshot, "alarm": alarm}


class YamNetBackend:
    """Lazy YAMNet adapter.

    The adapter accepts an already-loaded callable model for tests or a local
    TensorFlow-Hub handle.  It never downloads a checkpoint during module
    import.  ``class_names`` must align with the model score dimension; callers
    can load the official YAMNet class map that accompanies their checkpoint.
    """

    backend_id = "yamnet"

    def __init__(
        self,
        *,
        model: Any | None = None,
        model_handle: str | None = None,
        class_names: Sequence[str] = (),
        model_version: str = "unresolved",
    ) -> None:
        self._model = model
        self.model_handle = model_handle
        self.class_names = tuple(str(item) for item in class_names)
        self.backend_version = model_version

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_handle:
            raise InvalidAudioInput("YamNetBackend requires a loaded model or model_handle")
        try:
            import tensorflow_hub as hub
        except ImportError as exc:  # pragma: no cover - optional heavy dependency
            raise OptionalDependencyError(
                "YAMNet requires tensorflow and tensorflow-hub; install the multimodal-models extra"
            ) from exc
        self._model = hub.load(self.model_handle)
        return self._model

    def score(self, window: AudioWindow) -> Mapping[str, float]:
        if window.sample_rate_hz != 16_000:
            raise InvalidAudioInput(
                "YAMNet expects 16 kHz mono audio; resample before invoking the backend"
            )
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("YAMNet adapter requires NumPy") from exc
        mono = np.asarray(window.waveform, dtype=np.float32).mean(axis=0)
        model = self._load()
        output = model(mono)
        scores = output[0] if isinstance(output, (tuple, list)) else output
        if hasattr(scores, "numpy"):
            scores = scores.numpy()
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim == 2:
            scores = scores.mean(axis=0)
        if scores.ndim != 1:
            raise InvalidAudioInput("YAMNet returned an unsupported score tensor")
        if not self.class_names:
            return {f"class_{index}": float(value) for index, value in enumerate(scores)}
        if len(self.class_names) != len(scores):
            raise InvalidAudioInput("YAMNet class map length does not match score dimension")
        return {
            label: max(0.0, min(1.0, float(value)))
            for label, value in zip(self.class_names, scores)
        }


@dataclass(frozen=True)
class AudioEventThreshold:
    label: str
    minimum_score: float
    minimum_consecutive_windows: int = 1
    refractory_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("audio event label cannot be empty")
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError("audio event threshold must be in [0, 1]")
        if self.minimum_consecutive_windows < 1:
            raise ValueError("minimum_consecutive_windows must be positive")
        if self.refractory_seconds < 0:
            raise ValueError("refractory_seconds cannot be negative")


class AudioEventClassifier:
    """Map model classes to semantic labels and debounce event intervals."""

    def __init__(
        self,
        backend: AudioEventBackend,
        *,
        thresholds: Sequence[AudioEventThreshold] = (
            AudioEventThreshold("gunshot", 0.35, 1, 0.5),
            AudioEventThreshold("alarm", 0.30, 2, 1.0),
        ),
        label_aliases: Mapping[str, Sequence[str]] = DEFAULT_LABEL_ALIASES,
        provider_id: str = "audio_event_classifier",
        provider_version: str = "1",
    ) -> None:
        self.backend = backend
        self.thresholds = {item.label: item for item in thresholds}
        self.label_aliases = {
            str(label): tuple(str(alias) for alias in aliases)
            for label, aliases in label_aliases.items()
        }
        self.provider_id = provider_id
        self.provider_version = provider_version
        self._consecutive: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._run_start: dict[tuple[str, str], EventTimeInterval] = {}
        self._last_emitted_end: dict[tuple[str, str], object] = {}

    def classify(self, window: AudioWindow) -> tuple[AudioEventObservation, ...]:
        raw_scores = {
            str(label): max(0.0, min(1.0, float(score)))
            for label, score in self.backend.score(window).items()
        }
        results: list[AudioEventObservation] = []
        for semantic_label, threshold in sorted(self.thresholds.items()):
            aliases = self.label_aliases.get(semantic_label, (semantic_label,))
            contributing = {
                label: raw_scores[label] for label in aliases if label in raw_scores
            }
            score = max(contributing.values(), default=raw_scores.get(semantic_label, 0.0))
            key = (window.source_id, semantic_label)
            if score < threshold.minimum_score:
                self._consecutive[key] = 0
                self._run_start.pop(key, None)
                continue
            if self._consecutive[key] == 0:
                self._run_start[key] = window.event_time_interval
            self._consecutive[key] += 1
            if self._consecutive[key] < threshold.minimum_consecutive_windows:
                continue
            last_end = self._last_emitted_end.get(key)
            if last_end is not None and window.event_time_interval.start < (
                last_end + timedelta(seconds=threshold.refractory_seconds)  # type: ignore[operator]
            ):
                continue
            start = self._run_start[key].start
            interval = EventTimeInterval(start=start, end=window.event_time_interval.end)
            observation = AudioEventObservation(
                occurrence_id=phase8_occurrence_id(
                    "AUDIO_EVENT",
                    {"location": window.source_id, "event": semantic_label},
                    interval,
                    self.provider_id,
                ),
                label=semantic_label,
                confidence=score,
                event_time_interval=interval,
                source_id=window.source_id,
                provider_id=self.provider_id,
                provider_version=self.provider_version,
                source_labels=tuple(sorted(contributing)),
                class_scores=contributing or {semantic_label: score},
                attributes={
                    "backend_id": self.backend.backend_id,
                    "backend_version": self.backend.backend_version,
                },
            )
            results.append(observation)
            self._last_emitted_end[key] = interval.end
            # A new event after the refractory period must establish a new run.
            self._consecutive[key] = 0
            self._run_start.pop(key, None)
        return tuple(results)


def audio_window_from_replay_payload(
    document: Mapping[str, Any],
    *,
    source_id: str,
    sample_rate_hz: int = 16_000,
    channel_indices: Sequence[int] | None = None,
    frame_duration_seconds: float | None = None,
    source_sequence: int | None = None,
) -> AudioWindow:
    """Convert the ReSpeaker replay payload to normalized channels-first audio.

    The existing replay service passes NumPy arrays through local IPC.  Tests
    and network bridges may supply nested Python lists instead.
    """

    payload = document.get("payload", document)
    if not isinstance(payload, Mapping):
        raise InvalidAudioInput("replay audio payload must be a mapping")
    waveform = payload.get("waveform")
    timestamp = payload.get("t")
    if waveform is None or timestamp is None:
        raise InvalidAudioInput("replay audio payload requires waveform and t")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError(
            "audio replay conversion requires NumPy; install the multimodal-core extra"
        ) from exc
    values = np.asarray(waveform)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise InvalidAudioInput("replay waveform must be samples x channels")
    indices = tuple(channel_indices) if channel_indices is not None else tuple(range(values.shape[1]))
    if not indices or min(indices) < 0 or max(indices) >= values.shape[1]:
        raise InvalidAudioInput("audio channel selection is outside the waveform")
    selected = values[:, indices].astype(np.float32)
    max_abs = float(np.max(np.abs(selected))) if selected.size else 0.0
    if max_abs > 1.5:
        selected = selected / 32768.0
    from datetime import datetime, timezone

    if isinstance(timestamp, datetime):
        event_start = timestamp
    elif isinstance(timestamp, (float, int)):
        event_start = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    elif isinstance(timestamp, str):
        event_start = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    else:
        raise InvalidAudioInput(f"unsupported audio timestamp {timestamp!r}")
    duration = frame_duration_seconds or (selected.shape[0] / float(sample_rate_hz))
    interval = EventTimeInterval(
        start=event_start,
        end=event_start + timedelta(seconds=duration),
    )
    return AudioWindow(
        source_id=source_id,
        event_time_interval=interval,
        sample_rate_hz=sample_rate_hz,
        channel_ids=tuple(f"ch{index}" for index in indices),
        waveform=tuple(tuple(float(value) for value in selected[:, index]) for index in range(selected.shape[1])),
        source_sequence=source_sequence,
    )


def _sigmoid(value: float) -> float:
    import math

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)
