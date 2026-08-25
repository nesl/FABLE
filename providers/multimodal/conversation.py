"""Voice activity, lightweight online diarization, and conversation reasoning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, hypot, sqrt
from typing import Any, Protocol

from fable.common.ids import deterministic_id
from fable.common.time import EventTimeInterval
from providers.vehicle.models import TrackObservation, TrackSet

from .errors import InvalidAudioInput, OptionalDependencyError
from .models import (
    AudioWindow,
    DiarizationTurn,
    InteractionPredicateObservation,
    SpeakerTurnSet,
    SpeechSegment,
    phase8_occurrence_id,
)


class VoiceActivityDetector(Protocol):
    provider_id: str
    provider_version: str

    def detect(self, window: AudioWindow) -> tuple[SpeechSegment, ...]: ...


class AsrProvider(Protocol):
    provider_id: str
    provider_version: str

    def transcribe(self, turns: SpeakerTurnSet) -> str: ...


class EnergyVoiceActivityDetector:
    """Small VAD baseline used by deterministic tests and deployment smoke tests."""

    provider_id = "energy_vad"
    provider_version = "1"

    def __init__(self, *, rms_threshold: float = 0.015, slope: float = 90.0) -> None:
        if rms_threshold < 0 or slope <= 0:
            raise ValueError("VAD thresholds must be non-negative and slope positive")
        self.rms_threshold = rms_threshold
        self.slope = slope

    def detect(self, window: AudioWindow) -> tuple[SpeechSegment, ...]:
        samples = [value for channel in window.waveform for value in channel]
        rms = sqrt(sum(value * value for value in samples) / max(len(samples), 1))
        probability = 1.0 / (1.0 + exp(-(rms - self.rms_threshold) * self.slope))
        if probability < 0.5:
            return ()
        segment_id = deterministic_id(
            "speech_segment",
            {
                "source_id": window.source_id,
                "interval": window.event_time_interval,
                "provider": self.provider_id,
            },
            length=32,
        )
        return (
            SpeechSegment(
                segment_id=segment_id,
                source_id=window.source_id,
                event_time_interval=window.event_time_interval,
                speech_probability=probability,
            ),
        )


class WebRtcVoiceActivityDetector:
    """Lazy adapter for the WebRTC VAD package.

    Audio must be mono 16-bit PCM at a supported sample rate. The adapter splits
    a bounded audio window into 10/20/30 ms frames and merges consecutive speech
    frames into typed event-time segments.
    """

    provider_id = "webrtc_vad"
    provider_version = "1"

    def __init__(self, *, aggressiveness: int = 2, frame_duration_ms: int = 20) -> None:
        if aggressiveness not in (0, 1, 2, 3):
            raise ValueError("WebRTC VAD aggressiveness must be 0-3")
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD frames must be 10, 20, or 30 ms")
        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self._vad: Any | None = None

    def _load(self) -> Any:
        if self._vad is not None:
            return self._vad
        try:
            import webrtcvad
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OptionalDependencyError(
                "WebRTC VAD requires the webrtcvad-wheels package; install the multimodal-audio extra"
            ) from exc
        self._vad = webrtcvad.Vad(self.aggressiveness)
        return self._vad

    def detect(self, window: AudioWindow) -> tuple[SpeechSegment, ...]:
        if window.sample_rate_hz not in (8_000, 16_000, 32_000, 48_000):
            raise InvalidAudioInput("WebRTC VAD received an unsupported sample rate")
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("WebRTC VAD adapter requires NumPy") from exc
        mono = np.asarray(window.waveform, dtype=np.float32).mean(axis=0)
        pcm = np.clip(mono, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        frame_samples = int(window.sample_rate_hz * self.frame_duration_ms / 1000)
        if frame_samples <= 0:
            raise InvalidAudioInput("invalid WebRTC frame size")
        vad = self._load()
        from datetime import timedelta

        active_start: int | None = None
        active_probabilities: list[float] = []
        segments: list[SpeechSegment] = []
        frame_count = len(pcm) // frame_samples
        for index in range(frame_count):
            frame = pcm[index * frame_samples : (index + 1) * frame_samples]
            speech = bool(
                vad.is_speech(frame.tobytes(), window.sample_rate_hz)
            )
            if speech and active_start is None:
                active_start = index
            if speech:
                active_probabilities.append(1.0)
            is_last = index == frame_count - 1
            if active_start is not None and ((not speech) or is_last):
                end_index = index + 1 if speech and is_last else index
                start = window.event_time_interval.start + timedelta(
                    milliseconds=active_start * self.frame_duration_ms
                )
                end = window.event_time_interval.start + timedelta(
                    milliseconds=end_index * self.frame_duration_ms
                )
                interval = EventTimeInterval(start=start, end=end)
                segment_id = deterministic_id(
                    "speech_segment",
                    {
                        "source_id": window.source_id,
                        "interval": interval,
                        "provider": self.provider_id,
                    },
                    length=32,
                )
                segments.append(
                    SpeechSegment(
                        segment_id=segment_id,
                        source_id=window.source_id,
                        event_time_interval=interval,
                        speech_probability=sum(active_probabilities)
                        / max(len(active_probabilities), 1),
                    )
                )
                active_start = None
                active_probabilities = []
        return tuple(segments)


@dataclass
class _SpeakerCentroid:
    speaker_id: str
    centroid: list[float]
    count: int = 1


class OnlineSpeakerDiarizer:
    """Deterministic cosine clustering over provider-supplied speaker embeddings.

    The class does not pretend to extract speaker embeddings. A compatible
    upstream provider must attach embeddings to ``SpeechSegment`` records. This
    separation keeps the embedding model/version an explicit artifact contract.
    """

    def __init__(
        self,
        *,
        maximum_cosine_distance: float = 0.25,
        model_id: str = "online_cosine_diarizer",
        model_version: str = "1",
    ) -> None:
        if not 0.0 <= maximum_cosine_distance <= 2.0:
            raise ValueError("cosine distance threshold must be in [0, 2]")
        self.maximum_cosine_distance = maximum_cosine_distance
        self.model_id = model_id
        self.model_version = model_version
        self._centroids: list[_SpeakerCentroid] = []

    def diarize(self, segments: Sequence[SpeechSegment]) -> SpeakerTurnSet:
        if not segments:
            raise ValueError("diarization requires at least one speech segment")
        ordered = tuple(sorted(segments, key=lambda item: item.event_time_interval.start))
        source_ids = {item.source_id for item in ordered}
        if len(source_ids) != 1:
            raise ValueError("one diarization batch must belong to one audio source")
        turns: list[DiarizationTurn] = []
        for segment in ordered:
            speaker_id = self._assign(segment.embedding)
            turn_id = deterministic_id(
                "diarization_turn",
                {"segment_id": segment.segment_id, "speaker_id": speaker_id},
                length=32,
            )
            turns.append(
                DiarizationTurn(
                    turn_id=turn_id,
                    speaker_id=speaker_id,
                    event_time_interval=segment.event_time_interval,
                    speech_probability=segment.speech_probability,
                    transcript=segment.transcript,
                    source_segment_ids=(segment.segment_id,),
                )
            )
        interval = EventTimeInterval(
            start=min(item.event_time_interval.start for item in turns),
            end=max(item.event_time_interval.end for item in turns),
        )
        return SpeakerTurnSet(
            source_id=ordered[0].source_id,
            event_time_interval=interval,
            turns=tuple(turns),
            speaker_count=len({item.speaker_id for item in turns}),
            diarization_model_id=self.model_id,
            diarization_model_version=self.model_version,
        )

    def _assign(self, embedding: tuple[float, ...]) -> str:
        if not embedding:
            # No embedding means the provider can establish speech activity but
            # not distinct speakers. Keep one explicit unknown speaker.
            return "speaker_unknown"
        norm = sqrt(sum(value * value for value in embedding))
        if norm <= 1e-12:
            return "speaker_unknown"
        vector = [value / norm for value in embedding]
        if not self._centroids:
            return self._create(vector)
        ranked = sorted(
            (
                (1.0 - _dot(vector, item.centroid), item.speaker_id, item)
                for item in self._centroids
            ),
            key=lambda item: (item[0], item[1]),
        )
        distance, _, centroid = ranked[0]
        if distance > self.maximum_cosine_distance:
            return self._create(vector)
        count = centroid.count + 1
        updated = [
            (old * centroid.count + new) / count
            for old, new in zip(centroid.centroid, vector)
        ]
        updated_norm = sqrt(sum(value * value for value in updated)) or 1.0
        centroid.centroid = [value / updated_norm for value in updated]
        centroid.count = count
        return centroid.speaker_id

    def _create(self, vector: list[float]) -> str:
        speaker_id = f"speaker_{len(self._centroids) + 1}"
        self._centroids.append(_SpeakerCentroid(speaker_id=speaker_id, centroid=vector))
        return speaker_id


class PersonProximityEvaluator:
    """Emit sustained pair proximity from world or image-space person tracks.

    World-coordinate gaps are measured directly.  Uncalibrated image-space
    gaps are normalized by the mean person-box width, making the threshold
    portable across replay resolutions without claiming metric geometry.
    """

    provider_id = "person_proximity_provider"
    provider_version = "1"

    def __init__(
        self,
        *,
        maximum_normalized_gap: float = 2.5,
        minimum_duration_seconds: float = 1.0,
    ) -> None:
        if maximum_normalized_gap <= 0 or minimum_duration_seconds < 0:
            raise ValueError("person-proximity thresholds are invalid")
        self.maximum_normalized_gap = maximum_normalized_gap
        self.minimum_duration_seconds = minimum_duration_seconds
        self._rows: dict[tuple[str, str], list[tuple[object, float, str]]] = {}

    def update(self, track_set: TrackSet) -> tuple[InteractionPredicateObservation, ...]:
        persons = sorted(
            (item for item in track_set.tracks if item.class_name.lower() == "person"),
            key=lambda item: item.scoped_track_id,
        )
        outputs: list[InteractionPredicateObservation] = []
        for index, left in enumerate(persons):
            for right in persons[index + 1 :]:
                if left.world_point is not None and right.world_point is not None:
                    if left.world_point.coordinate_frame_id != right.world_point.coordinate_frame_id:
                        continue
                    gap = _distance(left, right)
                    mode = "world_distance"
                else:
                    scale = max((left.bbox.width + right.bbox.width) / 2.0, 1.0)
                    gap = _distance(left, right) / scale
                    mode = "bbox_width_normalized"
                if gap > self.maximum_normalized_gap:
                    continue
                pair = (left.scoped_track_id, right.scoped_track_id)
                rows = self._rows.setdefault(pair, [])
                rows.append((track_set.event_time, gap, mode))
                start = rows[0][0]
                duration = (track_set.event_time - start).total_seconds()
                if duration < self.minimum_duration_seconds:
                    continue
                interval = EventTimeInterval(start=start, end=track_set.event_time)
                mean_gap = sum(row[1] for row in rows) / len(rows)
                bindings = {"participant_a": pair[0], "participant_b": pair[1]}
                outputs.append(
                    InteractionPredicateObservation(
                        occurrence_id=phase8_occurrence_id(
                            "PERSON_PROXIMITY", bindings, interval, self.provider_id
                        ),
                        predicate_id="PERSON_PROXIMITY",
                        truth=True,
                        confidence=max(
                            0.0,
                            min(1.0, 1.0 - 0.5 * mean_gap / self.maximum_normalized_gap),
                        ),
                        event_time_interval=interval,
                        bindings=bindings,
                        source_ids=(track_set.source_id,),
                        provider_id=self.provider_id,
                        provider_version=self.provider_version,
                        supporting_artifact_types=("track_set.v1",),
                        measurements={
                            "mean_gap": mean_gap,
                            "gap_mode": mode,
                            "duration_seconds": duration,
                        },
                    )
                )
        return tuple(outputs)


class ConversationEvaluator:
    """Combine person proximity with VAD/diarization evidence.

    ASR is invoked only when ``required_terms`` is non-empty and existing turns
    lack sufficient transcripts.  Ordinary conversation detection therefore
    does not disclose or compute speech content.
    """

    def __init__(
        self,
        *,
        maximum_distance_m: float = 2.5,
        minimum_duration_seconds: float = 1.0,
        minimum_speakers: int = 2,
        provider_id: str = "conversation_provider",
        provider_version: str = "1",
    ) -> None:
        if maximum_distance_m <= 0 or minimum_duration_seconds < 0:
            raise ValueError("conversation geometry thresholds are invalid")
        if minimum_speakers < 1:
            raise ValueError("minimum_speakers must be positive")
        self.maximum_distance_m = maximum_distance_m
        self.minimum_duration_seconds = minimum_duration_seconds
        self.minimum_speakers = minimum_speakers
        self.provider_id = provider_id
        self.provider_version = provider_version

    def evaluate(
        self,
        track_sets: Sequence[TrackSet],
        turns: SpeakerTurnSet,
        *,
        required_terms: Sequence[str] = (),
        asr_provider: AsrProvider | None = None,
    ) -> InteractionPredicateObservation | None:
        if turns.speaker_count < self.minimum_speakers:
            return None
        proximity = self._proximity_interval(track_sets)
        if proximity is None:
            return None
        participant_a, participant_b, proximity_interval, mean_distance = proximity
        overlap = proximity_interval.intersection(turns.event_time_interval)
        if overlap is None or overlap.duration.total_seconds() < self.minimum_duration_seconds:
            return None
        terms = tuple(term.lower().strip() for term in required_terms if term.strip())
        transcript_used = False
        transcript = " ".join(
            turn.transcript or "" for turn in turns.turns
        ).strip()
        if terms and not all(term in transcript.lower() for term in terms):
            if asr_provider is None:
                return None
            transcript = asr_provider.transcribe(turns)
            transcript_used = True
            if not all(term in transcript.lower() for term in terms):
                return None
        bindings = {
            "participant_a": participant_a,
            "participant_b": participant_b,
        }
        confidence = max(
            0.0,
            min(
                1.0,
                0.45
                + 0.25 * min(1.0, turns.speaker_count / max(self.minimum_speakers, 1))
                + 0.30 * max(0.0, 1.0 - mean_distance / self.maximum_distance_m),
            ),
        )
        return InteractionPredicateObservation(
            occurrence_id=phase8_occurrence_id(
                "CONVERSATION",
                bindings,
                overlap,
                self.provider_id,
            ),
            predicate_id="CONVERSATION",
            truth=True,
            confidence=confidence,
            event_time_interval=overlap,
            bindings=bindings,
            source_ids=tuple(sorted({item.source_id for item in track_sets} | {turns.source_id})),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            supporting_artifact_types=("track_summary.v1", "speaker_turn_set.v1"),
            measurements={
                "speaker_count": turns.speaker_count,
                "mean_distance_m": mean_distance,
                "content_required": bool(terms),
                "asr_invoked": transcript_used,
                # Store only whether terms matched; do not retain the transcript
                # in the compact predicate artifact by default.
                "required_terms_matched": bool(terms),
            },
        )

    def _proximity_interval(
        self,
        track_sets: Sequence[TrackSet],
    ) -> tuple[str, str, EventTimeInterval, float] | None:
        if not track_sets:
            return None
        ordered = sorted(track_sets, key=lambda item: item.event_time)
        pair_rows: dict[tuple[str, str], list[tuple[object, float]]] = {}
        for track_set in ordered:
            persons = sorted(
                (
                    item
                    for item in track_set.tracks
                    if item.class_name.lower() == "person"
                ),
                key=lambda item: item.scoped_track_id,
            )
            for index, left in enumerate(persons):
                for right in persons[index + 1 :]:
                    distance = _distance(left, right)
                    if distance <= self.maximum_distance_m:
                        pair_rows.setdefault(
                            (left.scoped_track_id, right.scoped_track_id), []
                        ).append((track_set.event_time, distance))
        candidates: list[tuple[float, str, str, EventTimeInterval, float]] = []
        for (left, right), rows in pair_rows.items():
            rows.sort(key=lambda item: item[0])
            start = rows[0][0]
            end = rows[-1][0]
            duration = (end - start).total_seconds()
            if duration < self.minimum_duration_seconds:
                continue
            interval = EventTimeInterval(start=start, end=end)
            mean_distance = sum(item[1] for item in rows) / len(rows)
            candidates.append((-duration, left, right, interval, mean_distance))
        if not candidates:
            return None
        _, left, right, interval, mean_distance = sorted(candidates)[0]
        return left, right, interval, mean_distance


def _point(track: TrackObservation) -> tuple[float, float]:
    if track.world_point is not None:
        return track.world_point.x, track.world_point.y
    return track.bbox.center


def _distance(left: TrackObservation, right: TrackObservation) -> float:
    if left.world_point is not None and right.world_point is not None:
        if left.world_point.coordinate_frame_id != right.world_point.coordinate_frame_id:
            raise ValueError("conversation tracks use incompatible coordinate frames")
    left_point = _point(left)
    right_point = _point(right)
    return hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("speaker embeddings have incompatible dimensions")
    return sum(a * b for a, b in zip(left, right))


class SpectralSpeakerEmbeddingProvider:
    """Compact spectral embedding baseline for replay/integration validation.

    This is not a production speaker-recognition model. It provides a stable
    embedding interface so the online diarizer, artifact compatibility, and
    conversation orchestration can be exercised without bundling a checkpoint.
    """

    provider_id = "spectral_speaker_embedding"
    provider_version = "1"

    def __init__(self, *, dimension: int = 16) -> None:
        if dimension < 4:
            raise ValueError("speaker embedding dimension must be at least four")
        self.dimension = dimension

    def attach(
        self,
        window: AudioWindow,
        segments: Sequence[SpeechSegment],
    ) -> tuple[SpeechSegment, ...]:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError(
                "spectral speaker embeddings require NumPy; install the multimodal-core extra"
            ) from exc
        mono = np.asarray(window.waveform, dtype=np.float32).mean(axis=0)
        if mono.size < self.dimension:
            raise InvalidAudioInput("audio window is too short for speaker embedding")
        spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size))) + 1e-9
        log_spectrum = np.log(spectrum)
        boundaries = np.linspace(0, len(log_spectrum), self.dimension + 1, dtype=int)
        vector = np.asarray(
            [
                float(log_spectrum[boundaries[index] : boundaries[index + 1]].mean())
                for index in range(self.dimension)
            ],
            dtype=np.float32,
        )
        vector -= vector.mean()
        norm = float(np.linalg.norm(vector)) or 1.0
        embedding = tuple(float(value / norm) for value in vector)
        return tuple(segment.model_copy(update={"embedding": embedding}) for segment in segments)


class TorchScriptSpeakerEmbeddingProvider:
    """Lazy wrapper for a user-supplied TorchScript speaker embedding model."""

    provider_id = "torchscript_speaker_embedding"

    def __init__(
        self,
        *,
        model_path: str,
        model_version: str,
        expected_sample_rate_hz: int = 16_000,
        model: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.provider_version = model_version
        self.expected_sample_rate_hz = expected_sample_rate_hz
        self._model = model

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional heavy dependency
            raise OptionalDependencyError(
                "TorchScript speaker embeddings require torch; install the multimodal-models extra"
            ) from exc
        self._model = torch.jit.load(self.model_path, map_location="cpu")
        self._model.eval()
        return self._model

    def attach(
        self,
        window: AudioWindow,
        segments: Sequence[SpeechSegment],
    ) -> tuple[SpeechSegment, ...]:
        if window.sample_rate_hz != self.expected_sample_rate_hz:
            raise InvalidAudioInput(
                "speaker embedding model received an incompatible sample rate"
            )
        try:
            import numpy as np
            import torch
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError(
                "TorchScript speaker embeddings require NumPy and torch"
            ) from exc
        mono = np.asarray(window.waveform, dtype=np.float32).mean(axis=0)
        tensor = torch.from_numpy(mono).unsqueeze(0)
        with torch.inference_mode():
            output = self._load()(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        vector = output.detach().cpu().numpy().reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise InvalidAudioInput("speaker embedding model returned a zero vector")
        embedding = tuple(float(value / norm) for value in vector)
        return tuple(segment.model_copy(update={"embedding": embedding}) for segment in segments)
