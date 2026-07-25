"""Microphone-array localization using GCC-PHAT time-difference estimates."""

from __future__ import annotations

from math import atan2, degrees, sqrt
from typing import Sequence

from fable.common.ids import deterministic_id

from .errors import InvalidAudioInput, OptionalDependencyError
from .models import (
    AudioLocalization,
    AudioWindow,
    BearingZone,
    MicrophoneArrayGeometry,
)


class GccPhatAudioLocalizer:
    """Estimate a 2-D arrival bearing from a calibrated microphone array.

    The implementation is intentionally explicit and checkpoint-friendly: it
    emits a compact typed localization artifact rather than retaining an opaque
    signal-processing object.  Geometry and event-time validity are part of the
    result provenance.
    """

    def __init__(
        self,
        geometry: MicrophoneArrayGeometry,
        *,
        bearing_zones: Sequence[BearingZone] = (),
        maximum_tau_seconds: float | None = None,
        interpolation: int = 8,
        provider_id: str = "gcc_phat_audio_localizer",
        provider_version: str = "1",
    ) -> None:
        if interpolation < 1:
            raise ValueError("GCC-PHAT interpolation must be positive")
        self.geometry = geometry
        self.bearing_zones = tuple(bearing_zones)
        self.maximum_tau_seconds = maximum_tau_seconds
        self.interpolation = interpolation
        self.provider_id = provider_id
        self.provider_version = provider_version

    def localize(self, window: AudioWindow) -> AudioLocalization:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OptionalDependencyError(
                "GCC-PHAT localization requires NumPy; install the multimodal-core extra"
            ) from exc
        channel_map = {channel_id: index for index, channel_id in enumerate(window.channel_ids)}
        missing = [
            mic.microphone_id
            for mic in self.geometry.microphones
            if mic.microphone_id not in channel_map
        ]
        if missing:
            raise InvalidAudioInput(
                f"audio window is missing calibrated microphone channels {missing}"
            )
        reference = next(
            item
            for item in self.geometry.microphones
            if item.microphone_id == self.geometry.reference_microphone_id
        )
        reference_signal = np.asarray(
            window.waveform[channel_map[reference.microphone_id]], dtype=np.float64
        )
        if reference_signal.size < 8:
            raise InvalidAudioInput("audio window is too short for GCC-PHAT")

        rows: list[list[float]] = []
        values: list[float] = []
        delays: dict[str, float] = {}
        peak_scores: list[float] = []
        for microphone in self.geometry.microphones:
            if microphone.microphone_id == reference.microphone_id:
                continue
            signal = np.asarray(
                window.waveform[channel_map[microphone.microphone_id]], dtype=np.float64
            )
            displacement = np.asarray(
                [
                    microphone.x_m - reference.x_m,
                    microphone.y_m - reference.y_m,
                ],
                dtype=np.float64,
            )
            spacing = float(np.linalg.norm(displacement))
            physical_max = spacing / self.geometry.speed_of_sound_mps
            max_tau = (
                min(physical_max, self.maximum_tau_seconds)
                if self.maximum_tau_seconds is not None
                else physical_max
            )
            delay, peak = gcc_phat_delay(
                signal,
                reference_signal,
                sample_rate_hz=window.sample_rate_hz,
                maximum_tau_seconds=max_tau,
                interpolation=self.interpolation,
            )
            rows.append([float(displacement[0]), float(displacement[1])])
            # For a far-field wave, tau = -(r_i-r_ref) dot u / c.
            values.append(-self.geometry.speed_of_sound_mps * delay)
            delays[f"{reference.microphone_id}->{microphone.microphone_id}"] = delay
            peak_scores.append(peak)

        matrix = np.asarray(rows, dtype=np.float64)
        target = np.asarray(values, dtype=np.float64)
        if matrix.shape[0] < 1 or np.linalg.matrix_rank(matrix) < 1:
            raise InvalidAudioInput("microphone geometry cannot constrain a bearing")
        direction, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            raise InvalidAudioInput("audio delays do not produce a stable bearing")
        direction = direction / norm
        predicted = matrix @ direction
        residual_distance = float(np.sqrt(np.mean((predicted - target) ** 2)))
        residual_seconds = residual_distance / self.geometry.speed_of_sound_mps
        azimuth = _normalize_angle(degrees(atan2(float(direction[1]), float(direction[0]))))
        peak_quality = sum(peak_scores) / max(len(peak_scores), 1)
        spacing_scale = max(
            sqrt(
                (item.x_m - reference.x_m) ** 2
                + (item.y_m - reference.y_m) ** 2
            )
            for item in self.geometry.microphones
            if item.microphone_id != reference.microphone_id
        )
        residual_quality = max(
            0.0,
            1.0 - residual_distance / max(spacing_scale, 1e-6),
        )
        confidence = max(0.0, min(1.0, 0.55 * peak_quality + 0.45 * residual_quality))
        zone_id = next(
            (zone.zone_id for zone in self.bearing_zones if zone.contains(azimuth)),
            None,
        )
        localization_id = deterministic_id(
            "audio_localization",
            {
                "array_id": self.geometry.array_id,
                "source_id": window.source_id,
                "interval": window.event_time_interval,
                "delays": delays,
            },
            length=40,
        )
        return AudioLocalization(
            localization_id=localization_id,
            source_id=window.source_id,
            array_id=self.geometry.array_id,
            event_time_interval=window.event_time_interval,
            azimuth_deg=azimuth,
            confidence=confidence,
            zone_id=zone_id,
            pair_delays_seconds=delays,
            residual_seconds=residual_seconds,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )


def gcc_phat_delay(
    signal: object,
    reference_signal: object,
    *,
    sample_rate_hz: int,
    maximum_tau_seconds: float | None = None,
    interpolation: int = 8,
) -> tuple[float, float]:
    """Return relative delay and a normalized correlation-peak quality.

    A positive delay means ``signal`` arrives later than ``reference_signal``.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("GCC-PHAT requires NumPy") from exc
    left = np.asarray(signal, dtype=np.float64)
    right = np.asarray(reference_signal, dtype=np.float64)
    if left.ndim != 1 or right.ndim != 1 or min(left.size, right.size) < 2:
        raise InvalidAudioInput("GCC-PHAT inputs must be non-empty one-dimensional signals")
    size = left.size + right.size
    fft_size = 1 << int((size - 1).bit_length())
    left_fft = np.fft.rfft(left, n=fft_size)
    right_fft = np.fft.rfft(right, n=fft_size)
    cross = left_fft * np.conj(right_fft)
    cross /= np.maximum(np.abs(cross), 1e-15)
    correlation = np.fft.irfft(cross, n=interpolation * fft_size)
    max_shift = interpolation * fft_size // 2
    if maximum_tau_seconds is not None:
        max_shift = min(
            max_shift,
            int(interpolation * sample_rate_hz * maximum_tau_seconds),
        )
    correlation = np.concatenate((correlation[-max_shift:], correlation[: max_shift + 1]))
    magnitudes = np.abs(correlation)
    shift = int(np.argmax(magnitudes)) - max_shift
    peak = float(magnitudes.max())
    mean = float(magnitudes.mean() + 1e-12)
    peak_quality = max(0.0, min(1.0, (peak / mean - 1.0) / 20.0))
    delay = shift / float(interpolation * sample_rate_hz)
    return delay, peak_quality


def _normalize_angle(value: float) -> float:
    normalized = ((value + 180.0) % 360.0) - 180.0
    return 180.0 if normalized == -180.0 and value > 0 else normalized
