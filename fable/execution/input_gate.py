"""Node-owned hysteresis gates for replay inputs."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping

from fable.providers.data_models import AudioWindow, VideoFrame


@dataclass(frozen=True, slots=True)
class InputGateConfig:
    kind: str
    on_threshold: float
    off_threshold: float

    def __post_init__(self) -> None:
        if self.kind not in {"video_frame_difference", "audio_rms"}:
            raise ValueError(f"unsupported input gate {self.kind!r}")
        if self.off_threshold < 0 or self.on_threshold < 0:
            raise ValueError("input gate thresholds must be non-negative")
        if self.off_threshold >= self.on_threshold:
            raise ValueError("off_threshold must be lower than on_threshold")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "InputGateConfig":
        unknown = set(raw) - {"type", "on_threshold", "off_threshold"}
        if unknown:
            raise ValueError(f"unknown input gate fields: {sorted(unknown)}")
        return cls(
            str(raw["type"]),
            float(raw["on_threshold"]),
            float(raw["off_threshold"]),
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "type": self.kind,
            "on_threshold": self.on_threshold,
            "off_threshold": self.off_threshold,
        }


class HysteresisInputGate:
    """Open above the on threshold and close below the off threshold."""

    def __init__(self, config: InputGateConfig) -> None:
        self.config = config
        self.is_open = False
        self._previous_gray = None

    def accept(self, value: object) -> bool:
        score = self.score(value)
        if self.is_open:
            if score <= self.config.off_threshold:
                self.is_open = False
        elif score >= self.config.on_threshold:
            self.is_open = True
        return self.is_open

    def score(self, value: object) -> float:
        if self.config.kind == "video_frame_difference":
            if not isinstance(value, VideoFrame):
                raise TypeError("video frame-difference gate requires VideoFrame")
            return self._video_difference(value.image)
        if not isinstance(value, AudioWindow):
            raise TypeError("audio RMS gate requires AudioWindow")
        if not value.samples:
            return 0.0
        return sqrt(sum(float(sample) ** 2 for sample in value.samples) / len(value.samples))

    def _video_difference(self, image: object) -> float:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - video adapters need NumPy
            raise RuntimeError("NumPy is required for video activity gating") from exc
        array = np.asarray(image)
        if array.ndim == 3:
            gray = array[..., :3].astype(np.float32).mean(axis=2)
        elif array.ndim == 2:
            gray = array.astype(np.float32)
        else:
            raise ValueError("video gate expects a two- or three-dimensional image")
        # A small fixed thumbnail keeps this node-level check inexpensive.
        row_step = max(1, gray.shape[0] // 64)
        col_step = max(1, gray.shape[1] // 64)
        gray = gray[::row_step, ::col_step]
        previous, self._previous_gray = self._previous_gray, gray
        if previous is None or previous.shape != gray.shape:
            return 0.0
        return float(np.mean(np.abs(gray - previous)) / 255.0)


def parse_input_gates(raw: Mapping[str, Any] | None) -> dict[str, InputGateConfig]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("input_gates must be a mapping by source ID")
    return {
        str(source_id): InputGateConfig.from_mapping(spec)
        for source_id, spec in raw.items()
    }
