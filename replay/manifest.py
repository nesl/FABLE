"""Versioned, synchronized recording manifests for refactored FABLE replay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ReplaySegment:
    path: Path
    event_time_start: datetime
    offset_seconds: float = 0.0
    duration_seconds: float | None = None


@dataclass(frozen=True)
class ReplaySource:
    source_id: str
    modality: str
    path: Path
    event_time_start: datetime
    offset_seconds: float = 0.0
    rotation_degrees: int = 0
    segments: tuple[ReplaySegment, ...] = ()


@dataclass(frozen=True)
class ReplayManifest:
    version: int
    replay_id: str
    speed: float
    sources: tuple[ReplaySource, ...]


def _time(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("replay event times must include a UTC offset")
    return parsed


def load_replay_manifest(path: str | Path) -> ReplayManifest:
    source = Path(path).resolve()
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if int(raw.get("version", 0)) != 1:
        raise ValueError("replay manifest version must be 1")
    speed = float(raw.get("speed", 1.0))
    if speed <= 0:
        raise ValueError("replay speed must be positive")
    rows = []
    seen: set[str] = set()
    for item in raw.get("sources", ()):
        source_id = str(item["source_id"])
        if source_id in seen:
            raise ValueError(f"duplicate replay source {source_id!r}")
        seen.add(source_id)
        segments = []
        for segment in item.get("segments", ()):
            segment_path = Path(str(segment["path"]))
            segment_path = segment_path if segment_path.is_absolute() else (source.parent / segment_path).resolve()
            if not segment_path.is_file():
                raise FileNotFoundError(segment_path)
            duration = segment.get("duration_seconds")
            if duration is not None and float(duration) <= 0:
                raise ValueError("segment duration_seconds must be positive")
            segments.append(ReplaySegment(
                segment_path, _time(segment["event_time_start"]),
                float(segment.get("offset_seconds", 0.0)),
                None if duration is None else float(duration),
            ))
        recording_value = item.get("path")
        if recording_value is None and not segments:
            raise ValueError(f"replay source {source_id!r} requires path or segments")
        recording = segments[0].path if recording_value is None else Path(str(recording_value))
        recording = recording if recording.is_absolute() else (source.parent / recording).resolve()
        if not recording.is_file():
            raise FileNotFoundError(recording)
        modality = str(item["modality"])
        if modality not in {"video", "audio"}:
            raise ValueError(f"unsupported replay modality {modality!r}")
        rotation = int(item.get("rotation_degrees", 0))
        if rotation not in {0, 90, 180, 270}:
            raise ValueError("rotation_degrees must be one of 0, 90, 180, or 270")
        if modality != "video" and rotation:
            raise ValueError("rotation_degrees is only valid for video replay sources")
        rows.append(
            ReplaySource(
                source_id,
                modality,
                recording,
                _time(item.get("event_time_start", segments[0].event_time_start.isoformat() if segments else None)),
                float(item.get("offset_seconds", 0.0)),
                rotation,
                tuple(segments),
            )
        )
    if not rows:
        raise ValueError("replay manifest must contain at least one source")
    return ReplayManifest(1, str(raw["replay_id"]), speed, tuple(rows))
