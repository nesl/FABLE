"""Versioned, synchronized recording manifests for refactored FABLE replay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ReplaySource:
    source_id: str
    modality: str
    path: Path
    event_time_start: datetime
    offset_seconds: float = 0.0


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
        recording = Path(str(item["path"]))
        recording = recording if recording.is_absolute() else (source.parent / recording).resolve()
        if not recording.is_file():
            raise FileNotFoundError(recording)
        modality = str(item["modality"])
        if modality not in {"video", "audio"}:
            raise ValueError(f"unsupported replay modality {modality!r}")
        rows.append(
            ReplaySource(
                source_id,
                modality,
                recording,
                _time(item["event_time_start"]),
                float(item.get("offset_seconds", 0.0)),
            )
        )
    if not rows:
        raise ValueError("replay manifest must contain at least one source")
    return ReplayManifest(1, str(raw["replay_id"]), speed, tuple(rows))
