"""Explicit, reviewable mapping between catalog experiments and recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RecordingBinding:
    source_id: str
    path: Path
    start_offset_seconds: float = 0.0


@dataclass(frozen=True)
class ExperimentRecordingMap:
    experiment_id: str
    verified: bool
    recordings: tuple[RecordingBinding, ...]


def load_recording_map(path: str | Path) -> tuple[ExperimentRecordingMap, ...]:
    source = Path(path).resolve()
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if int(raw.get("version", 0)) != 1:
        raise ValueError("recording map version must be 1")
    output = []
    seen: set[str] = set()
    for item in raw.get("experiments", ()):
        experiment_id = str(item["experiment_id"])
        if experiment_id in seen:
            raise ValueError(f"duplicate experiment mapping {experiment_id!r}")
        seen.add(experiment_id)
        bindings = []
        source_ids: set[str] = set()
        for row in item.get("recordings", ()):
            source_id = str(row["source_id"])
            if source_id in source_ids:
                raise ValueError(f"duplicate source {source_id!r} in {experiment_id!r}")
            source_ids.add(source_id)
            recording = Path(str(row["path"]))
            recording = recording if recording.is_absolute() else (source.parent / recording).resolve()
            if not recording.is_file():
                raise FileNotFoundError(recording)
            bindings.append(RecordingBinding(
                source_id, recording, float(row.get("start_offset_seconds", 0.0))
            ))
        if not bindings:
            raise ValueError(f"experiment {experiment_id!r} has no recordings")
        output.append(ExperimentRecordingMap(
            experiment_id, bool(item.get("verified", False)), tuple(bindings)
        ))
    if not output:
        raise ValueError("recording map contains no experiments")
    return tuple(output)
