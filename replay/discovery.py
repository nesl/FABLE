"""Conservative recording discovery; it never mutates or mounts data roots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_TIMESTAMP = re.compile(r"(?<!\d)(?P<stamp>\d{13}|\d{8}[_-]?\d{6})(?!\d)")
_MEDIA_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".wav", ".flac"})


@dataclass(frozen=True)
class RecordingFile:
    path: Path
    device: str
    timestamp_token: str | None
    modality: str


def discover_recordings(root: str | Path) -> tuple[RecordingFile, ...]:
    """Index recognizable media beneath an explicitly supplied existing root."""
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"recording root does not exist: {base}")
    found: list[RecordingFile] = []
    for path in sorted(base.rglob("*")):
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in _MEDIA_SUFFIXES:
            continue
        match = _TIMESTAMP.search(path.stem)
        relative = path.relative_to(base)
        device = relative.parts[0] if len(relative.parts) > 1 else base.name
        found.append(RecordingFile(
            path=path,
            device=device,
            timestamp_token=match.group("stamp") if match else None,
            modality="audio" if suffix in {".wav", ".flac"} else "video",
        ))
    return tuple(found)
