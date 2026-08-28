"""Resolve Android mobile-camera recordings without assuming handset identity."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PREFIXED_FILENAME = re.compile(
    r"^recording_(?P<prefix>.+)_(?P<epoch_ms>\d{13})\.mp4$"
)
TIMESTAMP_FILENAME = re.compile(r"^recording__(?P<epoch_ms>\d{13})\.mp4$")
ARCHIVE = re.compile(r"^mobile\s+(?P<number>\d+)\s+all$", re.IGNORECASE)


@dataclass(frozen=True)
class MobileRecordingSegment:
    path: Path
    recording_start: datetime
    duration_seconds: float
    trim_start_seconds: float
    trim_end_seconds: float


@dataclass(frozen=True)
class MobileRecording:
    archive_id: str
    logical_id: str
    path: Path
    recording_start: datetime
    duration_seconds: float
    trim_start_seconds: float
    trim_end_seconds: float
    segments: tuple[MobileRecordingSegment, ...] = ()
    timeline_start: datetime | None = None


def load_alias_map(path: Path | None) -> dict[str, str]:
    """Load an explicit archive-to-topology mapping.

    Keys must be stable archive IDs (``mobile_archive_1``), never directory
    ordinals interpreted as topology identities.
    """

    if path is None:
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    aliases = document.get("archive_to_logical_id", document)
    if not isinstance(aliases, dict):
        raise ValueError("mobile alias map must be a JSON object")
    return {str(key): str(value) for key, value in aliases.items()}


def resolve_mobile_recordings(
    root: Path,
    *,
    recording_prefix: str | None,
    allow_any_prefix: bool = False,
    event_start: datetime,
    event_end: datetime,
    alias_map: dict[str, str] | None = None,
    local_timezone: str = "America/New_York",
    minimum_coverage_fraction: float = 0.5,
    maximum_required_coverage_seconds: float = 60.0,
) -> tuple[MobileRecording, ...]:
    """Select one covering recording per physical archive.

    Android filenames carry Unix epoch milliseconds, while the scenario
    catalog's legacy timestamps are site-local and timezone-naive. Files are
    matched in UTC after explicitly localizing those catalog timestamps.
    When retry files overlap, the largest covering file wins.
    """

    if event_start.tzinfo is None:
        event_start = event_start.replace(tzinfo=ZoneInfo(local_timezone))
    if event_end.tzinfo is None:
        event_end = event_end.replace(tzinfo=ZoneInfo(local_timezone))
    start_ts = event_start.timestamp()
    end_ts = event_end.timestamp()
    aliases = alias_map or {}
    selected: list[MobileRecording] = []
    for directory in sorted(root.iterdir()):
        match = ARCHIVE.match(directory.name)
        if not match or not directory.is_dir():
            continue
        archive_id = f"mobile_archive_{int(match.group('number'))}"
        candidates = []
        pattern = (
            f"recording_{recording_prefix}_*.mp4"
            if recording_prefix
            else "recording_*_*.mp4" if allow_any_prefix else "recording__*.mp4"
        )
        for path in directory.glob(pattern):
            # A failed/partial recorder or an external staging process can leave
            # a directory whose name ends in .mp4.  Glob matching alone does not
            # establish that the candidate is probeable media.
            if not path.is_file():
                continue
            name = (
                PREFIXED_FILENAME.match(path.name)
                if recording_prefix or allow_any_prefix
                else TIMESTAMP_FILENAME.match(path.name)
            )
            if name is None or (
                recording_prefix
                and name.group("prefix") != recording_prefix
            ):
                continue
            file_start = int(name.group("epoch_ms")) / 1000.0
            # Android recordings are normally five-minute chunks. Avoid
            # probing an entire multi-day archive when only a narrow scenario
            # interval can possibly overlap. The one-hour lookback safely
            # covers retries and unusually long chunks observed in the corpus.
            if file_start > end_ts or file_start < start_ts - 3600:
                continue
            try:
                duration = _duration(path)
            except (OSError, subprocess.SubprocessError, ValueError):
                # One malformed archive member must not make every otherwise
                # usable mobile source (or the complete CE bundle) unavailable.
                continue
            overlap = max(
                0.0,
                min(end_ts, file_start + duration) - max(start_ts, file_start),
            )
            required = min(
                max(0.0, end_ts - start_ts) * minimum_coverage_fraction,
                maximum_required_coverage_seconds,
            )
            if overlap > 0:
                candidates.append(
                    (overlap, path.stat().st_size, duration, file_start, path)
                )
        if not candidates:
            continue
        segments = _covering_segments(
            candidates,
            event_start_ts=start_ts,
            event_end_ts=end_ts,
        )
        coverage = sum(
            item.trim_end_seconds - item.trim_start_seconds for item in segments
        )
        if coverage < required:
            continue
        first = segments[0]
        selected.append(
            MobileRecording(
                archive_id=archive_id,
                logical_id=aliases.get(archive_id, archive_id),
                path=first.path,
                recording_start=first.recording_start,
                duration_seconds=first.duration_seconds,
                trim_start_seconds=first.trim_start_seconds,
                trim_end_seconds=first.trim_end_seconds,
                segments=segments,
                timeline_start=datetime.fromtimestamp(
                    start_ts, tz=ZoneInfo("UTC")
                ),
            )
        )
    return tuple(selected)


def _covering_segments(
    candidates: list[tuple[float, int, float, float, Path]],
    *,
    event_start_ts: float,
    event_end_ts: float,
) -> tuple[MobileRecordingSegment, ...]:
    """Choose non-overlapping pieces from timestamped adjacent recordings."""

    remaining = [
        {
            "duration": duration,
            "start": file_start,
            "end": file_start + duration,
            "path": path.resolve(),
            "size": size,
        }
        for _, size, duration, file_start, path in candidates
    ]
    cursor = event_start_ts
    chosen: list[MobileRecordingSegment] = []
    while cursor < event_end_ts:
        covering = [
            item
            for item in remaining
            if item["start"] <= cursor < item["end"]
        ]
        if covering:
            item = max(covering, key=lambda row: (row["end"], row["size"]))
        else:
            future = [item for item in remaining if item["start"] > cursor]
            if not future:
                break
            item = min(future, key=lambda row: (row["start"], -row["end"]))
            cursor = min(event_end_ts, float(item["start"]))
        piece_end = min(event_end_ts, float(item["end"]))
        if piece_end <= cursor:
            remaining.remove(item)
            continue
        file_start = float(item["start"])
        chosen.append(
            MobileRecordingSegment(
                path=Path(item["path"]),
                recording_start=datetime.fromtimestamp(
                    file_start, tz=ZoneInfo("UTC")
                ),
                duration_seconds=float(item["duration"]),
                trim_start_seconds=max(0.0, cursor - file_start),
                trim_end_seconds=max(0.0, piece_end - file_start),
            )
        )
        cursor = piece_end
        remaining.remove(item)
    return tuple(chosen)


def _duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return float(result.stdout.strip())
