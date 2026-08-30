#!/usr/bin/env python3
"""Generate timestamp-overlap candidates; output is never marked verified."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import yaml

from evaluation.catalog import load_experiment_catalog
from replay.discovery import discover_recordings


def _recording_start(token: str | None) -> datetime | None:
    if token is None:
        return None
    if len(token) == 13:
        return datetime.fromtimestamp(int(token) / 1000, tz=timezone.utc)
    return None


def _duration(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=False, capture_output=True, text=True, timeout=10,
    )
    try:
        return max(0.0, float(completed.stdout.strip())) if completed.returncode == 0 else 0.0
    except ValueError:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("recording_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--padding-seconds", type=float, default=5.0)
    args = parser.parse_args()
    zone = ZoneInfo(args.timezone)
    device_roots = sorted(
        path for path in args.recording_root.iterdir()
        if path.is_dir() and path.name.lower().startswith("mobile ")
    )
    discovery_roots = device_roots or [args.recording_root]
    discovered = tuple(
        item for root in discovery_roots for item in discover_recordings(root)
    )
    timestamped = [
        (item, _recording_start(item.timestamp_token))
        for item in discovered
    ]
    timestamped = [(item, started) for item, started in timestamped if started is not None]
    with ThreadPoolExecutor(max_workers=12) as executor:
        durations = executor.map(lambda row: _duration(row[0].path), timestamped)
        recordings = [
            (item, started, duration)
            for (item, started), duration in zip(timestamped, durations)
        ]
    experiments = []
    for record in load_experiment_catalog(args.catalog):
        if not record.recording_start or not record.recording_end:
            continue
        start = datetime.fromisoformat(record.recording_start).replace(tzinfo=zone).astimezone(timezone.utc)
        end = datetime.fromisoformat(record.recording_end).replace(tzinfo=zone).astimezone(timezone.utc)
        best_by_source = {}
        for item, media_start, duration in recordings:
            media_end = media_start.timestamp() + duration
            if media_start.timestamp() <= end.timestamp() + args.padding_seconds and media_end >= start.timestamp() - args.padding_seconds:
                source_id = item.device.lower().replace(" ", "_")
                overlap = max(0.0, min(media_end, end.timestamp()) - max(media_start.timestamp(), start.timestamp()))
                candidate = ({
                    "source_id": item.device.lower().replace(" ", "_"),
                    "path": str(item.path),
                    "start_offset_seconds": max(0.0, start.timestamp() - media_start.timestamp()),
                }, overlap)
                if source_id not in best_by_source or overlap > best_by_source[source_id][1]:
                    best_by_source[source_id] = candidate
        matches = [best_by_source[key][0] for key in sorted(best_by_source)]
        if matches:
            experiments.append({
                "experiment_id": record.experiment_id,
                "verified": False,
                "recordings": matches,
            })
    document = {
        "version": 1,
        "generated": True,
        "warning": "Timestamp-overlap candidates only; manually verify before evaluation.",
        "experiments": experiments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(f"wrote {len(experiments)} candidate experiment mappings to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
