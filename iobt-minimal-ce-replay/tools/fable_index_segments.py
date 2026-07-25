#!/usr/bin/env python3
"""Index replay files into the node-local Phase-6 historical segment store."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fable.common.time import EventTimeInterval, utc_now
from fable.distributed.models import SegmentRef
from fable.distributed.segment_store import SegmentStore

TIMESTAMP = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})(?:_(?P<micro>\d{1,6}))?")


def timestamp_from_name(path: Path) -> datetime | None:
    match = TIMESTAMP.search(path.name)
    if not match:
        return None
    micro = (match.group("micro") or "0").ljust(6, "0")[:6]
    return datetime.strptime(
        f"{match.group('date')}_{match.group('time')}_{micro}",
        "%Y%m%d_%H%M%S_%f",
    ).replace(tzinfo=UTC)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--glob", default="**/*")
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--retention-sec", type=float, default=1800.0)
    parser.add_argument("--media-type", default="application/octet-stream")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    store = SegmentStore(args.db)
    registered = skipped = 0
    try:
        for path in sorted(args.root.glob(args.glob)):
            if not path.is_file():
                continue
            start = timestamp_from_name(path)
            if start is None:
                skipped += 1
                continue
            segment = SegmentRef(
                source_id=args.source_id,
                path=str(path.resolve()),
                event_time_interval=EventTimeInterval(
                    start=start,
                    end=start + timedelta(seconds=args.duration_sec),
                ),
                bytes=path.stat().st_size,
                media_type=args.media_type,
                created_at=utc_now(),
                expires_at=utc_now() + timedelta(seconds=args.retention_sec),
                metadata={"indexed_from": "iobt-minimal-ce-replay"},
            )
            if args.dry_run:
                print(segment.model_dump_json())
            else:
                store.register(segment)
            registered += 1
    finally:
        store.close()
    print(f"registered={registered} skipped_without_timestamp={skipped} db={args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
