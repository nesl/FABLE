"""Node-local raw segment catalog with event-time interval lookup."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Literal

from fable.common.time import EventTimeInterval, ensure_utc, utc_now

from .models import SegmentRef

UTC = timezone.utc


class SegmentStore:
    """SQLite index over node-local media files.

    The store does not copy media.  It records durable paths and intervals so a
    retrospective demand can verify that every required interval is still
    locally available before provider execution is admitted.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._initialize()

    def _initialize(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    event_start TEXT NOT NULL,
                    event_end TEXT NOT NULL,
                    bytes INTEGER NOT NULL,
                    checksum TEXT,
                    media_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_segments_source_time "
                "ON segments(source_id, event_start, event_end)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_segments_expiry ON segments(expires_at)"
            )

    def register(self, segment: SegmentRef, *, require_file: bool = True) -> bool:
        path = Path(segment.path)
        if require_file and not path.is_file():
            raise FileNotFoundError(path)
        payload = segment.model_dump_json(exclude_none=False)
        import json

        metadata_json = json.dumps(segment.metadata, sort_keys=True, separators=(",", ":"))
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT path, event_start, event_end FROM segments WHERE segment_id=?",
                (segment.segment_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["path"] != segment.path
                    or existing["event_start"] != _iso(segment.event_time_interval.start)
                    or existing["event_end"] != _iso(segment.event_time_interval.end)
                ):
                    raise ValueError(f"segment ID {segment.segment_id} has conflicting content")
                return False
            self._conn.execute(
                """
                INSERT INTO segments(
                    segment_id, source_id, path, event_start, event_end, bytes,
                    checksum, media_type, created_at, expires_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    segment.segment_id,
                    segment.source_id,
                    segment.path,
                    _iso(segment.event_time_interval.start),
                    _iso(segment.event_time_interval.end),
                    segment.bytes,
                    segment.checksum,
                    segment.media_type,
                    _iso(segment.created_at),
                    None if segment.expires_at is None else _iso(segment.expires_at),
                    metadata_json,
                ),
            )
            return True

    def query(
        self,
        *,
        source_id: str,
        interval: EventTimeInterval,
        mode: Literal["overlap", "contained", "covering"] = "overlap",
        now: datetime | None = None,
        require_existing_file: bool = True,
    ) -> tuple[SegmentRef, ...]:
        observed_now = ensure_utc(now or utc_now())
        if mode == "overlap":
            predicate = "event_start < ? AND event_end > ?"
            args = (_iso(interval.end), _iso(interval.start))
        elif mode == "contained":
            predicate = "event_start >= ? AND event_end <= ?"
            args = (_iso(interval.start), _iso(interval.end))
        elif mode == "covering":
            predicate = "event_start <= ? AND event_end >= ?"
            args = (_iso(interval.start), _iso(interval.end))
        else:
            raise ValueError(f"unknown segment query mode {mode}")
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM segments
                WHERE source_id=? AND {predicate}
                  AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY event_start, event_end, segment_id
                """,
                (source_id, *args, _iso(observed_now)),
            ).fetchall()
        segments = tuple(self._row_to_segment(row) for row in rows)
        if require_existing_file:
            segments = tuple(item for item in segments if Path(item.path).is_file())
        return segments

    def covers(
        self,
        *,
        source_id: str,
        interval: EventTimeInterval,
        now: datetime | None = None,
        require_existing_file: bool = True,
    ) -> bool:
        segments = self.query(
            source_id=source_id,
            interval=interval,
            mode="overlap",
            now=now,
            require_existing_file=require_existing_file,
        )
        if not segments:
            return False
        cursor = interval.start
        for segment in segments:
            if segment.event_time_interval.end <= cursor:
                continue
            if segment.event_time_interval.start > cursor:
                return False
            cursor = max(cursor, segment.event_time_interval.end)
            if cursor >= interval.end:
                return True
        return False

    def source_buffer_interval(
        self,
        source_id: str,
        *,
        now: datetime | None = None,
        require_existing_file: bool = True,
    ) -> EventTimeInterval | None:
        observed_now = ensure_utc(now or utc_now())
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM segments
                WHERE source_id=? AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY event_start, event_end
                """,
                (source_id, _iso(observed_now)),
            ).fetchall()
        segments = [self._row_to_segment(row) for row in rows]
        if require_existing_file:
            segments = [item for item in segments if Path(item.path).is_file()]
        if not segments:
            return None
        return EventTimeInterval(
            start=min(item.event_time_interval.start for item in segments),
            end=max(item.event_time_interval.end for item in segments),
        )

    def expire(self, *, now: datetime | None = None, delete_files: bool = False) -> tuple[str, ...]:
        observed_now = ensure_utc(now or utc_now())
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM segments WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (_iso(observed_now),),
            ).fetchall()
        expired = [self._row_to_segment(row) for row in rows]
        if delete_files:
            for segment in expired:
                try:
                    Path(segment.path).unlink(missing_ok=True)
                except OSError:
                    pass
        ids = tuple(item.segment_id or "" for item in expired)
        if ids:
            placeholders = ",".join("?" for _ in ids)
            with self._lock, self._conn:
                self._conn.execute(
                    f"DELETE FROM segments WHERE segment_id IN ({placeholders})", ids
                )
        return ids

    def get(self, segment_id: str) -> SegmentRef | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM segments WHERE segment_id=?", (segment_id,)
            ).fetchone()
        return None if row is None else self._row_to_segment(row)

    @property
    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()
        return int(row["n"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_to_segment(row: sqlite3.Row) -> SegmentRef:
        import json

        return SegmentRef(
            segment_id=row["segment_id"],
            source_id=row["source_id"],
            path=row["path"],
            event_time_interval=EventTimeInterval(
                start=datetime.fromisoformat(row["event_start"]),
                end=datetime.fromisoformat(row["event_end"]),
            ),
            bytes=int(row["bytes"]),
            checksum=row["checksum"],
            media_type=row["media_type"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=None if row["expires_at"] is None else datetime.fromisoformat(row["expires_at"]),
            metadata=json.loads(row["metadata_json"]),
        )


def _iso(value: datetime) -> str:
    return ensure_utc(value).isoformat()
