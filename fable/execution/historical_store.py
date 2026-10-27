"""Node-local historical segment and compact evidence storage."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import uuid
from typing import Literal, Mapping

from .stream_bus import StreamBus, StreamKey, Subscription


def _aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class TimeInterval:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        start, end = _aware(self.start, "start"), _aware(self.end, "end")
        if end < start:
            raise ValueError("interval end precedes start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True, slots=True)
class HistoricalSegment:
    segment_id: str
    source_id: str
    path: str
    interval: TimeInterval
    media_type: str = "application/octet-stream"
    checksum: str | None = None
    expires_at: datetime | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segment_id or not self.source_id or not self.path:
            raise ValueError("segment_id, source_id, and path are required")
        if self.expires_at is not None:
            object.__setattr__(self, "expires_at", _aware(self.expires_at, "expires_at"))


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    record_id: str
    source_id: str
    event_time: datetime
    kind: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_time", _aware(self.event_time, "event_time"))


class HistoricalStore:
    """SQLite index; media stays as node-local files referenced by path."""

    def __init__(self, database: str | Path) -> None:
        path = Path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._connection:
            self._connection.executescript("""
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, path TEXT NOT NULL,
                    event_start TEXT NOT NULL, event_end TEXT NOT NULL, media_type TEXT NOT NULL,
                    checksum TEXT, expires_at TEXT, metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS segments_source_time
                    ON segments(source_id, event_start, event_end);
                CREATE TABLE IF NOT EXISTS evidence (
                    record_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, event_time TEXT NOT NULL,
                    kind TEXT NOT NULL, payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS evidence_source_time
                    ON evidence(source_id, event_time);
            """)

    def add_segment(self, segment: HistoricalSegment) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "INSERT OR IGNORE INTO segments VALUES (?,?,?,?,?,?,?,?,?)",
                (segment.segment_id, segment.source_id, segment.path,
                 segment.interval.start.isoformat(), segment.interval.end.isoformat(),
                 segment.media_type, segment.checksum,
                 None if segment.expires_at is None else segment.expires_at.isoformat(),
                 json.dumps(dict(segment.metadata), sort_keys=True)),
            )
        return cursor.rowcount == 1

    def add_evidence(self, record: EvidenceRecord) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?)",
                (record.record_id, record.source_id, record.event_time.isoformat(),
                 record.kind, json.dumps(dict(record.payload), sort_keys=True)),
            )
        return cursor.rowcount == 1

    def query_segments(
        self, source_id: str, interval: TimeInterval, *,
        mode: Literal["overlap", "contained", "covering"] = "overlap",
        now: datetime | None = None, require_existing_file: bool = True,
    ) -> tuple[HistoricalSegment, ...]:
        predicates = {
            "overlap": ("event_start < ? AND event_end > ?", (interval.end, interval.start)),
            "contained": ("event_start >= ? AND event_end <= ?", (interval.start, interval.end)),
            "covering": ("event_start <= ? AND event_end >= ?", (interval.start, interval.end)),
        }
        try:
            predicate, values = predicates[mode]
        except KeyError as exc:
            raise ValueError(f"unknown query mode {mode!r}") from exc
        observed = _aware(now or datetime.now(timezone.utc), "now").isoformat()
        args = (source_id, *(value.isoformat() for value in values), observed)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM segments WHERE source_id=? AND {predicate} "
                "AND (expires_at IS NULL OR expires_at > ?) ORDER BY event_start, segment_id", args,
            ).fetchall()
        result = tuple(self._segment(row) for row in rows)
        return tuple(row for row in result if Path(row.path).is_file()) if require_existing_file else result

    def query_evidence(
        self, source_id: str, interval: TimeInterval, *, kind: str | None = None,
    ) -> tuple[EvidenceRecord, ...]:
        sql = "SELECT * FROM evidence WHERE source_id=? AND event_time>=? AND event_time<=?"
        args: list[str] = [source_id, interval.start.isoformat(), interval.end.isoformat()]
        if kind is not None:
            sql += " AND kind=?"
            args.append(kind)
        with self._lock:
            rows = self._connection.execute(sql + " ORDER BY event_time, record_id", args).fetchall()
        return tuple(EvidenceRecord(row["record_id"], row["source_id"],
                                    datetime.fromisoformat(row["event_time"]), row["kind"],
                                    json.loads(row["payload_json"])) for row in rows)

    def expire(self, *, now: datetime | None = None, delete_files: bool = False) -> tuple[str, ...]:
        observed = _aware(now or datetime.now(timezone.utc), "now").isoformat()
        with self._lock:
            rows = self._connection.execute(
                "SELECT segment_id,path FROM segments WHERE expires_at IS NOT NULL AND expires_at<=?",
                (observed,),
            ).fetchall()
            if delete_files:
                for row in rows:
                    Path(row["path"]).unlink(missing_ok=True)
            with self._connection:
                self._connection.executemany(
                    "DELETE FROM segments WHERE segment_id=?", ((row["segment_id"],) for row in rows)
                )
        return tuple(row["segment_id"] for row in rows)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _segment(row: sqlite3.Row) -> HistoricalSegment:
        return HistoricalSegment(
            row["segment_id"], row["source_id"], row["path"],
            TimeInterval(datetime.fromisoformat(row["event_start"]), datetime.fromisoformat(row["event_end"])),
            row["media_type"], row["checksum"],
            None if row["expires_at"] is None else datetime.fromisoformat(row["expires_at"]),
            json.loads(row["metadata_json"]),
        )


class StreamHistoryRecorder:
    """Persist selected compact StreamBus values for retrospective queries."""

    def __init__(self, store: HistoricalStore, bus: StreamBus, data_types: tuple[str, ...]) -> None:
        self.store, self.bus = store, bus
        self._subscriptions = tuple(
            bus.subscribe(data_type, self._record) for data_type in data_types
        )

    def close(self) -> None:
        for subscription in self._subscriptions:
            self.bus.unsubscribe(subscription)

    def _record(self, key: StreamKey, value: object) -> None:
        from dataclasses import asdict, is_dataclass

        event_time = getattr(value, "event_time", None)
        if not isinstance(event_time, datetime):
            return
        payload = asdict(value) if is_dataclass(value) else {"value": repr(value)}
        for source_id in key.source_ids:
            self.store.add_evidence(EvidenceRecord(
                str(uuid.uuid4()), source_id, event_time, key.data_type,
                json.loads(json.dumps(payload, default=_json_default)),
            ))


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return _aware(value, "datetime").isoformat()
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    # Raw arrays/images are deliberately not copied into compact evidence.
    return repr(value)
