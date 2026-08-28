"""Durable state machine for evaluator-driven resource transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
from pathlib import Path
import sqlite3
import threading

from .models import ResourceChange


class ResourceTransitionStage(StrEnum):
    RECEIVED = "RECEIVED"
    MUTATING = "MUTATING"
    MUTATED = "MUTATED"
    REPLANNING = "REPLANNING"
    COMPLETED = "COMPLETED"
    INFEASIBLE = "INFEASIBLE"
    ERROR = "ERROR"
    ACKED = "ACKED"


_ALLOWED = {
    ResourceTransitionStage.RECEIVED: {ResourceTransitionStage.MUTATING, ResourceTransitionStage.ERROR},
    ResourceTransitionStage.MUTATING: {ResourceTransitionStage.MUTATED, ResourceTransitionStage.ERROR},
    ResourceTransitionStage.MUTATED: {ResourceTransitionStage.REPLANNING, ResourceTransitionStage.ERROR},
    ResourceTransitionStage.REPLANNING: {
        ResourceTransitionStage.COMPLETED,
        ResourceTransitionStage.INFEASIBLE,
        ResourceTransitionStage.ERROR,
    },
    ResourceTransitionStage.COMPLETED: {ResourceTransitionStage.ACKED},
    ResourceTransitionStage.INFEASIBLE: {ResourceTransitionStage.ACKED},
    ResourceTransitionStage.ERROR: {ResourceTransitionStage.ACKED},
    ResourceTransitionStage.ACKED: set(),
}


@dataclass(frozen=True)
class ResourceTransitionRecord:
    message_id: str
    run_id: str
    resource_kind: str
    target_id: str
    revision: int
    stage: ResourceTransitionStage
    reason: str
    updated_at: datetime


class SQLiteResourceTransitionJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS resource_transitions (
                    message_id TEXT PRIMARY KEY,
                    payload_hash TEXT NOT NULL,
                    payload BLOB,
                    run_id TEXT NOT NULL,
                    resource_kind TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(run_id, resource_kind, target_id, revision)
                )
                """
            )
            columns = {
                row["name"] for row in self._conn.execute("PRAGMA table_info(resource_transitions)")
            }
            if "payload" not in columns:
                self._conn.execute("ALTER TABLE resource_transitions ADD COLUMN payload BLOB")

    @staticmethod
    def _scope(change: ResourceChange) -> tuple[str, str, str, int]:
        return (
            change.run_id,
            change.resource_kind.value,
            change.target_id or "GLOBAL",
            change.condition_epoch,
        )

    def begin(self, change: ResourceChange, payload: bytes) -> ResourceTransitionRecord:
        digest = hashlib.sha256(payload).hexdigest()
        scope = self._scope(change)
        message_id = str(change.message_id)
        now = datetime.now(UTC).isoformat()
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT * FROM resource_transitions WHERE message_id=?", (message_id,)
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != digest:
                    raise ValueError("resource message ID reused with different payload")
                return self._record(existing)
            latest = self._conn.execute(
                """SELECT MAX(revision) AS revision FROM resource_transitions
                   WHERE run_id=? AND resource_kind=? AND target_id=?""",
                scope[:3],
            ).fetchone()["revision"]
            if latest is not None and change.condition_epoch <= int(latest):
                raise ValueError(
                    f"stale resource revision {change.condition_epoch}; latest is {latest}"
                )
            self._conn.execute(
                """INSERT INTO resource_transitions
                   (message_id,payload_hash,payload,run_id,resource_kind,target_id,revision,stage,reason,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (message_id, digest, sqlite3.Binary(payload), *scope, ResourceTransitionStage.RECEIVED.value, "", now),
            )
        return self.get(message_id)

    def advance(
        self, message_id: str, stage: ResourceTransitionStage, *, reason: str = ""
    ) -> ResourceTransitionRecord:
        with self._lock, self._conn:
            current = self._conn.execute(
                "SELECT * FROM resource_transitions WHERE message_id=?", (message_id,)
            ).fetchone()
            if current is None:
                raise KeyError(f"unknown resource transition: {message_id}")
            current_stage = ResourceTransitionStage(current["stage"])
            if stage == current_stage:
                return self._record(current)
            if stage not in _ALLOWED[current_stage]:
                raise ValueError(f"invalid resource transition {current_stage} -> {stage}")
            self._conn.execute(
                "UPDATE resource_transitions SET stage=?, reason=?, updated_at=? WHERE message_id=?",
                (stage.value, reason, datetime.now(UTC).isoformat(), message_id),
            )
        return self.get(message_id)

    def get(self, message_id: str) -> ResourceTransitionRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM resource_transitions WHERE message_id=?", (message_id,)
            ).fetchone()
        if row is None:
            raise KeyError(message_id)
        return self._record(row)

    def incomplete(self) -> tuple[ResourceTransitionRecord, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM resource_transitions WHERE stage != ? ORDER BY updated_at",
                (ResourceTransitionStage.ACKED.value,),
            ).fetchall()
        return tuple(self._record(row) for row in rows)

    def payload_for(self, message_id: str) -> bytes:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM resource_transitions WHERE message_id=?", (message_id,)
            ).fetchone()
        if row is None or row["payload"] is None:
            raise KeyError(f"resource transition has no recovery payload: {message_id}")
        return bytes(row["payload"])

    @staticmethod
    def _record(row: sqlite3.Row) -> ResourceTransitionRecord:
        return ResourceTransitionRecord(
            message_id=row["message_id"], run_id=row["run_id"],
            resource_kind=row["resource_kind"], target_id=row["target_id"],
            revision=int(row["revision"]), stage=ResourceTransitionStage(row["stage"]),
            reason=row["reason"], updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
