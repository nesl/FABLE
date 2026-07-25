"""SQLite-backed application outbox and inbound deduplication ledger.

MQTT QoS 1 guarantees broker-level at-least-once delivery, not that the
application parsed and committed a message.  The outbox retains a message until
an explicit FABLE application acknowledgment is received.  The inbound ledger
makes command and result handlers idempotent across duplicates and restarts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import sqlite3
import threading
from typing import Iterable


UTC = timezone.utc


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


@dataclass(frozen=True)
class OutboxItem:
    message_id: str
    topic: str
    payload: bytes
    qos: int
    retain: bool
    requires_ack: bool
    attempts: int
    created_at: datetime
    last_attempt_at: datetime | None
    broker_acked_at: datetime | None
    application_acked_at: datetime | None


class SQLiteOutbox:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._initialize()

    def _initialize(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS outbox (
                    message_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    payload_hash TEXT NOT NULL,
                    qos INTEGER NOT NULL,
                    retain INTEGER NOT NULL,
                    requires_ack INTEGER NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_attempt_at TEXT,
                    broker_acked_at TEXT,
                    application_acked_at TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_outbox_pending ON outbox(application_acked_at, created_at)"
            )

    def enqueue(
        self,
        *,
        message_id: str,
        topic: str,
        payload: bytes,
        qos: int = 1,
        retain: bool = False,
        requires_ack: bool = True,
    ) -> bool:
        """Insert a message once.

        Returns True for a new row and False for an exact duplicate.  Reusing a
        message ID with different content is rejected because it would destroy
        idempotency.
        """
        digest = hashlib.sha256(payload).hexdigest()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT payload_hash, topic, qos, retain, requires_ack FROM outbox WHERE message_id=?",
                (message_id,),
            ).fetchone()
            if row is not None:
                if (
                    row["payload_hash"] != digest
                    or row["topic"] != topic
                    or int(row["qos"]) != int(qos)
                    or bool(row["retain"]) != bool(retain)
                    or bool(row["requires_ack"]) != bool(requires_ack)
                ):
                    raise ValueError(f"message_id {message_id} already exists with different content")
                return False
            self._conn.execute(
                """
                INSERT INTO outbox(
                    message_id, topic, payload, payload_hash, qos, retain,
                    requires_ack, attempts, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    message_id,
                    topic,
                    sqlite3.Binary(payload),
                    digest,
                    int(qos),
                    int(retain),
                    int(requires_ack),
                    _now_iso(),
                ),
            )
            return True

    def pending(
        self,
        *,
        limit: int = 100,
        retry_after: timedelta | None = None,
    ) -> tuple[OutboxItem, ...]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM outbox
                WHERE application_acked_at IS NULL
                ORDER BY created_at, message_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        items = tuple(self._row_to_item(row) for row in rows)
        if retry_after is None:
            return items
        cutoff = datetime.now(tz=UTC) - retry_after
        return tuple(
            item
            for item in items
            if item.last_attempt_at is None or item.last_attempt_at <= cutoff
        )

    def mark_attempt(self, message_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE outbox
                SET attempts=attempts+1, last_attempt_at=?
                WHERE message_id=? AND application_acked_at IS NULL
                """,
                (_now_iso(), message_id),
            )

    def mark_broker_acked(self, message_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE outbox SET broker_acked_at=COALESCE(broker_acked_at, ?) WHERE message_id=?",
                (_now_iso(), message_id),
            )

    def mark_application_acked(self, message_id: str) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE outbox
                SET application_acked_at=COALESCE(application_acked_at, ?)
                WHERE message_id=?
                """,
                (_now_iso(), message_id),
            )
            return cursor.rowcount > 0

    def get(self, message_id: str) -> OutboxItem | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM outbox WHERE message_id=?", (message_id,)
            ).fetchone()
        return None if row is None else self._row_to_item(row)

    def delete_acked_before(self, cutoff: datetime) -> int:
        cutoff = cutoff.astimezone(UTC)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM outbox WHERE application_acked_at IS NOT NULL AND application_acked_at < ?",
                (cutoff.isoformat(),),
            )
            return cursor.rowcount

    @property
    def pending_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM outbox WHERE application_acked_at IS NULL"
            ).fetchone()
        return int(row["n"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> OutboxItem:
        def parse(value: str | None) -> datetime | None:
            return None if value is None else datetime.fromisoformat(value).astimezone(UTC)

        return OutboxItem(
            message_id=row["message_id"],
            topic=row["topic"],
            payload=bytes(row["payload"]),
            qos=int(row["qos"]),
            retain=bool(row["retain"]),
            requires_ack=bool(row["requires_ack"]),
            attempts=int(row["attempts"]),
            created_at=parse(row["created_at"]),  # type: ignore[arg-type]
            last_attempt_at=parse(row["last_attempt_at"]),
            broker_acked_at=parse(row["broker_acked_at"]),
            application_acked_at=parse(row["application_acked_at"]),
        )


@dataclass(frozen=True)
class ProcessedMessage:
    message_id: str
    payload_hash: str
    outcome: str
    response_payload: bytes | None
    processed_at: datetime


class SQLiteProcessedLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._initialize()

    def _initialize(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    payload_hash TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    response_payload BLOB,
                    processed_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processed_at ON processed_messages(processed_at)"
            )

    def get(self, message_id: str) -> ProcessedMessage | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM processed_messages WHERE message_id=?", (message_id,)
            ).fetchone()
        if row is None:
            return None
        return ProcessedMessage(
            message_id=row["message_id"],
            payload_hash=row["payload_hash"],
            outcome=row["outcome"],
            response_payload=None if row["response_payload"] is None else bytes(row["response_payload"]),
            processed_at=datetime.fromisoformat(row["processed_at"]).astimezone(UTC),
        )

    def record(
        self,
        *,
        message_id: str,
        payload: bytes,
        outcome: str,
        response_payload: bytes | None = None,
    ) -> bool:
        digest = hashlib.sha256(payload).hexdigest()
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT payload_hash FROM processed_messages WHERE message_id=?", (message_id,)
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != digest:
                    raise ValueError(
                        f"processed message ID {message_id} was reused with different payload"
                    )
                return False
            self._conn.execute(
                """
                INSERT INTO processed_messages(
                    message_id, payload_hash, outcome, response_payload, processed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    digest,
                    outcome,
                    None if response_payload is None else sqlite3.Binary(response_payload),
                    _now_iso(),
                ),
            )
            return True

    def delete_before(self, cutoff: datetime) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM processed_messages WHERE processed_at < ?",
                (cutoff.astimezone(UTC).isoformat(),),
            )
            return cursor.rowcount

    @property
    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM processed_messages").fetchone()
        return int(row["n"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
