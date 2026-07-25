from __future__ import annotations

from datetime import timedelta

import pytest

from fable.distributed.outbox import SQLiteOutbox, SQLiteProcessedLedger
from fable.common.time import utc_now


def test_sqlite_outbox_survives_reopen_and_application_ack(tmp_path):
    path = tmp_path / "outbox.sqlite"
    outbox = SQLiteOutbox(path)
    assert outbox.enqueue(
        message_id="m1", topic="fable/test", payload=b'{"v":1}', requires_ack=True
    )
    assert outbox.pending_count == 1
    outbox.close()

    reopened = SQLiteOutbox(path)
    assert reopened.pending_count == 1
    assert reopened.get("m1").attempts == 0
    reopened.mark_attempt("m1")
    assert reopened.get("m1").attempts == 1
    assert reopened.mark_application_acked("m1")
    assert reopened.pending_count == 0


def test_outbox_rejects_message_id_reuse_with_different_bytes(tmp_path):
    outbox = SQLiteOutbox(tmp_path / "outbox.sqlite")
    assert outbox.enqueue(message_id="same", topic="a", payload=b"one")
    assert not outbox.enqueue(message_id="same", topic="a", payload=b"one")
    with pytest.raises(ValueError):
        outbox.enqueue(message_id="same", topic="a", payload=b"two")


def test_processed_ledger_is_restart_persistent_and_content_safe(tmp_path):
    path = tmp_path / "processed.sqlite"
    ledger = SQLiteProcessedLedger(path)
    assert ledger.record(message_id="m1", payload=b"payload", outcome="accepted")
    ledger.close()

    reopened = SQLiteProcessedLedger(path)
    assert reopened.count == 1
    assert reopened.get("m1").outcome == "accepted"
    assert not reopened.record(message_id="m1", payload=b"payload", outcome="ignored")
    with pytest.raises(ValueError):
        reopened.record(message_id="m1", payload=b"different", outcome="bad")
    assert reopened.delete_before(utc_now() + timedelta(seconds=1)) == 1
