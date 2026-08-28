from __future__ import annotations

import pytest

from fable.distributed.models import ResourceChange
from fable.distributed.resource_journal import (
    ResourceTransitionStage,
    SQLiteResourceTransitionJournal,
)


def change(*, revision: int = 1, run_id: str = "run-1") -> ResourceChange:
    return ResourceChange(
        run_id=run_id,
        condition="E1",
        action="APPLY",
        condition_epoch=revision,
        target_id="x86server",
        resource_kind="COMPUTE",
    )


def test_transition_journal_persists_and_recovers_incomplete_state(tmp_path) -> None:
    path = tmp_path / "journal.sqlite"
    first = SQLiteResourceTransitionJournal(path)
    item = change()
    payload = item.model_dump_json().encode()
    first.begin(item, payload)
    first.advance(str(item.message_id), ResourceTransitionStage.MUTATING)
    first.close()

    reopened = SQLiteResourceTransitionJournal(path)
    assert reopened.payload_for(str(item.message_id)) == payload
    assert reopened.incomplete()[0].stage == ResourceTransitionStage.MUTATING
    reopened.advance(str(item.message_id), ResourceTransitionStage.MUTATED)
    reopened.advance(str(item.message_id), ResourceTransitionStage.REPLANNING)
    reopened.advance(str(item.message_id), ResourceTransitionStage.COMPLETED)
    reopened.advance(str(item.message_id), ResourceTransitionStage.ACKED)
    assert reopened.incomplete() == ()


def test_transition_journal_rejects_stale_scoped_revision(tmp_path) -> None:
    journal = SQLiteResourceTransitionJournal(tmp_path / "journal.sqlite")
    latest = change(revision=4)
    journal.begin(latest, latest.model_dump_json().encode())
    stale = change(revision=3)
    with pytest.raises(ValueError, match="stale resource revision"):
        journal.begin(stale, stale.model_dump_json().encode())
    # Revisions are isolated by run identity.
    other = change(revision=1, run_id="run-2")
    journal.begin(other, other.model_dump_json().encode())


def test_transition_journal_rejects_illegal_stage_jump(tmp_path) -> None:
    journal = SQLiteResourceTransitionJournal(tmp_path / "journal.sqlite")
    item = change()
    journal.begin(item, item.model_dump_json().encode())
    with pytest.raises(ValueError, match="invalid resource transition"):
        journal.advance(str(item.message_id), ResourceTransitionStage.COMPLETED)
