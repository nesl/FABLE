"""Fake Phase-1 trace loader used by tests and the standalone demo."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from fable.common.time import EventTimeInterval
from fable.semantic import ScriptedResultSpec

FIXTURE_DIR = Path(__file__).resolve().parent / "phase1_fixtures"


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_trace(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def result_specs(trace: dict) -> tuple[ScriptedResultSpec, ...]:
    return tuple(
        ScriptedResultSpec(
            node_key=item["node_key"],
            source_id=item["source_id"],
            event_time_interval=EventTimeInterval(
                start=parse_timestamp(item["start"]),
                end=parse_timestamp(item["end"]),
            ),
            introduced=item.get("introduced", {}),
            validated=item.get("validated", {}),
            truth=item.get("truth", "TRUE"),
            occurrence_id=item.get("occurrence_id"),
        )
        for item in trace["results"]
    )
