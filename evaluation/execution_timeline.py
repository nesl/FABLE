"""Compact plan/provider/disturbance timeline for adaptation experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "relative_seconds", "event_kind", "event", "trigger", "condition",
    "target", "added_nodes", "removed_nodes", "selected_nodes",
    "added_providers", "removed_providers", "selected_providers", "reason",
)


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def build_execution_timeline(
    result: dict[str, Any], record_dir: Path
) -> list[dict[str, Any]]:
    plans = _load(record_dir / "plan_decision.jsonl")
    commands = _load(record_dir / "provider_command.jsonl")
    lifecycle = _load(record_dir / "provider_lifecycle.jsonl")
    monotonic = [
        int(row["monotonic_timestamp_ns"])
        for row in (*plans, *commands, *lifecycle)
        if row.get("monotonic_timestamp_ns") is not None
    ]
    origin = min(monotonic) if monotonic else 0

    def seconds(row: dict[str, Any]) -> float:
        value = row.get("monotonic_timestamp_ns")
        return round((int(value) - origin) / 1_000_000_000, 6) if value else 0.0

    events: list[dict[str, Any]] = []
    prior_nodes: set[str] = set()
    prior_providers: set[str] = set()
    for row in sorted(plans, key=seconds):
        nodes = set(map(str, row.get("selected_node_ids") or ()))
        providers = set(map(str, row.get("activated_provider_keys") or ()))
        events.append({
            "relative_seconds": seconds(row), "event_kind": "PLAN",
            "event": "PLAN_SELECTED" if not events else "PLAN_CHANGED",
            "trigger": row.get("replan_trigger") or "", "condition": "", "target": "",
            "added_nodes": ";".join(sorted(nodes - prior_nodes)),
            "removed_nodes": ";".join(sorted(prior_nodes - nodes)),
            "selected_nodes": ";".join(sorted(nodes)),
            "added_providers": ";".join(sorted(providers - prior_providers)),
            "removed_providers": ";".join(sorted(prior_providers - providers)),
            "selected_providers": ";".join(sorted(providers)),
            "reason": row.get("reason") or "",
        })
        prior_nodes, prior_providers = nodes, providers
    for row in commands:
        command = str(row.get("command") or "").upper()
        if command not in {"ACTIVATE", "DEACTIVATE", "RELEASE", "CANCEL"}:
            continue
        provider = f"{row.get('provider_id', '')}@{row.get('node_id', '')}"
        events.append({
            "relative_seconds": seconds(row), "event_kind": "PROVIDER_COMMAND",
            "event": command, "trigger": "", "condition": "", "target": provider,
            "added_nodes": "", "removed_nodes": "", "selected_nodes": "",
            "added_providers": provider if command == "ACTIVATE" else "",
            "removed_providers": provider if command != "ACTIVATE" else "",
            "selected_providers": "", "reason": "",
        })
    seen_lifecycle: set[tuple[str, str]] = set()
    for row in sorted(lifecycle, key=seconds):
        state = str(row.get("lifecycle_event") or "").upper()
        if state not in {"READY", "FAILED", "STOPPED", "TERMINATED"}:
            continue
        provider = f"{row.get('provider_id', '')}@{row.get('node_id', '')}"
        key = (provider, state)
        if key in seen_lifecycle:
            continue
        seen_lifecycle.add(key)
        events.append({
            "relative_seconds": seconds(row), "event_kind": "PROVIDER_LIFECYCLE",
            "event": state, "trigger": "", "condition": "", "target": provider,
            "added_nodes": "", "removed_nodes": "", "selected_nodes": "",
            "added_providers": "", "removed_providers": "", "selected_providers": "",
            "reason": str((row.get("metadata") or {}).get("reason") or ""),
        })
    for row in result.get("disturbance_results") or ():
        response = row.get("response") or {}
        events.append({
            "relative_seconds": row.get("applied_offset_s") or row.get("requested_offset_s") or 0,
            "event_kind": "DISTURBANCE", "event": row.get("action") or "",
            "trigger": row.get("transition_id") or "",
            "condition": row.get("condition") or response.get("measurements", {}).get("profile_id") or "",
            "target": row.get("target") or "", "added_nodes": "", "removed_nodes": "",
            "selected_nodes": "", "added_providers": "", "removed_providers": "",
            "selected_providers": "",
            "reason": response.get("reason") or row.get("reason") or "",
        })
    return sorted(events, key=lambda row: (float(row["relative_seconds"]), row["event_kind"]))


def write_execution_timeline(result: dict[str, Any], record_dir: Path) -> dict[str, Any]:
    rows = build_execution_timeline(result, record_dir)
    jsonl = record_dir / "execution_changes.jsonl"
    csv_path = record_dir / "execution_changes.csv"
    jsonl.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "schema_version": "fable.execution_change_timeline.v1",
        "event_count": len(rows), "jsonl": str(jsonl), "csv": str(csv_path),
    }
