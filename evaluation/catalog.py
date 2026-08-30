"""Typed reader for the immutable complex-event experiment catalog."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    ce_variant: str
    campaign_year: int
    duration_seconds: float
    recording_start: str
    recording_end: str
    relevant_nodes: tuple[str, ...]
    recommended: bool
    quality_status: str


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_experiment_catalog(path: str | Path) -> tuple[ExperimentRecord, ...]:
    """Load label metadata without coupling it to runtime or replay behavior."""
    records: list[ExperimentRecord] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            experiment_id = (row.get("experiment_id") or "").strip()
            if not experiment_id:
                continue
            nodes = tuple(
                part.strip()
                for part in (row.get("relevant_nodes") or "").replace(";", ",").split(",")
                if part.strip()
            )
            records.append(ExperimentRecord(
                experiment_id=experiment_id,
                ce_variant=(row.get("ce_variant") or "").strip(),
                campaign_year=int(row["campaign_year"]),
                duration_seconds=float(row.get("duration_seconds") or 0),
                recording_start=(row.get("recording_start_est") or "").strip(),
                recording_end=(row.get("recording_end_est") or "").strip(),
                relevant_nodes=nodes,
                recommended=_truthy(row.get("recommended_for_use") or ""),
                quality_status=(row.get("quality_status") or "").strip(),
            ))
    return tuple(records)


def group_counts(records: tuple[ExperimentRecord, ...], *, recommended_only: bool = False) -> dict[tuple[int, str], int]:
    counts: dict[tuple[int, str], int] = {}
    for record in records:
        if recommended_only and not record.recommended:
            continue
        key = (record.campaign_year, record.ce_variant)
        counts[key] = counts.get(key, 0) + 1
    return counts
