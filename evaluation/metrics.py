"""Common outcome records and deterministic summary calculations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class CellOutcome:
    cell_id: str
    event: str
    policy: str
    status: str
    elapsed_seconds: float
    planned_provider_count: int = 0
    estimated_completion_ms: float | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_outcomes(rows: Iterable[CellOutcome]) -> dict[str, object]:
    grouped: dict[str, list[CellOutcome]] = defaultdict(list)
    for row in rows:
        grouped[row.policy].append(row)
    return {
        "schema_version": "fable.evaluation.summary.v1",
        "policies": {
            policy: {
                "cells": len(items),
                "successful": sum(item.status == "SUCCESS" for item in items),
                "failed": sum(item.status != "SUCCESS" for item in items),
                "mean_elapsed_seconds": mean(item.elapsed_seconds for item in items),
                "mean_planned_provider_count": mean(
                    item.planned_provider_count for item in items
                ),
            }
            for policy, items in sorted(grouped.items())
        },
    }
