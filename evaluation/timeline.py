"""Compact provider/plan transition records shared by live experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class PlanTimelineEntry:
    recorded_at: str
    cell_id: str
    reason: str
    started: tuple[str, ...]
    kept: tuple[str, ...]
    stopped: tuple[str, ...]
    selected_steps: tuple[str, ...]

    @classmethod
    def create(
        cls,
        cell_id: str,
        reason: str,
        *,
        started=(),
        kept=(),
        stopped=(),
        selected_steps=(),
    ) -> "PlanTimelineEntry":
        return cls(
            datetime.now(timezone.utc).isoformat(),
            cell_id,
            reason,
            tuple(sorted(started)),
            tuple(sorted(kept)),
            tuple(sorted(stopped)),
            tuple(sorted(selected_steps)),
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
