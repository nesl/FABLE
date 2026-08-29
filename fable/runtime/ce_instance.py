"""Runtime state for one candidate complex-event occurrence.

A :class:`CEInstance` is the simplified replacement for the old FABLE
"hypothesis" object.  It stores only the logical object bindings and the small
amount of pattern-progress / temporal state needed to determine what evidence
could advance that particular candidate.

The semantic identity of a candidate is intentionally human-readable: it was
matched at a particular event time and source, then accumulated a particular
set of role bindings and pattern progress.  ``instance_id`` is only a local
bookkeeping key used by :mod:`fable.runtime.instance_manager`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeAlias


PatternPath: TypeAlias = tuple[int, ...]


@dataclass(slots=True)
class CEInstance:
    """One plausible occurrence of one authored complex event."""

    event_name: str
    matched_at: datetime
    matched_source: str | None = None
    matched_predicate: str | None = None
    matched_path: PatternPath | None = None
    instance_id: str = ""

    # Authored role name -> provider/runtime object identity.
    bindings: dict[str, str] = field(default_factory=dict)

    # Pattern progress keyed by deterministic AST path.
    satisfied_at: dict[PatternPath, datetime] = field(default_factory=dict)
    activated_at: dict[PatternPath, datetime] = field(default_factory=dict)

    # Semantic expiry times only.  These are not scheduler deadlines.
    expires_at: dict[PatternPath, datetime] = field(default_factory=dict)
    failed_paths: dict[PatternPath, str] = field(default_factory=dict)

    # Bookkeeping for ``for:`` sustained predicates.
    sustain_started_at: dict[PatternPath, datetime] = field(default_factory=dict)
    sustain_last_match_at: dict[PatternPath, datetime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_aware(self.matched_at, "matched_at")
        if not isinstance(self.event_name, str) or not self.event_name:
            raise ValueError("event_name must be a non-empty string")
        if self.matched_source is not None and (
            not isinstance(self.matched_source, str) or not self.matched_source
        ):
            raise ValueError("matched_source must be None or a non-empty string")
        if self.matched_predicate is not None and (
            not isinstance(self.matched_predicate, str) or not self.matched_predicate
        ):
            raise ValueError("matched_predicate must be None or a non-empty string")

    @property
    def completed(self) -> bool:
        return () in self.satisfied_at

    @property
    def failed(self) -> bool:
        return () in self.failed_paths

    @property
    def completed_at(self) -> datetime | None:
        return self.satisfied_at.get(())

    def clone(self, *, instance_id: str | None = None) -> "CEInstance":
        """Return an independent branch of this candidate."""

        result = deepcopy(self)
        if instance_id is not None:
            result.instance_id = instance_id
        return result


def _require_aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
