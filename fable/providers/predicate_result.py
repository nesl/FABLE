"""Semantic result emitted by predicate implementations.

All perception/model-specific data stays inside ``fable.providers``.  The CE
runtime sees only :class:`PredicateMatch` records.  Keeping this contract in the
provider package makes the boundary explicit: providers produce predicate
results; the event runtime consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, Mapping, TypeAlias

Scalar: TypeAlias = str | int | float | bool | None
_PREDICATE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ARGUMENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class PredicateMatch:
    """One provider-produced semantic predicate result.

    ``provider_id`` intentionally identifies the exact implementation that
    generated the match.  This is useful for provenance, debugging, evaluation,
    and later planning without leaking raw provider artifacts into CE logic.
    """

    predicate: str
    event_time: datetime
    arguments: Mapping[str, Scalar] = field(default_factory=dict)
    provider_id: str = ""
    source_ids: tuple[str, ...] = ()
    confidence: float = 1.0
    provider_version: str | None = None
    classes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.predicate, str) or not _PREDICATE_RE.fullmatch(self.predicate):
            raise ValueError("predicate must be a lowercase identifier")
        if not isinstance(self.event_time, datetime):
            raise ValueError("event_time must be a datetime")
        if self.event_time.tzinfo is None or self.event_time.utcoffset() is None:
            raise ValueError("event_time must be timezone-aware")
        if not isinstance(self.provider_id, str) or not self.provider_id.strip():
            raise ValueError("provider_id must be a non-empty string")
        if self.provider_version is not None and (
            not isinstance(self.provider_version, str) or not self.provider_version.strip()
        ):
            raise ValueError("provider_version must be None or a non-empty string")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a number in [0, 1]")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

        if not isinstance(self.arguments, Mapping):
            raise ValueError("arguments must be a mapping")
        normalized: dict[str, Scalar] = {}
        for name, value in self.arguments.items():
            if not isinstance(name, str) or not _ARGUMENT_RE.fullmatch(name):
                raise ValueError(f"argument name {name!r} must be a lowercase identifier")
            if not _is_scalar(value):
                raise ValueError(
                    f"argument {name!r} must contain a scalar semantic value; "
                    f"got {type(value).__name__}"
                )
            normalized[name] = value
        object.__setattr__(self, "arguments", normalized)

        if not isinstance(self.classes, Mapping):
            raise ValueError("classes must be a mapping")
        normalized_classes: dict[str, str] = {}
        for name, value in self.classes.items():
            if not isinstance(name, str) or not _ARGUMENT_RE.fullmatch(name):
                raise ValueError(f"class argument name {name!r} must be a lowercase identifier")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"class value for {name!r} must be a non-empty string")
            normalized_classes[name] = value
        object.__setattr__(self, "classes", normalized_classes)

        if not isinstance(self.source_ids, tuple):
            object.__setattr__(self, "source_ids", tuple(self.source_ids))
        if any(not isinstance(source, str) or not source.strip() for source in self.source_ids):
            raise ValueError("source_ids must contain only non-empty strings")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source_ids must not contain duplicates")
        object.__setattr__(self, "confidence", float(self.confidence))

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "predicate": self.predicate,
            "event_time": self.event_time.isoformat(),
            "arguments": dict(self.arguments),
            "provider_id": self.provider_id,
            "source_ids": list(self.source_ids),
            "confidence": self.confidence,
            "classes": dict(self.classes),
        }
        if self.provider_version is not None:
            data["provider_version"] = self.provider_version
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PredicateMatch":
        if not isinstance(raw, Mapping):
            raise ValueError("PredicateMatch input must be a mapping")
        allowed = {
            "predicate", "event_time", "arguments", "provider_id",
            "source_ids", "confidence", "provider_version", "classes",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"PredicateMatch has unknown fields {sorted(unknown)}")
        missing = sorted({"predicate", "event_time", "provider_id"} - set(raw))
        if missing:
            raise ValueError(f"PredicateMatch is missing required fields {missing}")

        event_time_raw = raw["event_time"]
        if isinstance(event_time_raw, datetime):
            event_time = event_time_raw
        elif isinstance(event_time_raw, str):
            try:
                event_time = datetime.fromisoformat(event_time_raw)
            except ValueError as exc:
                raise ValueError("event_time must be an ISO-8601 datetime") from exc
        else:
            raise ValueError("event_time must be an ISO-8601 string or datetime")

        source_ids = raw.get("source_ids", ())
        if isinstance(source_ids, str) or not isinstance(source_ids, (list, tuple)):
            raise ValueError("source_ids must be a list/tuple of strings")
        return cls(
            predicate=raw["predicate"],
            event_time=event_time,
            arguments=raw.get("arguments", {}),
            provider_id=raw["provider_id"],
            source_ids=tuple(source_ids),
            confidence=raw.get("confidence", 1.0),
            provider_version=raw.get("provider_version"),
            classes=raw.get("classes", {}),
        )


def _is_scalar(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    return isinstance(value, (int, float)) and not isinstance(value, bool)
