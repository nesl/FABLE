from __future__ import annotations
from datetime import datetime
from typing import Mapping
from ..predicate_result import PredicateMatch


def make_match(
    predicate: str,
    event_time: datetime,
    arguments: dict[str, str | float | int | bool | None],
    provider: object,
    source_ids: tuple[str, ...],
    confidence: float,
    *,
    classes: Mapping[str, str] | None = None,
) -> PredicateMatch:
    return PredicateMatch(
        predicate=predicate,
        event_time=event_time,
        arguments=arguments,
        provider_id=str(getattr(provider, "provider_id")),
        provider_version=str(getattr(provider, "provider_version", "1")),
        source_ids=source_ids,
        confidence=max(0.0, min(1.0, float(confidence))),
        classes={} if classes is None else dict(classes),
    )
