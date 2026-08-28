"""Deterministic media-position sampling for physical replay workers."""

from __future__ import annotations

from typing import Any, Iterable


def deterministic_sample_due(
    frame_number: int,
    *,
    next_sample_frame: float,
) -> bool:
    """Return whether a decoded media frame is the next inference sample."""

    return float(frame_number) + 1e-9 >= next_sample_frame


def deterministic_sample_numbers(
    *,
    frame_count: int,
    source_fps: float,
    maximum_rate_hz: float,
) -> tuple[int, ...]:
    """Return the exact source frames selected by the physical worker."""

    if frame_count < 0 or source_fps <= 0 or maximum_rate_hz <= 0:
        raise ValueError("frame_count must be nonnegative and rates must be positive")
    period = source_fps / maximum_rate_hz
    next_sample = 1.0
    selected = []
    for frame_number in range(1, frame_count + 1):
        if not deterministic_sample_due(
            frame_number, next_sample_frame=next_sample
        ):
            continue
        selected.append(frame_number)
        while next_sample <= frame_number + 1e-9:
            next_sample += period
    return tuple(selected)


def attach_replay_provenance(
    rows: Iterable[dict[str, Any]],
    *,
    replay_id: object,
    scenario: object = None,
) -> list[dict[str, Any]]:
    """Attach the synchronized replay generation to physical provider output.

    Physical frames cross an unframed video proxy, so their run identity comes
    from the separately synchronized replay-control message.  Every derived
    detection must carry that identity before it enters MQTT; downstream
    trackers and predicates intentionally do not guess a missing generation.
    """

    normalized_replay_id = str(replay_id or "").strip()
    if not normalized_replay_id:
        raise ValueError("physical provider output requires a replay_id")
    normalized_scenario = str(scenario or "").strip()
    stamped = []
    for row in rows:
        item = dict(row)
        item["replay_id"] = normalized_replay_id
        if normalized_scenario:
            item["scenario"] = normalized_scenario
        stamped.append(item)
    return stamped
