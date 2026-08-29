from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fable.providers.predicate_result import PredicateMatch


def test_local_visual_match_keeps_provider_provenance() -> None:
    match = PredicateMatch(
        predicate="near",
        event_time=datetime(2026, 8, 29, 18, 0, tzinfo=timezone.utc),
        arguments={
            "object_a": "camera_3:track_17",
            "object_b": "camera_3:track_42",
        },
        provider_id="near_geometry",
        provider_version="1",
        source_ids=("camera_3",),
        confidence=0.91,
        classes={"object_a": "dog", "object_b": "dog"},
    )

    assert match.predicate == "near"
    assert match.provider_id == "near_geometry"
    assert match.provider_version == "1"
    assert match.source_ids == ("camera_3",)
    assert match.arguments["object_a"] == "camera_3:track_17"
    assert match.classes == {"object_a": "dog", "object_b": "dog"}


def test_audio_match_uses_same_contract() -> None:
    match = PredicateMatch(
        predicate="audio_event",
        event_time=datetime(2026, 8, 29, 18, 1, tzinfo=timezone.utc),
        arguments={"class": "gunshot"},
        provider_id="yamnet_audio_event",
        source_ids=("microphone_2",),
        confidence=0.84,
    )

    assert match.arguments == {"class": "gunshot"}
    assert match.provider_id == "yamnet_audio_event"


def test_cross_sensor_match_can_name_multiple_sources() -> None:
    match = PredicateMatch(
        predicate="follows",
        event_time=datetime(2026, 8, 29, 18, 2, tzinfo=timezone.utc),
        arguments={"leader": "vehicle:17", "follower": "vehicle:22"},
        provider_id="follows_cross_camera",
        source_ids=("camera_1", "camera_2"),
    )

    assert match.source_ids == ("camera_1", "camera_2")


def test_match_round_trips_through_wire_mapping() -> None:
    original = PredicateMatch(
        predicate="enters",
        event_time=datetime(2026, 8, 29, 18, 3, tzinfo=timezone.utc),
        arguments={"object": "camera_4:track_9"},
        provider_id="track_lifecycle",
        source_ids=("camera_4",),
        classes={"object": "car"},
    )

    restored = PredicateMatch.from_dict(original.to_dict())
    assert restored == original


def test_naive_event_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        PredicateMatch(
            predicate="moving",
            event_time=datetime(2026, 8, 29, 18, 4),
            arguments={"object": "track_1"},
            provider_id="motion_geometry",
        )


def test_provider_id_is_required() -> None:
    with pytest.raises(ValueError, match="provider_id"):
        PredicateMatch(
            predicate="moving",
            event_time=datetime(2026, 8, 29, 18, 4, tzinfo=timezone.utc),
            arguments={"object": "track_1"},
        )


def test_nested_provider_payloads_do_not_cross_semantic_boundary() -> None:
    with pytest.raises(ValueError, match="scalar semantic value"):
        PredicateMatch(
            predicate="near",
            event_time=datetime(2026, 8, 29, 18, 5, tzinfo=timezone.utc),
            arguments={"object_a": {"bbox": [0, 0, 1, 1]}},
            provider_id="near_geometry",
        )


def test_duplicate_sources_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        PredicateMatch(
            predicate="near",
            event_time=datetime(2026, 8, 29, 18, 5, tzinfo=timezone.utc),
            arguments={"object_a": "a", "object_b": "b"},
            provider_id="near_geometry",
            source_ids=("camera_1", "camera_1"),
        )
