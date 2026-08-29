from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fable.execution import IdentityResolver, LocalRunner
from fable.language import compile_event, parse_event
from fable.providers import BoundingBox, Detection, DetectionFrame, PredicateMatch, Track, TrackFrame


T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def event_enter_exit():
    return compile_event(parse_event({
        "event": "enter_exit",
        "roles": {"VEHICLE": {"class": "vehicle"}},
        "pattern": {
            "seq": [
                {"enters": {"object": "VEHICLE"}},
                {"exits": {"object": "VEHICLE"}},
            ]
        },
    }))


def event_vehicle_person():
    return compile_event(parse_event({
        "event": "vehicle_person",
        "roles": {
            "VEHICLE": {"class": "vehicle"},
            "PERSON": {"class": "person"},
        },
        "pattern": {
            "seq": [
                {"enters": {"object": "VEHICLE"}},
                {"within": {"max": "10s", "pattern": {"enters": {"object": "PERSON"}}}},
            ]
        },
    }))


def match(predicate, when, arguments, *, source="cam1", classes=None):
    return PredicateMatch(
        predicate=predicate,
        event_time=when,
        arguments=arguments,
        provider_id=f"{predicate}_test",
        source_ids=(source,),
        confidence=1.0,
        classes=classes or {},
    )


def track_frame(source, when, tracks=()):
    return TrackFrame(source, when, tuple(tracks))


def car(object_id, source, when, x=0.0):
    return Track(
        object_id=object_id,
        source_id=source,
        class_name="car",
        confidence=0.95,
        bbox=BoundingBox(x, 0, x + 20, 20),
        event_time=when,
    )


def test_identity_resolution_happens_before_ce_matching() -> None:
    runner = LocalRunner(event_enter_exit())

    runner.process_predicate_match(match(
        "enters", T0 + timedelta(seconds=1),
        {"object": "cam_a:track_17"}, source="cam_a", classes={"object": "car"},
    ))
    assert runner.manager.active_instances()[0].bindings == {"VEHICLE": "cam_a:track_17"}

    runner.merge_identity("cam_a:track_17", "cam_b:track_8")
    update = runner.process_predicate_match(match(
        "exits", T0 + timedelta(seconds=2),
        {"object": "cam_b:track_8"}, source="cam_b", classes={"object": "car"},
    ))

    assert update.matches[0].arguments["object"] == "cam_a:track_17"
    assert len(update.completed_instances) == 1
    assert update.completed_instances[0].bindings == {"VEHICLE": "cam_a:track_17"}


def test_late_reid_recanonicalizes_active_bindings_without_merging_instances() -> None:
    resolver = IdentityResolver()
    resolver.register("cam_a:track_17", object_class="car")
    runner = LocalRunner(event_vehicle_person(), identity_resolver=resolver)

    runner.process_predicate_match(match(
        "enters", T0 + timedelta(seconds=1),
        {"object": "cam_b:track_8"}, source="cam_b", classes={"object": "car"},
    ))
    assert runner.manager.active_instances()[0].bindings["VEHICLE"] == "cam_b:track_8"

    runner.merge_identity("cam_a:track_17", "cam_b:track_8")
    assert runner.manager.active_instances()[0].bindings["VEHICLE"] == "cam_a:track_17"


def test_same_physical_vehicle_at_different_times_stays_two_ce_instances() -> None:
    runner = LocalRunner(event_vehicle_person())
    runner.process_predicate_match(match(
        "enters", T0 + timedelta(seconds=1), {"object": "A"}, classes={"object": "car"}
    ))
    runner.process_predicate_match(match(
        "enters", T0 + timedelta(seconds=5), {"object": "B"}, classes={"object": "car"}
    ))

    runner.merge_identity("A", "B")
    removed = runner.deduplicate_ce_instances()

    assert removed == ()
    active = runner.manager.active_instances()
    assert len(active) == 2
    assert {instance.matched_at for instance in active} == {
        T0 + timedelta(seconds=1), T0 + timedelta(seconds=5)
    }
    assert {instance.bindings["VEHICLE"] for instance in active} == {"A"}


def test_ce_instance_dedup_is_separate_and_conservative() -> None:
    runner = LocalRunner(event_vehicle_person())
    when = T0 + timedelta(seconds=1)
    runner.process_predicate_match(match(
        "enters", when, {"object": "A"}, classes={"object": "car"}
    ))
    runner.process_predicate_match(match(
        "enters", when, {"object": "B"}, classes={"object": "car"}
    ))
    assert len(runner.manager.active_instances()) == 2

    runner.merge_identity("A", "B")
    # ReID only canonicalizes object identities.  Both CE candidates still exist.
    assert len(runner.manager.active_instances()) == 2

    removed = runner.deduplicate_ce_instances()
    assert len(removed) == 1
    assert len(runner.manager.active_instances()) == 1


def test_identity_resolver_rejects_incompatible_classes() -> None:
    resolver = IdentityResolver()
    resolver.register("dog", object_class="dog")
    resolver.register("car", object_class="car")
    with pytest.raises(ValueError, match="incompatible observed classes"):
        resolver.merge("dog", "car")


def test_discovery_is_persistent_while_continuation_provider_activates() -> None:
    runner = LocalRunner(event_vehicle_person())
    initial = runner.sync(T0)
    assert {event.provider_id for event in initial if event.action == "activate"} >= {
        "yolo_vehicle_fast_640", "multi_object_tracker", "enters_basic"
    }

    update = runner.process_predicate_match(match(
        "enters", T0 + timedelta(seconds=1), {"object": "V1"}, classes={"object": "car"}
    ))

    # Person continuation upgrades the shared detector to full context.  The
    # persistent discovery predicate is still represented by enters_basic.
    assert "yolo_full_context_960" in runner.running_provider_ids
    assert "yolo_vehicle_fast_640" not in runner.running_provider_ids
    assert "enters_basic" in runner.running_provider_ids
    assert any(e.action == "activate" and e.provider_id == "yolo_full_context_960" for e in update.activation_events)


def test_multiple_candidates_share_one_physical_continuation_pipeline() -> None:
    runner = LocalRunner(event_vehicle_person())
    runner.sync(T0)
    runner.process_predicate_match(match(
        "enters", T0 + timedelta(seconds=1), {"object": "V1"}, classes={"object": "car"}
    ))
    runner.process_predicate_match(match(
        "enters", T0 + timedelta(seconds=2), {"object": "V2"}, classes={"object": "car"}
    ))

    frontier = runner.current_frontier(T0 + timedelta(seconds=3))
    assert len(frontier.continuation) == 2
    assert runner.running_provider_ids.count("yolo_full_context_960") == 1
    assert runner.running_provider_ids.count("multi_object_tracker") == 1


def test_track_frame_pipeline_can_complete_enter_exit_ce() -> None:
    runner = LocalRunner(event_enter_exit())

    # Initialize the persistent enters provider with an empty view.
    runner.process_track_frame(track_frame("cam1", T0))

    enter_update = runner.process_track_frame(track_frame(
        "cam1", T0 + timedelta(seconds=1),
        [car("cam1:track_1", "cam1", T0 + timedelta(seconds=1))],
    ))
    assert any(match.predicate == "enters" for match in enter_update.matches)
    assert runner.manager.active_instances()

    # exits_basic activates only after the candidate exists.  Give it one frame
    # containing the vehicle so its local visibility state is initialized.
    runner.process_track_frame(track_frame(
        "cam1", T0 + timedelta(seconds=1.1),
        [car("cam1:track_1", "cam1", T0 + timedelta(seconds=1.1))],
    ))
    runner.process_track_frame(track_frame("cam1", T0 + timedelta(seconds=2.0)))
    exit_update = runner.process_track_frame(track_frame("cam1", T0 + timedelta(seconds=2.6)))

    assert any(match.predicate == "exits" for match in exit_update.matches)
    assert len(exit_update.completed_instances) == 1
    assert exit_update.completed_instances[0].bindings == {"VEHICLE": "cam1:track_1"}

    # After completion, continuation-only full-context work disappears while
    # vehicle discovery remains logically active.
    assert "yolo_vehicle_fast_640" in runner.running_provider_ids
    assert "yolo_full_context_960" not in runner.running_provider_ids


class FakeDetector:
    def detect(self, image, *, source_id, event_time, frame_id=""):
        detections = ()
        if image == "car":
            detections = (Detection("car", 0.9, BoundingBox(0, 0, 20, 20), "d1"),)
        return DetectionFrame(source_id, event_time, detections, frame_id)


def test_image_path_runs_detector_tracker_predicate_and_ce_manager() -> None:
    event = compile_event(parse_event({
        "event": "vehicle_arrival",
        "roles": {"VEHICLE": {"class": "vehicle"}},
        "pattern": {"enters": {"object": "VEHICLE"}},
    }))
    runner = LocalRunner(
        event,
        detector_factories={"yolo_vehicle_fast_640": FakeDetector},
    )

    runner.process_image("empty", source_id="cam1", event_time=T0)
    update = runner.process_image(
        "car", source_id="cam1", event_time=T0 + timedelta(seconds=1)
    )

    assert any(match.predicate == "enters" for match in update.matches)
    assert len(update.completed_instances) == 1
    assert update.completed_instances[0].matched_source == "cam1"
