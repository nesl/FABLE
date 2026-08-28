from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta

from fable.common.examples import BASE_TIME
from providers.calibration_worker import process_request, worker_capabilities
from providers.vehicle.geometry import RoutePolyline
from providers.vehicle.models import (
    BoundingBox,
    Point2D,
    TrackObservation,
    TrackSet,
    VehicleZone,
    scoped_track_identity,
)


def _track(local_id: int, x: float) -> TrackObservation:
    source = "camera"
    session = "session"
    return TrackObservation(
        local_track_id=local_id,
        scoped_track_id=scoped_track_identity(source, session, local_id),
        source_id=source,
        tracker_session_id=session,
        class_name="car",
        confidence=0.9,
        bbox=BoundingBox(x1=x, y1=0, x2=x + 2, y2=2),
        event_time=BASE_TIME,
        world_point=Point2D(x=x, y=0, coordinate_frame_id="world"),
    )


def _track_at(local_id: int, x: float, second: int) -> TrackObservation:
    return _track(local_id, x).model_copy(
        update={"event_time": BASE_TIME + timedelta(seconds=second)}
    )


def _track_set(second: int, tracks) -> TrackSet:
    return TrackSet(
        source_id="camera",
        tracker_family="fixture",
        tracker_version="1",
        tracker_session_id="session",
        event_time=BASE_TIME + timedelta(seconds=second),
        tracks=tuple(tracks),
    )


def test_pairwise_provider_calibration_worker_round_trip() -> None:
    request = {
        "schema_version": "fable.calibration_worker_request.v1",
        "target": {
            "target_id": "pairwise",
            "provider_id": "pairwise_distance_evaluator",
            "tier": "sensor",
            "input_class": "projected_track_set.v1",
        },
        "invocation_number": 1,
        "fixture": {
            "left": _track(1, 0).model_dump(mode="json"),
            "right": _track(2, 3).model_dump(mode="json"),
            "maximum_distance_m": 5,
            "expected_truth": True,
        },
    }
    completed = subprocess.run(
        (sys.executable, "-m", "providers.calibration_worker"),
        input=json.dumps(request),
        text=True,
        capture_output=True,
        shell=False,
        check=True,
    )
    response = json.loads(completed.stdout)
    assert response["schema_version"] == (
        "fable.calibration_worker_response.v1"
    )
    assert response["successful"] is True
    assert response["quality_score"] == 1.0
    assert response["provider_execution_ms"] >= 0


def test_persistent_worker_serves_multiple_jsonl_requests() -> None:
    left = _track(1, 0)
    right = _track(2, 3)
    request = {
        "schema_version": "fable.calibration_worker_request.v1",
        "target": {
            "target_id": "pairwise",
            "provider_id": "pairwise_distance_evaluator",
            "tier": "sensor",
            "input_class": "projected_track_set.v1",
        },
        "invocation_number": 1,
        "fixture": {
            "left": left.model_dump(mode="json"),
            "right": right.model_dump(mode="json"),
            "maximum_distance_m": 5,
            "expected_truth": True,
        },
    }
    process = subprocess.Popen(
        (sys.executable, "-m", "providers.calibration_worker", "--serve"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    for invocation in (1, 2):
        request["invocation_number"] = invocation
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        response = json.loads(process.stdout.readline())
        assert response["successful"] is True
    process.stdin.close()
    assert process.wait(timeout=2) == 0


def _request(provider_id: str, input_class: str, fixture: dict) -> dict:
    return {
        "schema_version": "fable.calibration_worker_request.v1",
        "target": {
            "target_id": provider_id,
            "provider_id": provider_id,
            "tier": "sensor",
            "input_class": input_class,
        },
        "invocation_number": 1,
        "fixture": fixture,
    }


def test_geometry_calibration_operations_use_real_provider_semantics() -> None:
    motion = process_request(
        _request(
            "motion_state_evaluator",
            "projected_track_set.v1",
            {
                "track_sets": [
                    _track_set(0, (_track_at(1, 0, 0),)).model_dump(mode="json"),
                    _track_set(2, (_track_at(1, 4, 2),)).model_dump(mode="json"),
                ],
                "expected_predicate_id": "MOVING",
            },
        )
    )
    assert motion["successful"] is True

    route = RoutePolyline(
        route_id="route-a",
        coordinate_frame_id="world",
        points=(
            Point2D(x=0, y=0, coordinate_frame_id="world"),
            Point2D(x=10, y=0, coordinate_frame_id="world"),
        ),
    )
    route_result = process_request(
        _request(
            "route_map_matcher",
            "projected_track_set.v1+route_graph.v1",
            {
                "track": _track(1, 3).model_dump(mode="json"),
                "routes": [route.model_dump(mode="json")],
                "expected_route_id": "route-a",
            },
        )
    )
    assert route_result["successful"] is True

    zone = VehicleZone(
        zone_id="zone-a",
        coordinate_frame_id="world",
        polygon=(
            Point2D(x=-1, y=-1, coordinate_frame_id="world"),
            Point2D(x=5, y=-1, coordinate_frame_id="world"),
            Point2D(x=5, y=5, coordinate_frame_id="world"),
            Point2D(x=-1, y=5, coordinate_frame_id="world"),
        ),
    )
    zone_result = process_request(
        _request(
            "zone_membership_evaluator",
            "projected_track_set.v1+route_graph.v1",
            {
                "track": _track(1, 2).model_dump(mode="json"),
                "zone": zone.model_dump(mode="json"),
                "expected_truth": True,
            },
        )
    )
    assert zone_result["successful"] is True


def test_follows_calibration_operation_preserves_state_across_fixture_frames() -> None:
    leader_id = scoped_track_identity("camera", "session", 1)
    follower_id = scoped_track_identity("camera", "session", 2)
    frames = [
        _track_set(
            second,
                (
                    _track_at(1, 20 + second * 4, second),
                    _track_at(2, 10 + second * 4, second).model_copy(
                        update={"attributes": {"longitudinal_gap_to_leader_m": 10.0}}
                    ),
                ),
        ).model_dump(mode="json")
        for second in range(4)
    ]
    result = process_request(
        _request(
            "follows_local_geometry",
            "projected_track_set.v1+pair_trajectory.v1",
            {
                "track_sets": frames,
                "leader_id": leader_id,
                "follower_id": follower_id,
                "minimum_duration_s": 2,
                "expected_truth": True,
            },
        )
    )
    assert result["successful"] is True


def test_worker_capabilities_separate_measurements_from_validation_backends() -> None:
    operations = worker_capabilities()["operations"]
    assert operations["pairwise_distance_evaluator"]["measurement_status"] == (
        "MEASURED_PROVIDER"
    )
    assert operations["follows_local_geometry"]["measurement_status"] == (
        "MEASURED_PROVIDER"
    )
    assert operations["audio_event_classifier"]["measurement_status"] == (
        "IMPLEMENTATION_VALIDATION_ONLY"
    )


def test_transition_pass_and_proximity_workers_preserve_event_time_state() -> None:
    zone = VehicleZone(
        zone_id="door",
        coordinate_frame_id="world",
        polygon=(
            {"x": 0, "y": 0, "coordinate_frame_id": "world"},
            {"x": 5, "y": 0, "coordinate_frame_id": "world"},
            {"x": 5, "y": 5, "coordinate_frame_id": "world"},
            {"x": 0, "y": 5, "coordinate_frame_id": "world"},
        ),
    )
    transition = process_request(
        _request(
            "zone_transition_evaluator",
            "projected_track_set.v1+route_graph.v1",
            {
                "zone": zone.model_dump(mode="json"),
                "tracks": [
                    _track_at(1, -1, 0).model_dump(mode="json"),
                    _track_at(1, 2, 1).model_dump(mode="json"),
                ],
                "expected_predicate_id": "ENTERS",
            },
        )
    )
    assert transition["successful"]

    crossing = process_request(
        _request(
            "pass_reference_evaluator",
            "projected_track_set.v1+route_graph.v1",
            {
                "reference": {
                    "reference_id": "gate",
                    "start": {
                        "x": 0,
                        "y": -5,
                        "coordinate_frame_id": "world",
                    },
                    "end": {
                        "x": 0,
                        "y": 5,
                        "coordinate_frame_id": "world",
                    },
                },
                "tracks": [
                    _track_at(1, -1, 0).model_dump(mode="json"),
                    _track_at(1, 1, 1).model_dump(mode="json"),
                ],
                "expected_truth": True,
            },
        )
    )
    assert crossing["successful"]

    proximity_frames = []
    for second in (0, 1):
        left = _track_at(1, 0, second).model_copy(
            update={"class_name": "person"}
        )
        right = _track_at(2, 1, second).model_copy(
            update={"class_name": "person"}
        )
        proximity_frames.append(
            _track_set(second, (left, right)).model_dump(mode="json")
        )
    proximity = process_request(
        _request(
            "person_proximity_provider",
            "track_set.v1",
            {
                "track_sets": proximity_frames,
                "minimum_duration_seconds": 1,
                "expected_truth": True,
            },
        )
    )
    assert proximity["successful"]
