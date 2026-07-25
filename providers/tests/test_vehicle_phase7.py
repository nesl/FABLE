from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from fable.common.examples import BASE_TIME
from fable.common.time import EventTimeInterval
from fable.planning.provider_registry import ProviderRegistry
from providers.vehicle.association import CrossSensorIdentityAssociator
from providers.vehicle.descriptors import DeterministicDescriptorProvider
from providers.vehicle.detector import LegacyReplayYoloAdapter
from providers.vehicle.errors import ArtifactCompatibilityError, InvalidProviderInput
from providers.vehicle.follows import FollowsLocalGeometryEvaluator, summarize_tracks
from providers.vehicle.geometry import (
    DwellEvaluator,
    MotionStateEvaluator,
    PassReferenceEvaluator,
    ReferenceLine,
    RouteMapMatcher,
    RoutePolyline,
    VehicleZone,
    ZoneTransitionEvaluator,
)
from providers.vehicle.models import (
    BoundingBox,
    Detection,
    DetectionFrame,
    Point2D,
    TrackObservation,
    TrackSet,
    scoped_track_identity,
)
from providers.vehicle.profiling import ProviderProfileRecord, load_profile_records
from providers.vehicle.replay import JsonlDetectionStore, RetrospectiveVehicleExecutor
from providers.vehicle.tracker import DetectionReplayTracker, RoboflowTrackerAdapter


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "phase7_fixtures"


class FakeTracked:
    def __init__(self, x: float, track_id: int = 7) -> None:
        self.xyxy = [[x, 0.0, x + 4.0, 2.0]]
        self.tracker_id = [track_id]
        self.confidence = [0.91]
        self.class_id = [0]


class FakeTracker:
    def __init__(self) -> None:
        self.index = 0
        self.reset_calls = 0

    def update(self, detections, frame=None, timestamp=None):
        self.index += 1
        return FakeTracked(float(self.index - 1))

    def reset(self):
        self.reset_calls += 1


def detection_frame(second: int, *, x: float = 0.0) -> DetectionFrame:
    time = BASE_TIME + timedelta(seconds=second)
    return DetectionFrame(
        source_id="camera_a",
        event_time=time,
        frame_id=f"frame_{second}",
        detector_id="yolo_vehicle_fast_640",
        detector_version="test",
        detections=(
            Detection(
                detection_id=f"det_{second}",
                class_name="car",
                confidence=0.9,
                bbox=BoundingBox(x1=x, y1=0, x2=x + 4, y2=2),
                world_point=Point2D(x=x, y=0, coordinate_frame_id="world"),
            ),
        ),
    )


def track(
    track_id: int,
    second: int,
    *,
    x: float,
    route_progress: float | None = None,
    source_id: str = "camera_a",
    session: str = "session_a",
) -> TrackObservation:
    return TrackObservation(
        local_track_id=track_id,
        scoped_track_id=scoped_track_identity(source_id, session, track_id),
        source_id=source_id,
        tracker_session_id=session,
        class_name="car",
        confidence=0.9,
        bbox=BoundingBox(x1=x, y1=0, x2=x + 4, y2=2),
        event_time=BASE_TIME + timedelta(seconds=second),
        world_point=Point2D(x=x, y=0, coordinate_frame_id="world"),
        route_id="route_a" if route_progress is not None else None,
        route_progress_m=route_progress,
    )


def track_set(second: int, rows: tuple[TrackObservation, ...]) -> TrackSet:
    return TrackSet(
        source_id="camera_a",
        tracker_family="fake",
        tracker_version="1",
        tracker_session_id="session_a",
        event_time=BASE_TIME + timedelta(seconds=second),
        tracks=rows,
    )


def test_legacy_replay_adapter_filters_non_vehicle_and_preserves_world_point():
    payload = json.loads((FIXTURES / "legacy_yolo_frames.jsonl").read_text().splitlines()[0])
    frame = LegacyReplayYoloAdapter().parse(payload, source_id="camera_a")
    assert len(frame.detections) == 1
    assert frame.detections[0].class_name == "car"
    assert frame.detections[0].world_point is not None
    assert frame.detections[0].bbox.center == (10.0, 5.0)


def test_tracker_scopes_local_ids_by_source_and_session():
    adapter = RoboflowTrackerAdapter(
        tracker=FakeTracker(),
        detections_factory=lambda _: object(),
        session_id="session_a",
    )
    output = adapter.update(detection_frame(0))
    assert output.tracks[0].scoped_track_id == "camera_a:session_a:7"
    assert output.tracks[0].attributes["matched_detection_id"] == "det_0"


def test_tracker_rejects_out_of_order_event_time():
    adapter = RoboflowTrackerAdapter(
        tracker=FakeTracker(), detections_factory=lambda _: object(), session_id="session_a"
    )
    adapter.update(detection_frame(2))
    with pytest.raises(InvalidProviderInput):
        adapter.update(detection_frame(1))


def test_retained_detection_replay_reconstructs_without_checkpoint_claim():
    replay = DetectionReplayTracker(
        lambda: RoboflowTrackerAdapter(
            tracker=FakeTracker(), detections_factory=lambda _: object(), session_id="replay_session"
        )
    )
    outputs = replay.replay((detection_frame(0), detection_frame(1)))
    assert outputs[-1].reconstructed_from_detection_replay
    assert outputs[-1].replay_interval is not None
    assert outputs[-1].tracks[0].scoped_track_id.startswith("camera_a:replay_session:")


def test_zone_transition_pass_motion_and_dwell_providers():
    zone = VehicleZone(
        zone_id="zone_a",
        coordinate_frame_id="world",
        polygon=(
            Point2D(x=0, y=-5, coordinate_frame_id="world"),
            Point2D(x=10, y=-5, coordinate_frame_id="world"),
            Point2D(x=10, y=5, coordinate_frame_id="world"),
            Point2D(x=0, y=5, coordinate_frame_id="world"),
        ),
    )
    transition = ZoneTransitionEvaluator()
    assert transition.update(track(1, 0, x=-2), zone) is None
    entered = transition.update(track(1, 1, x=2), zone)
    assert entered and entered.predicate_id == "ENTERS"
    exited = transition.update(track(1, 2, x=12), zone)
    assert exited and exited.predicate_id == "EXITS"

    line = ReferenceLine(
        reference_id="gate",
        start=Point2D(x=0, y=-10, coordinate_frame_id="world"),
        end=Point2D(x=0, y=10, coordinate_frame_id="world"),
        direction=-1,
    )
    passes = PassReferenceEvaluator()
    assert passes.update(track(2, 0, x=-1), line) is None
    crossed = passes.update(track(2, 1, x=1), line)
    assert crossed and crossed.predicate_id == "PASSES"

    motion = MotionStateEvaluator(minimum_window_s=1.0)
    assert motion.update(track_set(0, (track(3, 0, x=0),))) == ()
    observations = motion.update(track_set(2, (track(3, 2, x=4),)))
    assert observations[0].predicate_id == "MOVING"

    dwell = DwellEvaluator()
    assert dwell.update(track(4, 0, x=2), zone, minimum_duration_s=2) is None
    dwelled = dwell.update(track(4, 2, x=2), zone, minimum_duration_s=2)
    assert dwelled and dwelled.predicate_id == "DWELLS"


def test_route_matcher_and_sustained_follows():
    route = RoutePolyline(
        route_id="route_a",
        coordinate_frame_id="world",
        points=(
            Point2D(x=0, y=0, coordinate_frame_id="world"),
            Point2D(x=100, y=0, coordinate_frame_id="world"),
        ),
    )
    matched = RouteMapMatcher().match(track(1, 0, x=20), (route,))
    assert matched.route_id == "route_a"
    assert matched.route_progress_m == pytest.approx(20.0)

    follows = FollowsLocalGeometryEvaluator(maximum_gap_m=15, minimum_duration_s=2)
    leader_id = scoped_track_identity("camera_a", "session_a", 1)
    follower_id = scoped_track_identity("camera_a", "session_a", 2)
    for second in (0, 1):
        assert follows.update(
            track_set(
                second,
                (
                    track(1, second, x=20 + second, route_progress=20 + second),
                    track(2, second, x=10 + second, route_progress=10 + second),
                ),
            ),
            leader_id=leader_id,
            follower_id=follower_id,
        ) == ()
    result = follows.update(
        track_set(
            2,
            (
                track(1, 2, x=22, route_progress=22),
                track(2, 2, x=12, route_progress=12),
            ),
        ),
        leader_id=leader_id,
        follower_id=follower_id,
    )
    assert result and result[0].predicate_id == "FOLLOWS"
    assert result[0].bindings["follower"] == follower_id


def test_track_summary_is_compact_and_contains_scoped_identity():
    summary = summarize_tracks(
        (
            track_set(0, (track(1, 0, x=0),)),
            track_set(1, (track(1, 1, x=1),)),
        )
    )
    assert summary["schema_version"] == "track_summary.v1"
    assert summary["tracks"][0]["scoped_track_id"] == "camera_a:session_a:1"
    assert "tracker_internal_state" not in summary


def test_reid_association_requires_compatible_identity_calibrated_features():
    provider = DeterministicDescriptorProvider(calibrated_for_identity=True)
    left = provider.encode_ids(("vehicle_a",), source_id="camera_a", event_time=BASE_TIME)
    right = provider.encode_ids(("vehicle_a",), source_id="camera_b", event_time=BASE_TIME)
    result = CrossSensorIdentityAssociator(maximum_cosine_distance=0.01).associate(left, right)
    assert len(result.associations) == 1

    incompatible = right.model_copy(update={"model_version": "2"})
    with pytest.raises(ArtifactCompatibilityError):
        CrossSensorIdentityAssociator().associate(left, incompatible)

    general = DeterministicDescriptorProvider(calibrated_for_identity=False).encode_ids(
        ("vehicle_a",), source_id="camera_b", event_time=BASE_TIME
    )
    with pytest.raises(ArtifactCompatibilityError):
        CrossSensorIdentityAssociator().associate(general, general)


def test_jsonl_detection_store_and_retrospective_executor(tmp_path):
    store = JsonlDetectionStore(tmp_path / "detections.jsonl")
    for second in range(3):
        store.append(detection_frame(second))
    interval = EventTimeInterval(
        start=BASE_TIME + timedelta(seconds=1),
        end=BASE_TIME + timedelta(seconds=2),
    )
    rows = store.query(source_id="camera_a", interval=interval)
    assert [row.frame_id for row in rows] == ["frame_1", "frame_2"]
    executor = RetrospectiveVehicleExecutor(
        detection_store=store,
        tracker_factory=lambda: RoboflowTrackerAdapter(
            tracker=FakeTracker(), detections_factory=lambda _: object(), session_id="retro"
        ),
    )
    outputs = executor.replay_tracks(source_id="camera_a", interval=interval)
    assert outputs[-1].event_time == BASE_TIME + timedelta(seconds=2)
    assert outputs[-1].reconstructed_from_detection_replay


def test_profile_records_feed_planner_profiles():
    records = load_profile_records(ROOT / "providers/registry/profiles.reference.json")
    profile = records[0].to_planner_profile()
    assert profile.startup_ms == 420
    assert profile.execution_ms == 32
    registry = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
        profiles_path=ROOT / "providers/registry/profiles.reference.json",
    )
    assert registry.profile("yolo_vehicle_fast_640", "reference_cpu").execution_ms == 32
