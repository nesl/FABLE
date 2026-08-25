from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import threading
from uuid import uuid4

import yaml
import pytest

from fable.distributed.config import ProviderRuntimeResolver
from fable.distributed.models import ReplayOutputAdapter, RuntimeMode
from fable.distributed.node_agent import NodeAgent
from fable.planning.testing import fake_follow_demand, fake_provider_registry
from evaluation.planning_cases import compile_evaluation_planning_case
from fable.planning.artifact_catalog import ArtifactCatalog
from fable.planning.testing import fake_deployment
from fable.common.examples import BASE_TIME
from fable.integrations.replay import build_replay_output_adapter_registry
from fable.orchestration.controller import _requires_source_discovery_fanout
from providers.vehicle.detector import LegacyReplayYoloAdapter
from providers.vehicle.geometry import ReferenceLine, RoutePolyline, TrackLifecycleExitEvaluator
from providers.vehicle.models import (
    BoundingBox,
    Point2D,
    PredicateObservation,
    TrackObservation,
    TrackSet,
    VehicleZone,
)
from providers.vehicle.service import VehicleReplayProcessor, VehicleServiceConfig
from providers.vehicle.tracker import RoboflowTrackerAdapter


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "iobt-minimal-ce-replay"
FIXTURES = ROOT / "providers/tests/phase7_fixtures"


class FakeTracked:
    xyxy = [[0.0, 0.0, 4.0, 2.0]]
    tracker_id = [11]
    confidence = [0.9]
    class_id = [0]


class FakeTracker:
    def update(self, detections, frame=None, timestamp=None):
        return FakeTracked()

    def reset(self):
        return None


def test_uncalibrated_track_lifecycle_emits_pass_only_after_detector_absence() -> None:
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    evaluator = TrackLifecycleExitEvaluator(absence_seconds=0.5)

    def track_set(when: datetime, x: float | None) -> TrackSet:
        tracks = ()
        if x is not None:
            tracks = (
                TrackObservation(
                    local_track_id=1,
                    scoped_track_id="camera:session:1",
                    source_id="camera",
                    tracker_session_id="session",
                    class_name="car",
                    confidence=0.9,
                    bbox=BoundingBox(x1=x, y1=0, x2=x + 20, y2=10),
                    event_time=when,
                    attributes={"matched_detection_id": f"det-{x}"},
                ),
            )
        return TrackSet(
            source_id="camera",
            tracker_family="test",
            tracker_version="1",
            tracker_session_id="session",
            event_time=when,
            tracks=tracks,
        )

    assert evaluator.update(track_set(at, 0)) == ()
    assert evaluator.update(track_set(at + timedelta(seconds=0.2), 25)) == ()
    outputs = evaluator.update(track_set(at + timedelta(seconds=0.8), None))
    assert [item.predicate_id for item in outputs] == ["EXITS", "PASSES"]
    passed = outputs[1]
    assert passed.bindings["reference"] == "camera_fov:camera"
    assert passed.measurements["evidence_mode"] == "image_space_uncalibrated"


def test_uncalibrated_track_lifecycle_rejects_stationary_disappearance() -> None:
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    evaluator = TrackLifecycleExitEvaluator(absence_seconds=0.5)

    track = TrackObservation(
        local_track_id=1,
        scoped_track_id="camera:session:1",
        source_id="camera",
        tracker_session_id="session",
        class_name="car",
        confidence=0.9,
        bbox=BoundingBox(x1=0, y1=0, x2=20, y2=10),
        event_time=at,
        attributes={"matched_detection_id": "det"},
    )
    first = TrackSet(
        source_id="camera", tracker_family="test", tracker_version="1",
        tracker_session_id="session", event_time=at, tracks=(track,),
    )
    empty = first.model_copy(
        update={"event_time": at + timedelta(seconds=1), "tracks": ()}
    )
    assert evaluator.update(first) == ()
    assert [item.predicate_id for item in evaluator.update(empty)] == ["EXITS"]


def test_uncalibrated_track_lifecycle_accepts_half_box_traversal() -> None:
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    evaluator = TrackLifecycleExitEvaluator(absence_seconds=0.5)

    def row(when: datetime, x: float | None) -> TrackSet:
        tracks = () if x is None else (
            TrackObservation(
                local_track_id=1,
                scoped_track_id="camera:session:1",
                source_id="camera",
                tracker_session_id="session",
                class_name="car",
                confidence=0.9,
                bbox=BoundingBox(x1=x, y1=0, x2=x + 20, y2=10),
                event_time=when,
                attributes={"matched_detection_id": f"det-{x}"},
            ),
        )
        return TrackSet(
            source_id="camera", tracker_family="test", tracker_version="1",
            tracker_session_id="session", event_time=when, tracks=tracks,
        )

    evaluator.update(row(at, 0))
    evaluator.update(row(at + timedelta(seconds=0.2), 12))
    assert [item.predicate_id for item in evaluator.update(
        row(at + timedelta(seconds=0.8), None)
    )] == ["EXITS", "PASSES"]


def test_uncalibrated_track_lifecycle_reused_id_starts_new_visit() -> None:
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    evaluator = TrackLifecycleExitEvaluator(absence_seconds=0.5)

    def row(seconds: float, x: float | None) -> TrackSet:
        tracks = () if x is None else (
            TrackObservation(
                local_track_id=1,
                scoped_track_id="camera:session:1",
                source_id="camera",
                tracker_session_id="session",
                class_name="car",
                confidence=0.9,
                bbox=BoundingBox(x1=x, y1=0, x2=x + 20, y2=10),
                event_time=at + timedelta(seconds=seconds),
                attributes={"matched_detection_id": f"det-{seconds}"},
            ),
        )
        return TrackSet(
            source_id="camera", tracker_family="test", tracker_version="1",
            tracker_session_id="session", event_time=at + timedelta(seconds=seconds),
            tracks=tracks,
        )

    for offset in (0.0, 2.0):
        evaluator.update(row(offset, 0))
        evaluator.update(row(offset + 0.2, 12))
        outputs = evaluator.update(row(offset + 0.8, None))
        assert [item.predicate_id for item in outputs] == ["EXITS", "PASSES"]


def test_uncalibrated_track_lifecycle_flushes_at_replay_eof() -> None:
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    evaluator = TrackLifecycleExitEvaluator(absence_seconds=0.5)
    tracks = lambda when, x: TrackSet(
        source_id="camera", tracker_family="test", tracker_version="1",
        tracker_session_id="session", event_time=when,
        tracks=(TrackObservation(
            local_track_id=1, scoped_track_id="camera:session:1",
            source_id="camera", tracker_session_id="session", class_name="car",
            confidence=0.9, bbox=BoundingBox(x1=x, y1=0, x2=x + 20, y2=10),
            event_time=when, attributes={"matched_detection_id": f"det-{x}"},
        ),),
    )
    evaluator.update(tracks(at, 0))
    evaluator.update(tracks(at + timedelta(seconds=0.2), 12))
    assert [item.predicate_id for item in evaluator.flush()] == ["EXITS", "PASSES"]


def test_node_agent_vehicle_adapter_introduces_only_unbound_follower():
    demand = fake_follow_demand()
    interval = demand.event_time_interval
    observation = PredicateObservation(
        occurrence_id="follow_occurrence_1",
        predicate_id="FOLLOWS",
        truth=True,
        confidence=0.88,
        event_time_interval=interval,
        bindings={"leader": "vehicle_17", "follower": "camera_b:session:5"},
        source_ids=("camera_downstream",),
        provider_id="follows_local_geometry",
        provider_version="1",
    )
    agent = object.__new__(NodeAgent)
    agent.node_id = "sensor_b"
    agent.output_adapters = build_replay_output_adapter_registry()
    adapted = agent._adapt_provider_output(
        SimpleNamespace(demand=demand, runtime=SimpleNamespace(output_label_aliases={})),
        ReplayOutputAdapter.VEHICLE_PREDICATE,
        observation.model_dump(mode="json"),
    )
    assert adapted is not None
    assert adapted[2] == {"follower": "camera_b:session:5"}

    mismatch = observation.model_copy(
        update={"bindings": {"leader": "wrong_vehicle", "follower": "camera_b:session:5"}}
    )
    assert agent._adapt_provider_output(
        SimpleNamespace(demand=demand, runtime=SimpleNamespace(output_label_aliases={})),
        ReplayOutputAdapter.VEHICLE_PREDICATE,
        mismatch.model_dump(mode="json"),
    ) is None


def test_node_agent_replays_prefrontier_exit_for_late_bound_lease() -> None:
    """An exit emitted while PASSES is active must survive frontier advance."""

    case = compile_evaluation_planning_case(
        variant="Vehicle rendezvous",
        run_id="run",
        trace_id="late-exit",
        request_id="request-late-exit",
        now=BASE_TIME,
        provider_registry=fake_provider_registry(),
        artifact_catalog=ArtifactCatalog(),
        deployment=fake_deployment(),
    )
    vehicle_id = "camera_mobile:session:7"
    demand = next(
        item
        for item in case.all_task_demands
        if item.semantic_predicate.predicate_id == "EXITS"
    ).model_copy(
        update={
            "bound_roles": {"vehicle": vehicle_id},
            "unbound_roles": (),
            "eligible_source_ids": ("camera_mobile",),
        }
    )
    observation = PredicateObservation(
        occurrence_id="exit-before-frontier",
        predicate_id="EXITS",
        truth=True,
        confidence=0.93,
        event_time_interval=demand.event_time_interval.model_copy(
            update={"end": demand.event_time_interval.start + timedelta(seconds=1)}
        ),
        bindings={"vehicle": vehicle_id, "zone": "camera_fov:sensor_a"},
        source_ids=("camera_mobile",),
        provider_id="track_lifecycle_exit_evaluator",
        provider_version="1",
    )
    agent = object.__new__(NodeAgent)
    agent.node_id = "sensor_a"
    agent.output_adapters = build_replay_output_adapter_registry()
    agent._lock = threading.RLock()
    agent._provider_output_cache = deque()
    agent._provider_output_fingerprints = set()
    agent._provider_output_cache_limit = 32_768
    agent._provider_output_retention = timedelta(minutes=5)
    agent._forwarded_occurrences = set()
    forwarded = []
    agent.forward_result = lambda **kwargs: forwarded.append(kwargs["result"])
    adapter = ReplayOutputAdapter.VEHICLE_PREDICATE
    agent._buffer_provider_output(
        adapter,
        "/sensor_a/fable/vehicle/predicates",
        observation.model_dump(mode="json"),
    )
    # A second occurrence with the same binding is still a distinct temporal
    # fact. Counting graphs must not collapse it merely because the entity is
    # unchanged.
    agent._buffer_provider_output(
        adapter,
        "/sensor_a/fable/vehicle/predicates",
        observation.model_copy(
            update={"occurrence_id": "exit-before-frontier-repeat"}
        ).model_dump(mode="json"),
    )
    assert not forwarded

    command = SimpleNamespace(
        demand=demand,
        runtime=SimpleNamespace(
            provider_id="track_lifecycle_exit_evaluator",
            provider_contract_version=1,
            output_adapter=adapter,
            output_label_aliases={},
        ),
        issued_hypothesis_version=demand.hypothesis_version,
        provider_instance_id="exit-provider-sensor-a",
        attempt_id=uuid4(),
        result_topic="fable/results",
    )
    agent._replay_buffered_outputs(command)

    assert len(forwarded) == 2
    assert forwarded[0].occurrence_id == "exit-before-frontier"
    assert forwarded[1].occurrence_id == "exit-before-frontier-repeat"
    assert forwarded[0].demand_id == demand.demand_id
    # Replaying the same cache for the same demand remains idempotent.
    agent._replay_buffered_outputs(command)
    assert len(forwarded) == 2


def test_node_agent_vehicle_adapter_resolves_authored_camera_reference() -> None:
    demand = fake_follow_demand().model_copy(
        update={"bound_roles": {"leader": "convergence_gate"}}
    )
    observation = PredicateObservation(
        occurrence_id="uncalibrated_camera_pass",
        predicate_id="FOLLOWS",
        truth=True,
        confidence=0.9,
        event_time_interval=demand.event_time_interval,
        bindings={
            "leader": "camera_fov:mobile_archive_4",
            "follower": "mobile_archive_4:session:2",
        },
        source_ids=("mobile_archive_4",),
        provider_id="follows_local_geometry",
        provider_version="1",
        replay_id="current-replay",
    )
    agent = object.__new__(NodeAgent)
    agent.node_id = "mobile_archive_4"
    agent.output_adapters = build_replay_output_adapter_registry()

    adapted = agent._adapt_provider_output(
        SimpleNamespace(demand=demand, runtime=SimpleNamespace(output_label_aliases={})),
        ReplayOutputAdapter.VEHICLE_PREDICATE,
        observation.model_dump(mode="json"),
    )

    assert adapted is not None
    assert adapted[2] == {"follower": "mobile_archive_4:session:2"}
    assert adapted[4] == ("mobile_archive_4",)


def test_symbolic_camera_gate_requires_source_discovery_fanout() -> None:
    demand = fake_follow_demand().model_copy(
        update={
            "bound_roles": {"reference": "convergence_gate"},
            "eligible_source_ids": (
                "mobile_archive_4_camera",
                "mobile_archive_5_camera",
            ),
        }
    )

    assert _requires_source_discovery_fanout(demand)
    concrete = demand.model_copy(
        update={"bound_roles": {"reference": "camera_fov:mobile_archive_4"}}
    )
    assert not _requires_source_discovery_fanout(concrete)


def test_unbound_pair_relation_requires_source_discovery_fanout() -> None:
    demand = fake_follow_demand().model_copy(
        update={
            "bound_roles": {},
            "unbound_roles": ("left", "right"),
            "eligible_source_ids": (
                "mobile_archive_4_camera",
                "mobile_archive_5_camera",
            ),
        }
    )

    assert _requires_source_discovery_fanout(demand)


def test_yolo_adapter_cannot_impersonate_behavioral_predicates() -> None:
    demand = fake_follow_demand()
    agent = object.__new__(NodeAgent)
    agent.node_id = "sensor_b"
    agent.output_adapters = build_replay_output_adapter_registry()
    command = SimpleNamespace(
        demand=demand,
        node_id="sensor_b",
        runtime=SimpleNamespace(output_label_aliases={}),
    )
    payload = [
        {
            "class": "car",
            "conf": 0.95,
            "box": [10, 10, 4, 2],
            "t": demand.event_time_interval.start.isoformat(),
        }
    ]

    assert agent._adapt_provider_output(
        command,
        ReplayOutputAdapter.YOLO_OBJECT_PRESENT,
        payload,
    ) is None

    object_demand = demand.model_copy(
        update={
            "semantic_predicate": demand.semantic_predicate.model_copy(
                update={"predicate_id": "OBJECT_PRESENT"}
            )
        }
    )
    object_command = SimpleNamespace(
        demand=object_demand,
        node_id="sensor_b",
        runtime=SimpleNamespace(output_label_aliases={}),
    )
    assert agent._adapt_provider_output(
        object_command,
        ReplayOutputAdapter.YOLO_OBJECT_PRESENT,
        payload,
    ) is not None


def test_vehicle_replay_processor_converts_existing_yolo_topic_to_typed_predicates():
    lines = (FIXTURES / "legacy_yolo_frames.jsonl").read_text().splitlines()
    line = ReferenceLine(
        reference_id="gate",
        start=Point2D(x=0, y=-10, coordinate_frame_id="replay_world"),
        end=Point2D(x=0, y=10, coordinate_frame_id="replay_world"),
        direction=-1,
    )
    route = RoutePolyline(
        route_id="eastbound",
        coordinate_frame_id="replay_world",
        points=(
            Point2D(x=-20, y=0, coordinate_frame_id="replay_world"),
            Point2D(x=20, y=0, coordinate_frame_id="replay_world"),
        ),
    )
    tracker = RoboflowTrackerAdapter(
        tracker=FakeTracker(),
        detections_factory=lambda _: object(),
        session_id="replay_test",
    )
    processor = VehicleReplayProcessor(
        config=VehicleServiceConfig(
            source_id="dvpg_gq_orin_11",
            input_topic="/in",
            track_topic="/tracks",
            predicate_topic="/predicates",
            readiness_topic="/ready",
        ),
        tracker=tracker,
        detector_adapter=LegacyReplayYoloAdapter(),
        references=(line,),
        routes=(route,),
    )
    first_tracks, first_results = processor.process(json.loads(lines[0]))
    assert first_tracks.tracks[0].route_id == "eastbound"
    assert not any(item.predicate_id == "PASSES" for item in first_results)
    recovered = [
        item for item in first_results
        if item.predicate_id == "VEHICLE_PRESENT_BEFORE"
    ]
    assert len(recovered) == 1
    assert recovered[0].provider_id == "historical_vehicle_interval_matcher"
    assert recovered[0].bindings["vehicle"].startswith("dvpg_gq_orin_11:")
    _, second_results = processor.process(json.loads(lines[1]))
    assert any(item.predicate_id == "PASSES" for item in second_results)


def test_vehicle_processor_emits_replay_scoped_pairwise_candidates() -> None:
    class TwoTrackAdapter:
        def update(self, detections):
            tracks = (
                TrackObservation(
                    local_track_id=index,
                    scoped_track_id=f"camera:pairwise:{index}",
                    source_id="camera",
                    tracker_session_id="pairwise",
                    class_name="car",
                    confidence=0.9,
                    bbox=BoundingBox(x1=x, y1=0, x2=x + 20, y2=10),
                    event_time=detections.event_time,
                )
                for index, x in ((1, 0.0), (2, 22.0))
            )
            return TrackSet(
                source_id="camera",
                tracker_family="test",
                tracker_version="1",
                tracker_session_id="pairwise",
                event_time=detections.event_time,
                replay_id=detections.replay_id,
                tracks=tracks,
            )

    processor = VehicleReplayProcessor(
        config=VehicleServiceConfig(
            source_id="camera",
            input_topic="/in",
            track_topic="/tracks",
            predicate_topic="/predicates",
            readiness_topic="/ready",
        ),
        tracker=TwoTrackAdapter(),
    )
    _, observations = processor.process(
        [
            {
                "class": "car",
                "conf": 0.9,
                "box": [10, 5, 20, 10],
                "t": "2025-08-12T21:01:30Z",
                "replay_id": "replay-current",
            }
        ]
    )

    pairwise = [item for item in observations if item.predicate_id == "DISTANCE_LT"]
    assert len(pairwise) == 1
    assert pairwise[0].truth is True
    assert pairwise[0].replay_id == "replay-current"
    assert pairwise[0].measurements["evidence_mode"] == "image_space_vehicle_width_proxy"
    assert all(item.replay_id == "replay-current" for item in observations)


def test_tracker_resets_when_replay_generation_changes() -> None:
    class CountingTracker(FakeTracker):
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    backend = CountingTracker()
    tracker = RoboflowTrackerAdapter(
        tracker=backend,
        detections_factory=lambda _: object(),
        session_id="initial",
    )
    adapter = LegacyReplayYoloAdapter()
    later = adapter.parse(
        [{"t": "2026-01-01T00:00:10Z", "class": "car", "box": [5, 5, 2, 2], "replay_id": "live"}],
        source_id="camera",
    )
    earlier = adapter.parse(
        [{"t": "2026-01-01T00:00:01Z", "class": "car", "box": [5, 5, 2, 2], "replay_id": "recovery"}],
        source_id="camera",
    )
    first = tracker.update(later)
    second = tracker.update(earlier)
    assert backend.reset_count == 2
    assert first.replay_id == "live"
    assert second.replay_id == "recovery"
    assert first.tracker_session_id != second.tracker_session_id


def test_vehicle_processor_filters_image_geometry_to_its_own_source() -> None:
    def zone(zone_id: str, frame_id: str) -> VehicleZone:
        return VehicleZone(
            zone_id=zone_id,
            coordinate_frame_id=frame_id,
            polygon=(
                Point2D(x=0, y=0, coordinate_frame_id=frame_id),
                Point2D(x=20, y=0, coordinate_frame_id=frame_id),
                Point2D(x=20, y=20, coordinate_frame_id=frame_id),
                Point2D(x=0, y=20, coordinate_frame_id=frame_id),
            ),
        )

    processor = VehicleReplayProcessor(
        config=VehicleServiceConfig(
            source_id="dvpg_gq_orin_7",
            input_topic="/in",
            track_topic="/tracks",
            predicate_topic="/predicates",
            readiness_topic="/ready",
        ),
        tracker=RoboflowTrackerAdapter(
            tracker=FakeTracker(),
            detections_factory=lambda _: object(),
            session_id="geometry_test",
        ),
        zones=(
            zone("orin7_camera_fov", "image:dvpg_gq_orin_7"),
            zone("mobile1_camera_fov", "image:mobile_archive_1"),
        ),
    )

    assert [item.zone_id for item in processor.zones] == ["orin7_camera_fov"]
    _tracks, observations = processor.process(
        [
            {
                "class": "car",
                "conf": 0.9,
                "box": [0, 0, 4, 2],
                "world": [100, 200, 0],
                "t": 0.0,
            }
        ]
    )
    inside = [item for item in observations if item.predicate_id == "INSIDE"]
    assert len(inside) == 1
    assert inside[0].bindings["zone"] == "orin7_camera_fov"


def test_vehicle_processor_isolates_optional_geometry_frame_mismatch() -> None:
    frame_id = "unavailable_map_frame"
    zone = VehicleZone(
        zone_id="optional_map_zone",
        coordinate_frame_id=frame_id,
        polygon=(
            Point2D(x=0, y=0, coordinate_frame_id=frame_id),
            Point2D(x=20, y=0, coordinate_frame_id=frame_id),
            Point2D(x=20, y=20, coordinate_frame_id=frame_id),
            Point2D(x=0, y=20, coordinate_frame_id=frame_id),
        ),
    )
    processor = VehicleReplayProcessor(
        config=VehicleServiceConfig(
            source_id="camera",
            input_topic="/in",
            track_topic="/tracks",
            predicate_topic="/predicates",
            readiness_topic="/ready",
        ),
        tracker=RoboflowTrackerAdapter(
            tracker=FakeTracker(),
            detections_factory=lambda _: object(),
            session_id="optional_geometry_test",
        ),
        zones=(zone,),
    )

    _tracks, observations = processor.process(
        [{"class": "car", "conf": 0.9, "box": [4, 4, 4, 2], "t": 0.0}]
    )

    assert "INSIDE" not in {item.predicate_id for item in observations}
    assert "VEHICLE_PRESENT_BEFORE" in {item.predicate_id for item in observations}


def test_vehicle_processor_does_not_hide_noncompatibility_evaluator_errors() -> None:
    frame_id = "image:camera"
    zone = VehicleZone(
        zone_id="broken_zone",
        coordinate_frame_id=frame_id,
        polygon=(
            Point2D(x=0, y=0, coordinate_frame_id=frame_id),
            Point2D(x=20, y=0, coordinate_frame_id=frame_id),
            Point2D(x=20, y=20, coordinate_frame_id=frame_id),
            Point2D(x=0, y=20, coordinate_frame_id=frame_id),
        ),
    )
    processor = VehicleReplayProcessor(
        config=VehicleServiceConfig(
            source_id="camera",
            input_topic="/in",
            track_topic="/tracks",
            predicate_topic="/predicates",
            readiness_topic="/ready",
        ),
        tracker=RoboflowTrackerAdapter(
            tracker=FakeTracker(),
            detections_factory=lambda _: object(),
            session_id="malformed_geometry_test",
        ),
        zones=(zone,),
    )

    def fail(*_args: object, **_kwargs: object) -> PredicateObservation:
        raise ValueError("malformed zone evaluator state")

    processor.membership_evaluator.evaluate = fail  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="malformed zone evaluator state"):
        processor.process([{"class": "car", "conf": 0.9, "box": [4, 4, 4, 2], "t": 0.0}])


def test_catalog_exposes_real_vehicle_alternatives_without_tracker_snapshot_claims():
    registry = fake_provider_registry()
    demand = fake_follow_demand()
    chains = {item.chain_id for item in registry.candidate_chains(demand)}
    assert chains == {
        "follows_local_tracks",
        "follows_local_from_retained_detections",
        "follows_cross_camera_reid",
    }
    assert "tracker_state_snapshot.v1" not in registry.data_types
    tracker_types = {port.data_type for port in registry.provider("multi_object_tracker").ports}
    assert "track_summary.v1" in tracker_types
    assert "tracker_state_snapshot.v1" not in tracker_types
    assert registry.profile("yolo_vehicle_fast_640", "sensor").execution_ms < registry.profile(
        "yolo_vehicle_balanced_960", "sensor"
    ).execution_ms
    assert registry.profile("yolo_vehicle_fast_640", "sensor").quality_score < registry.profile(
        "yolo_vehicle_balanced_960", "sensor"
    ).quality_score


def test_phase7_replay_overlay_and_runtime_config_use_existing_replay_topics():
    compose = yaml.safe_load((REPLAY / "compose.fable.phase7.yaml").read_text())
    service = compose["services"]["fable-vehicle-orin11"]
    assert service["environment"]["YOLO_TOPIC"] == "/dvpg_gq_orin_11/analytics/yolo/bbox"
    assert service["depends_on"]["yolo-orin11"]["condition"] == "service_started"

    resolver = ProviderRuntimeResolver.from_yaml(REPLAY / "config/fable_provider_runtimes.yaml")
    follows = resolver.resolve(
        node_id="dvpg_gq_orin_11", provider_id="follows_local_geometry"
    )
    assert follows.mode == RuntimeMode.ADOPT_EXISTING
    assert follows.container_name == "fable-vehicle-orin11"
    assert follows.output_adapter == ReplayOutputAdapter.VEHICLE_PREDICATE
    tracker = resolver.resolve(
        node_id="dvpg_gq_orin_11", provider_id="multi_object_tracker"
    )
    assert tracker.output_adapter == ReplayOutputAdapter.NONE


def test_vehicle_replay_processor_emits_all_pair_follows_for_bound_leader_filtering():
    """The adopted replay bridge emits pair evidence; the node agent selects the bound leader."""
    from datetime import timedelta

    from fable.common.examples import BASE_TIME
    from providers.vehicle.follows import FollowsLocalGeometryEvaluator
    from providers.vehicle.models import BoundingBox, TrackObservation, TrackSet

    class PairTracker:
        def __init__(self) -> None:
            self.index = -1

        def update(self, detections):
            self.index += 1
            second = self.index
            timestamp = BASE_TIME + timedelta(seconds=second)

            def row(local_id: int, progress: float) -> TrackObservation:
                scoped = f"camera_a:pair_session:{local_id}"
                return TrackObservation(
                    local_track_id=local_id,
                    scoped_track_id=scoped,
                    source_id="camera_a",
                    tracker_session_id="pair_session",
                    class_name="car",
                    confidence=0.9,
                    bbox=BoundingBox(x1=progress, y1=0, x2=progress + 4, y2=2),
                    event_time=timestamp,
                    route_id="eastbound",
                    route_progress_m=progress,
                )

            return TrackSet(
                source_id="camera_a",
                tracker_family="fake",
                tracker_version="1",
                tracker_session_id="pair_session",
                event_time=timestamp,
                tracks=(
                    row(1, 20 + second),
                    row(2, 10 + second),
                ),
            )

    processor = VehicleReplayProcessor(
        config=VehicleServiceConfig(
            source_id="camera_a",
            input_topic="/in",
            track_topic="/tracks",
            predicate_topic="/predicates",
            readiness_topic="/ready",
        ),
        tracker=PairTracker(),  # type: ignore[arg-type]
    )
    processor.follows_evaluator = FollowsLocalGeometryEvaluator(minimum_duration_s=2)
    payload = [{"class": "car", "conf": 0.9, "box": [0, 0, 4, 2], "t": BASE_TIME.timestamp()}]
    for _ in range(2):
        _, observations = processor.process(payload)
        assert not any(item.predicate_id == "FOLLOWS" for item in observations)
    _, observations = processor.process(payload)
    follows = [item for item in observations if item.predicate_id == "FOLLOWS"]
    assert len(follows) == 1
    assert follows[0].bindings == {
        "leader": "camera_a:pair_session:1",
        "follower": "camera_a:pair_session:2",
    }
