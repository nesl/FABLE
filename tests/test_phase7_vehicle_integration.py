from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from fable.distributed.config import ProviderRuntimeResolver
from fable.distributed.models import ReplayOutputAdapter, RuntimeMode
from fable.distributed.node_agent import NodeAgent
from fable.planning.testing import fake_follow_demand, fake_provider_registry
from providers.vehicle.detector import LegacyReplayYoloAdapter
from providers.vehicle.geometry import ReferenceLine, RoutePolyline
from providers.vehicle.models import Point2D, PredicateObservation
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
    _, second_results = processor.process(json.loads(lines[1]))
    assert any(item.predicate_id == "PASSES" for item in second_results)


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
