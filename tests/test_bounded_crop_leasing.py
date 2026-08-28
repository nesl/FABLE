from __future__ import annotations

import base64
import json
from datetime import timedelta
from types import SimpleNamespace

from fable.common.examples import BASE_TIME
from providers.vehicle.models import BoundingBox, TrackObservation, TrackSet
from providers.vehicle.service import (
    VehicleMqttService,
    VehicleReplayProcessor,
    VehicleServiceConfig,
)
from providers.vehicle.tracker import RoboflowTrackerAdapter


def _track(local_id: int, confidence: float, *, crop: bool = True) -> TrackObservation:
    source = "camera"
    session = "session"
    reid = {"entity_kind": "vehicle", "vector": [1.0, 0.0]}
    if crop:
        reid["crop_data_url"] = "data:image/jpeg;base64," + base64.b64encode(
            f"crop-{local_id}".encode()
        ).decode()
    return TrackObservation(
        local_track_id=local_id,
        scoped_track_id=f"{source}:{session}:{local_id}",
        source_id=source,
        tracker_session_id=session,
        class_name="car",
        confidence=confidence,
        bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        event_time=BASE_TIME,
        attributes={"reid": reid},
    )


def _processor() -> VehicleReplayProcessor:
    return VehicleReplayProcessor(
        config=VehicleServiceConfig(
            source_id="camera",
            input_topic="/input",
            track_topic="/tracks",
            predicate_topic="/predicates",
            readiness_topic="/ready",
        ),
        tracker=RoboflowTrackerAdapter(algorithm="sort"),
    )


def test_bounded_crop_set_is_quality_ordered_and_limited_to_two() -> None:
    tracks = TrackSet(
        source_id="camera",
        tracker_family="fixture",
        tracker_version="1",
        tracker_session_id="session",
        event_time=BASE_TIME,
        replay_id="run-1",
        tracks=(_track(1, 0.4), _track(2, 0.9), _track(3, 0.7)),
    )
    result = _processor().bounded_crop_set(tracks)
    assert result is not None
    assert result["schema_version"] == "bounded_reid_crop_set.v1"
    assert result["replay_id"] == "run-1"
    assert [row["local_entity_id"] for row in result["records"]] == [
        "camera:session:2",
        "camera:session:3",
    ]


def test_bounded_crop_set_never_falls_back_to_context_or_missing_crop() -> None:
    track = _track(1, 0.9, crop=False).model_copy(
        update={
            "attributes": {
                "reid": {
                    "entity_kind": "vehicle",
                    "vector": [1.0, 0.0],
                    "context_image_data_url": "data:image/jpeg;base64,Y29udGV4dA==",
                }
            }
        }
    )
    tracks = TrackSet(
        source_id="camera",
        tracker_family="fixture",
        tracker_version="1",
        tracker_session_id="session",
        event_time=BASE_TIME,
        tracks=(track,),
    )
    assert _processor().bounded_crop_set(tracks) is None


def test_exact_historical_crop_replay_preserves_scoped_identity() -> None:
    processor = _processor()
    first = TrackSet(
        source_id="camera",
        tracker_family="fixture",
        tracker_version="1",
        tracker_session_id="session",
        event_time=BASE_TIME,
        replay_id="run-1",
        tracks=(_track(1, 0.9),),
    )
    second = first.model_copy(
        update={
            "event_time": BASE_TIME + timedelta(seconds=80),
            "tracks": (_track(2, 0.8),),
        }
    )
    processor.bounded_crop_set(first)
    processor.bounded_crop_set(second)

    replayed = processor.bounded_crop_sets_for_entities(
        ("camera:session:1", "camera:session:2")
    )

    assert len(replayed) == 2
    assert [row["records"][0]["local_entity_id"] for row in replayed] == [
        "camera:session:1",
        "camera:session:2",
    ]
    assert replayed[0]["event_time_interval"] != replayed[1]["event_time_interval"]


class _Client:
    def __init__(self) -> None:
        self.published = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))


class _ServiceProcessor:
    def process(self, _document):
        return SimpleNamespace(model_dump_json=lambda: "{}"), ()

    def descriptor_sets(self, _tracks):
        return ()

    def bounded_crop_set(self, _tracks):
        return {"schema_version": "bounded_reid_crop_set.v1", "records": [{}]}

    def bounded_crop_sets_for_entities(self, entity_ids):
        return tuple(
            {
                "schema_version": "bounded_reid_crop_set.v1",
                "records": [{"local_entity_id": entity_id}],
            }
            for entity_id in entity_ids
        )


def test_service_caches_bounded_crops_without_continuous_reid_publication() -> None:
    client = _Client()
    config = VehicleServiceConfig(
        source_id="camera",
        input_topic="/input",
        track_topic="/tracks",
        predicate_topic="/predicates",
        readiness_topic="/ready",
        bounded_crop_topic="/camera/fable/identity/bounded-crops",
    )
    service = VehicleMqttService(
        config=config,
        processor=_ServiceProcessor(),
        host="mqtt",
        port=1883,
        client=client,
    )
    message = SimpleNamespace(topic="/input", payload=json.dumps({}).encode())
    service._on_message(client, None, message)
    assert not any("bounded-crops" in row[0] for row in client.published)


def test_service_replays_only_source_local_exact_identity_crops() -> None:
    client = _Client()
    config = VehicleServiceConfig(
        source_id="camera",
        input_topic="/input",
        track_topic="/tracks",
        predicate_topic="/predicates",
        readiness_topic="/ready",
        bounded_crop_topic="/camera/fable/identity/bounded-crops",
    )
    service = VehicleMqttService(
        config=config,
        processor=_ServiceProcessor(),
        host="mqtt",
        port=1883,
        client=client,
    )
    message = SimpleNamespace(
        topic="/fable/identity/crop-demands",
        payload=json.dumps(
            {
                "demand_id": "demand-1",
                "local_entity_ids": ["camera:session:1", "other:session:2"],
            }
        ).encode(),
    )

    service._on_message(client, None, message)

    emitted = [row for row in client.published if "bounded-crops" in row[0]]
    assert len(emitted) == 1
    assert "camera:session:1" in emitted[0][1]
    assert "other:session:2" not in emitted[0][1]
