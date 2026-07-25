"""MQTT service that adapts the existing replay YOLO stream into vehicle evidence.

The service is intentionally a provider-side component. It publishes typed
tracks and primitive/derived predicate observations; it never updates a FABLE
semantic graph or declares a complete complex event.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
from dataclasses import dataclass
from typing import Any, Callable, Protocol

try:
    import paho.mqtt.client as mqtt
except ImportError:  # optional for pure provider/unit tests
    mqtt = None  # type: ignore[assignment]

from .detector import LegacyReplayYoloAdapter
from .follows import FollowsLocalGeometryEvaluator
from .geometry import (
    MotionStateEvaluator,
    PassReferenceEvaluator,
    ZoneMembershipEvaluator,
    ReferenceLine,
    RouteMapMatcher,
    RoutePolyline,
    ZoneTransitionEvaluator,
)
from .models import PredicateObservation, TrackSet, VehicleZone
from .tracker import RoboflowTrackerAdapter

LOGGER = logging.getLogger(__name__)


class Publisher(Protocol):
    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> Any: ...


@dataclass(frozen=True)
class VehicleServiceConfig:
    source_id: str
    input_topic: str
    track_topic: str
    predicate_topic: str
    readiness_topic: str
    tracker_algorithm: str = "bytetrack"
    leader_track_id: str | None = None


class VehicleReplayProcessor:
    """Pure processing core used by the MQTT service and deterministic tests."""

    def __init__(
        self,
        *,
        config: VehicleServiceConfig,
        tracker: RoboflowTrackerAdapter,
        detector_adapter: LegacyReplayYoloAdapter | None = None,
        references: tuple[ReferenceLine, ...] = (),
        zones: tuple[VehicleZone, ...] = (),
        routes: tuple[RoutePolyline, ...] = (),
    ) -> None:
        self.config = config
        self.tracker = tracker
        self.detector_adapter = detector_adapter or LegacyReplayYoloAdapter()
        self.references = references
        self.zones = zones
        self.routes = routes
        self.route_matcher = RouteMapMatcher()
        self.pass_evaluator = PassReferenceEvaluator()
        self.membership_evaluator = ZoneMembershipEvaluator()
        self.transition_evaluator = ZoneTransitionEvaluator()
        self.motion_evaluator = MotionStateEvaluator()
        self.follows_evaluator = FollowsLocalGeometryEvaluator()
        self._frame_sequence = 0

    def process(self, document: Any) -> tuple[TrackSet, tuple[PredicateObservation, ...]]:
        self._frame_sequence += 1
        detections = self.detector_adapter.parse(
            document,
            source_id=self.config.source_id,
            frame_id=f"{self.config.source_id}:{self._frame_sequence}",
            source_sequence=self._frame_sequence,
        )
        track_set = self.tracker.update(detections)
        if self.routes:
            track_set = track_set.model_copy(
                update={
                    "tracks": tuple(
                        self.route_matcher.match(track, self.routes) for track in track_set.tracks
                    )
                }
            )
        observations: list[PredicateObservation] = []
        for track in track_set.tracks:
            for reference in self.references:
                result = self.pass_evaluator.update(track, reference)
                if result is not None:
                    observations.append(result)
            for zone in self.zones:
                observations.append(self.membership_evaluator.evaluate(track, zone))
                result = self.transition_evaluator.update(track, zone)
                if result is not None:
                    observations.append(result)
        observations.extend(self.motion_evaluator.update(track_set))

        # FOLLOWS demands bind a concrete leader at the semantic frontier.  The
        # adopted replay service cannot be reconfigured per lease, so it emits
        # typed pair candidates for every currently visible leader.  The node
        # agent accepts only the observation whose bound leader and event-time
        # interval match the active demand.  A configured leader narrows this
        # work for debugging or a dedicated deployment.
        leader_ids = (
            (self.config.leader_track_id,)
            if self.config.leader_track_id
            else tuple(track.scoped_track_id for track in track_set.tracks)
        )
        for leader_id in leader_ids:
            observations.extend(
                self.follows_evaluator.update(
                    track_set,
                    leader_id=leader_id,
                )
            )
        return track_set, tuple(observations)


class VehicleMqttService:
    def __init__(
        self,
        *,
        config: VehicleServiceConfig,
        processor: VehicleReplayProcessor,
        host: str,
        port: int,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.processor = processor
        self.host = host
        self.port = port
        if client is None and mqtt is None:
            from .errors import OptionalDependencyError

            raise OptionalDependencyError("paho-mqtt is required to run VehicleMqttService")
        self.client = client or mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"fable-vehicle-{config.source_id}",
            clean_session=False,
        )
        self._stop = threading.Event()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any) -> None:
        if int(reason_code) != 0:
            LOGGER.error("MQTT connection failed: %s", reason_code)
            return
        client.subscribe(self.config.input_topic, qos=0)
        client.publish(
            self.config.readiness_topic,
            json.dumps({"ready": True, "source_id": self.config.source_id}),
            qos=1,
            retain=True,
        )

    def _on_message(self, client: Any, userdata: Any, message: Any) -> None:
        try:
            document = json.loads(message.payload.decode("utf-8"))
            tracks, observations = self.processor.process(document)
            client.publish(self.config.track_topic, tracks.model_dump_json(), qos=0)
            for observation in observations:
                client.publish(
                    self.config.predicate_topic,
                    observation.model_dump_json(),
                    qos=0,
                )
        except Exception:
            LOGGER.exception("vehicle provider failed to process topic=%s", message.topic)

    def run(self) -> None:
        self.client.connect(self.host, self.port, keepalive=60)
        self.client.loop_start()
        try:
            self._stop.wait()
        finally:
            self.client.publish(
                self.config.readiness_topic,
                json.dumps({"ready": False, "source_id": self.config.source_id}),
                qos=1,
                retain=True,
            )
            self.client.loop_stop()
            self.client.disconnect()

    def stop(self) -> None:
        self._stop.set()


def _load_geometry(path: str | None) -> tuple[tuple[ReferenceLine, ...], tuple[VehicleZone, ...], tuple[RoutePolyline, ...]]:
    if not path:
        return (), (), ()
    document = json.loads(open(path, "r", encoding="utf-8").read())
    return (
        tuple(ReferenceLine.model_validate(item) for item in document.get("references", [])),
        tuple(VehicleZone.model_validate(item) for item in document.get("zones", [])),
        tuple(RoutePolyline.model_validate(item) for item in document.get("routes", [])),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mqtt-host", default=os.getenv("MQTT_HOST", "mqtt"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--source-id", default=os.getenv("SOURCE_ID", "orin11"))
    parser.add_argument("--input-topic", default=os.getenv("YOLO_TOPIC"))
    parser.add_argument("--track-topic", default=os.getenv("TRACK_TOPIC"))
    parser.add_argument("--predicate-topic", default=os.getenv("PREDICATE_TOPIC"))
    parser.add_argument("--readiness-topic", default=os.getenv("READINESS_TOPIC"))
    parser.add_argument("--tracker", default=os.getenv("TRACKER_ALGORITHM", "bytetrack"), choices=("bytetrack", "sort"))
    parser.add_argument("--geometry", default=os.getenv("VEHICLE_GEOMETRY_CONFIG"))
    parser.add_argument("--leader-track-id", default=os.getenv("LEADER_TRACK_ID"))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    source_id = args.source_id
    config = VehicleServiceConfig(
        source_id=source_id,
        input_topic=args.input_topic or f"/{source_id}/analytics/yolo/bbox",
        track_topic=args.track_topic or f"/{source_id}/fable/vehicle/tracks",
        predicate_topic=args.predicate_topic or f"/{source_id}/fable/vehicle/predicates",
        readiness_topic=args.readiness_topic or f"/{source_id}/fable/vehicle/ready",
        tracker_algorithm=args.tracker,
        leader_track_id=args.leader_track_id,
    )
    references, zones, routes = _load_geometry(args.geometry)
    tracker = RoboflowTrackerAdapter(algorithm=args.tracker)
    processor = VehicleReplayProcessor(
        config=config,
        tracker=tracker,
        references=references,
        zones=zones,
        routes=routes,
    )
    service = VehicleMqttService(
        config=config,
        processor=processor,
        host=args.mqtt_host,
        port=args.mqtt_port,
    )
    signal.signal(signal.SIGTERM, lambda *_: service.stop())
    signal.signal(signal.SIGINT, lambda *_: service.stop())
    service.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
