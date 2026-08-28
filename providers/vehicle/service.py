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
from .errors import ArtifactCompatibilityError, InvalidProviderInput
from .follows import FollowsLocalGeometryEvaluator
from .geometry import (
    MotionStateEvaluator,
    PairwiseDistanceEvaluator,
    PassReferenceEvaluator,
    TrackLifecycleExitEvaluator,
    ZoneMembershipEvaluator,
    ReferenceLine,
    RouteMapMatcher,
    RoutePolyline,
    ZoneTransitionEvaluator,
)
from .models import PredicateObservation, TrackSet, VehicleZone
from .models import occurrence_id
from .replay import HistoricalVehicleIntervalMatcher
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
    bounded_crop_topic: str = ""
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
        expected_image_frame = f"image:{config.source_id}"

        def compatible(frame_id: str) -> bool:
            # World/replay coordinate systems are intentionally shareable.
            # Image geometry is camera-local and must never be applied to a
            # track originating from a different sensor.
            return not frame_id.startswith("image:") or frame_id == expected_image_frame

        self.references = tuple(
            item for item in references if compatible(item.start.coordinate_frame_id)
        )
        self.zones = tuple(item for item in zones if compatible(item.coordinate_frame_id))
        self.routes = tuple(item for item in routes if compatible(item.coordinate_frame_id))
        self.route_matcher = RouteMapMatcher()
        self.pass_evaluator = PassReferenceEvaluator()
        self.lifecycle_evaluator = TrackLifecycleExitEvaluator()
        self.membership_evaluator = ZoneMembershipEvaluator()
        self.transition_evaluator = ZoneTransitionEvaluator()
        self.motion_evaluator = MotionStateEvaluator()
        self.pairwise_distance_evaluator = PairwiseDistanceEvaluator()
        self.follows_evaluator = FollowsLocalGeometryEvaluator()
        self.historical_matcher = HistoricalVehicleIntervalMatcher()
        self._frame_sequence = 0
        self._reported_geometry_incompatibilities: set[tuple[str, str, str]] = set()
        # Bounded, detector-aligned evidence for late identity demands. Keep a
        # single best crop per scoped tracker identity; this is compact typed
        # evidence, not an always-on descriptor/model execution path.
        self._identity_crop_cache: dict[str, dict[str, Any]] = {}
        self._identity_crop_cache_limit = 512
        self._last_replay_id: str | None = None

    def _skip_incompatible_geometry(
        self,
        evaluator: str,
        source_id: str,
        geometry_id: str,
        error: ArtifactCompatibilityError,
    ) -> None:
        """Report an optional geometry incompatibility once per configuration.

        Compatibility failures are local to an optional evaluator.  They must
        not discard tracking, lifecycle, historical-presence, or other valid
        observations from the same frame.  Other exception types deliberately
        remain fatal so malformed inputs and implementation defects are visible.
        """
        key = (evaluator, source_id, geometry_id)
        if key in self._reported_geometry_incompatibilities:
            return
        self._reported_geometry_incompatibilities.add(key)
        LOGGER.warning(
            "skipping incompatible optional geometry evaluator=%s source_id=%s geometry_id=%s: %s",
            evaluator,
            source_id,
            geometry_id,
            error,
        )

    def bounded_crop_set(self, tracks: TrackSet) -> dict[str, Any] | None:
        """Return at most two quality-ranked, detector-aligned vehicle crops."""
        records: list[dict[str, Any]] = []
        for track in sorted(tracks.tracks, key=lambda row: row.confidence, reverse=True):
            reid = track.attributes.get("reid")
            if not isinstance(reid, dict):
                continue
            crop = reid.get("crop_data_url")
            if not isinstance(crop, str) or not crop.startswith("data:image/jpeg;base64,"):
                continue
            record = {
                "local_entity_id": track.scoped_track_id,
                "image_data_url": crop,
                "quality": track.confidence,
            }
            if len(records) < 2:
                records.append(record)
            cached = self._identity_crop_cache.get(track.scoped_track_id)
            if cached is None or float(cached["record"]["quality"]) < track.confidence:
                self._identity_crop_cache[track.scoped_track_id] = {
                    "record": record,
                    "event_time": tracks.event_time,
                    "replay_id": tracks.replay_id,
                }
                while len(self._identity_crop_cache) > self._identity_crop_cache_limit:
                    self._identity_crop_cache.pop(next(iter(self._identity_crop_cache)))
        if not records:
            return None
        interval = {
            "start": tracks.event_time.isoformat(),
            "end": tracks.event_time.isoformat(),
        }
        return {
            "schema_version": "bounded_reid_crop_set.v1",
            "source_id": tracks.source_id,
            "replay_id": tracks.replay_id,
            "event_time_interval": interval,
            "records": records,
        }

    def bounded_crop_sets_for_entities(
        self, entity_ids: tuple[str, ...]
    ) -> tuple[dict[str, Any], ...]:
        """Replay exact cached crops as separate descriptor snapshots.

        Separate messages are intentional: the identity associator compares
        temporally distinct descriptor sets and does not infer that two records
        bundled in one set are candidates for equality.
        """
        outputs: list[dict[str, Any]] = []
        for entity_id in entity_ids:
            cached = self._identity_crop_cache.get(entity_id)
            if cached is None:
                continue
            event_time = cached["event_time"]
            outputs.append(
                {
                    "schema_version": "bounded_reid_crop_set.v1",
                    "source_id": self.config.source_id,
                    "replay_id": cached["replay_id"],
                    "event_time_interval": {
                        "start": event_time.isoformat(),
                        "end": event_time.isoformat(),
                    },
                    "records": [cached["record"]],
                }
            )
        return tuple(outputs)

    def process(self, document: Any) -> tuple[TrackSet, tuple[PredicateObservation, ...]]:
        self._frame_sequence += 1
        detections = self.detector_adapter.parse(
            document,
            source_id=self.config.source_id,
            frame_id=f"{self.config.source_id}:{self._frame_sequence}",
            source_sequence=self._frame_sequence,
        )
        track_set = self.tracker.update(detections)
        self._last_replay_id = track_set.replay_id
        if self.routes:
            matched_tracks = []
            for track in track_set.tracks:
                try:
                    matched_tracks.append(self.route_matcher.match(track, self.routes))
                except ArtifactCompatibilityError as exc:
                    self._skip_incompatible_geometry(
                        "route_map_matcher", track.source_id, "configured_routes", exc
                    )
                    matched_tracks.append(track)
            track_set = track_set.model_copy(update={"tracks": tuple(matched_tracks)})
        observations: list[PredicateObservation] = []
        for track in track_set.tracks:
            for reference in self.references:
                try:
                    result = self.pass_evaluator.update(track, reference)
                except ArtifactCompatibilityError as exc:
                    self._skip_incompatible_geometry(
                        "pass_reference_evaluator", track.source_id, reference.reference_id, exc
                    )
                    continue
                if result is not None:
                    observations.append(result)
            for zone in self.zones:
                try:
                    observations.append(self.membership_evaluator.evaluate(track, zone))
                    result = self.transition_evaluator.update(track, zone)
                except ArtifactCompatibilityError as exc:
                    self._skip_incompatible_geometry(
                        "zone_evaluators", track.source_id, zone.zone_id, exc
                    )
                    continue
                if result is not None:
                    observations.append(result)
        observations.extend(self.motion_evaluator.update(track_set))
        # Emit typed pair candidates continuously; lease-aware node agents
        # forward them only while a DISTANCE_LT demand is active. Keeping this
        # in the shared vehicle worker avoids a catalog/runtime contract where
        # pairwise_distance_evaluator is selectable but never executable.
        for left_index, left in enumerate(track_set.tracks):
            for right in track_set.tracks[left_index + 1 :]:
                result = self.pairwise_distance_evaluator.evaluate(
                    left,
                    right,
                    maximum_distance_m=10.0,
                )
                if result.truth:
                    observations.append(result)
        observations.extend(self.lifecycle_evaluator.update(track_set))
        # When the node agent leases historical_vehicle_interval_matcher, raw
        # retrospective replay traverses the normal YOLO/tracker stream.  Emit
        # source-scoped presence candidates on that same typed predicate topic;
        # the agent admits them only for an active VEHICLE_PRESENT_BEFORE demand.
        for match in self.historical_matcher.match_many((track_set,)):
            bindings = {"vehicle": match.scoped_track_id}
            observations.append(
                PredicateObservation(
                    occurrence_id=occurrence_id(
                        "VEHICLE_PRESENT_BEFORE",
                        bindings,
                        match.event_time_interval,
                        "historical_vehicle_interval_matcher",
                    ),
                    predicate_id="VEHICLE_PRESENT_BEFORE",
                    truth=True,
                    confidence=match.confidence,
                    event_time_interval=match.event_time_interval,
                    bindings=bindings,
                    source_ids=(match.source_id,),
                    provider_id="historical_vehicle_interval_matcher",
                    provider_version="1",
                    supporting_artifact_types=("track_set.v1",),
                )
            )

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
            try:
                observations.extend(
                    self.follows_evaluator.update(
                        track_set,
                        leader_id=leader_id,
                    )
                )
            except InvalidProviderInput:
                # FOLLOWS is optional for uncalibrated sequential-pass graphs.
                # One inapplicable optional evaluator must not discard PASSES,
                # EXITS, and other valid observations from the same frame.
                continue
        # Predicate observations travel on a raw shared topic before the node
        # agent binds them to a demand. Preserve the replay generation here so
        # pending seed watches can reject stale generations without also
        # rejecting every current vehicle seed.
        return track_set, tuple(
            observation.model_copy(update={"replay_id": track_set.replay_id})
            for observation in observations
        )

    def flush_lifecycle(self) -> tuple[PredicateObservation, ...]:
        """Finalize active traversals at the authoritative replay boundary."""
        return tuple(
            observation.model_copy(update={"replay_id": self._last_replay_id})
            for observation in self.lifecycle_evaluator.flush()
        )


class VehicleMqttService:
    def __init__(
        self,
        *,
        config: VehicleServiceConfig,
        processor: VehicleReplayProcessor,
        host: str,
        port: int,
        client_id: str | None = None,
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
            client_id=client_id or f"fable-vehicle-{config.source_id}",
            clean_session=False,
        )
        self._stop = threading.Event()
        self._pending_subscription_mids: set[int] = set()
        self.client.on_connect = self._on_connect
        self.client.on_subscribe = self._on_subscribe
        self.client.on_message = self._on_message
        self._input_messages = 0
        self._predicate_messages = 0

    def _on_connect(self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any) -> None:
        if int(getattr(reason_code, "value", reason_code)) != 0:
            LOGGER.error("MQTT connection failed: %s", reason_code)
            return
        self._pending_subscription_mids.clear()
        for topic, qos in (
            (self.config.input_topic, 0),
            ("/fable/identity/crop-demands", 1),
            (f"/replay/status/zed/{self.config.source_id}", 1),
            (f"/replay/status/mobile/{self.config.source_id}", 1),
        ):
            outcome = client.subscribe(topic, qos=qos)
            if isinstance(outcome, tuple) and len(outcome) > 1:
                self._pending_subscription_mids.add(int(outcome[1]))
        # A real Paho client returns subscription message IDs and readiness is
        # published from _on_subscribe. Lightweight test clients may not.
        if not self._pending_subscription_mids:
            self._publish_ready(client)

    def _on_subscribe(
        self,
        client: Any,
        userdata: Any,
        mid: Any,
        reason_codes: Any,
        properties: Any,
    ) -> None:
        self._pending_subscription_mids.discard(int(mid))
        if not self._pending_subscription_mids:
            self._publish_ready(client)

    def _publish_ready(self, client: Any) -> None:
        client.publish(
            self.config.readiness_topic,
            json.dumps({"ready": True, "source_id": self.config.source_id}),
            qos=1,
            retain=True,
        )

    def _on_message(self, client: Any, userdata: Any, message: Any) -> None:
        try:
            document = json.loads(message.payload.decode("utf-8"))
            if message.topic.startswith("/replay/status/"):
                if document.get("event") == "complete":
                    for observation in self.processor.flush_lifecycle():
                        client.publish(
                            self.config.predicate_topic,
                            observation.model_dump_json(),
                            qos=0,
                        )
                return
            if message.topic == "/fable/identity/crop-demands":
                requested = tuple(
                    str(item)
                    for item in document.get("local_entity_ids", ())
                    if str(item).startswith(f"{self.config.source_id}:")
                )
                crop_sets = self.processor.bounded_crop_sets_for_entities(requested)
                for crop_set in crop_sets:
                    client.publish(
                        self.config.bounded_crop_topic,
                        json.dumps(crop_set),
                        qos=1,
                    )
                LOGGER.info(
                    "bounded identity crop replay demand_id=%s requested=%d emitted=%d missing=%s",
                    document.get("demand_id"),
                    len(requested),
                    len(crop_sets),
                    sorted(
                        entity_id
                        for entity_id in requested
                        if entity_id not in self.processor._identity_crop_cache
                    ),
                )
                return
            tracks, observations = self.processor.process(document)
            self._input_messages += 1
            client.publish(self.config.track_topic, tracks.model_dump_json(), qos=0)
            # Retain a compact, detector-aligned crop for each local identity,
            # but do not run the downstream ReID model continuously.  Exact
            # crops are transferred only when an IdentityComparisonDemand is
            # received on /fable/identity/crop-demands.  Publishing here used
            # to fill the descriptor worker's FIFO with ordinary frame crops,
            # starving the later demand-bound pair that graph progression
            # actually needed.
            self.processor.bounded_crop_set(tracks)
            for observation in observations:
                client.publish(
                    self.config.predicate_topic,
                    observation.model_dump_json(),
                    qos=0,
                )
                self._predicate_messages += 1
            if self._input_messages == 1 or self._input_messages % 100 == 0:
                raw_rows = document if isinstance(document, list) else [document]
                vehicle_rows = [
                    row
                    for row in raw_rows
                    if isinstance(row, dict)
                    and str(row.get("class") or row.get("label"))
                    in {"car", "truck", "bus", "motorcycle"}
                ]
                LOGGER.warning(
                    "vehicle provider progress source=%s inputs=%d vehicle_rows=%d "
                    "max_vehicle_confidence=%.3f tracks=%d predicates=%d",
                    self.config.source_id,
                    self._input_messages,
                    len(vehicle_rows),
                    max(
                        (
                            float(row.get("conf", row.get("confidence", 0.0)))
                            for row in vehicle_rows
                        ),
                        default=0.0,
                    ),
                    len(tracks.tracks),
                    self._predicate_messages,
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
    parser.add_argument("--mqtt-client-id", default=os.getenv("MQTT_CLIENT_ID"))
    parser.add_argument("--input-topic", default=os.getenv("YOLO_TOPIC"))
    parser.add_argument("--track-topic", default=os.getenv("TRACK_TOPIC"))
    parser.add_argument("--predicate-topic", default=os.getenv("PREDICATE_TOPIC"))
    parser.add_argument("--readiness-topic", default=os.getenv("READINESS_TOPIC"))
    parser.add_argument("--tracker", default=os.getenv("TRACKER_ALGORITHM", "bytetrack"), choices=("bytetrack", "sort"))
    parser.add_argument(
        "--tracker-frame-rate",
        type=float,
        default=float(os.getenv("TRACKER_FRAME_RATE", "5.0")),
    )
    parser.add_argument(
        "--track-activation-threshold",
        type=float,
        default=float(os.getenv("VEHICLE_TRACK_ACTIVATION_THRESHOLD", "0.25")),
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=float(os.getenv("VEHICLE_HIGH_CONFIDENCE_THRESHOLD", "0.30")),
    )
    parser.add_argument(
        "--minimum-consecutive-frames",
        type=int,
        default=int(os.getenv("VEHICLE_MINIMUM_CONSECUTIVE_FRAMES", "2")),
    )
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
        bounded_crop_topic=f"/{source_id}/fable/identity/bounded-crops",
        tracker_algorithm=args.tracker,
        leader_track_id=args.leader_track_id,
    )
    references, zones, routes = _load_geometry(args.geometry)
    tracker = RoboflowTrackerAdapter(
        algorithm=args.tracker,
        frame_rate=args.tracker_frame_rate,
        tracker_kwargs=(
            {
                "track_activation_threshold": args.track_activation_threshold,
                "high_conf_det_threshold": args.high_confidence_threshold,
                "minimum_consecutive_frames": args.minimum_consecutive_frames,
            }
            if args.tracker == "bytetrack"
            else {}
        ),
    )
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
        client_id=args.mqtt_client_id,
    )
    signal.signal(signal.SIGTERM, lambda *_: service.stop())
    signal.signal(signal.SIGINT, lambda *_: service.stop())
    service.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
