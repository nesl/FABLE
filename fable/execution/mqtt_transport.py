"""MQTT data-plane adapters for distributed FABLE deployments.

MQTT carries compact lifecycle commands, status, predicate matches, and identity
associations. Raw video/audio remains node-local and flows through StreamBus.
"""
from __future__ import annotations

from dataclasses import asdict
import base64
from datetime import datetime
import json
import logging
import os
import queue
import threading
import time
import uuid
import struct
from typing import Callable, Mapping, Protocol

from fable.providers.identity import IdentityAssociation
from fable.providers.data_models import AudioWindow, DetectionFrame, TrackFrame, VideoFrame
from fable.providers.predicate_result import PredicateMatch
from .stream_bus import StreamBus, StreamKey, Subscription

from .command_transport import (
    CommandTransport,
    _key_from_dict,
    _key_to_dict,
    _spec_from_dict,
    _spec_to_dict,
    _status_from_dict,
    _status_to_dict,
)
from .node_agent import CommandResult, NodeAgent, NodeStatus
from .plan_reconciler import ProviderInstanceKey, ProviderInstanceSpec


class MQTTClient(Protocol):
    def subscribe(self, topic: str, callback: Callable[[str, bytes], None], *, qos: int = 1) -> None: ...
    def publish(self, topic: str, payload: bytes, *, qos: int = 1, retain: bool = False) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class PahoMQTTClient:
    """Small Paho wrapper with reconnect-safe subscriptions."""

    def __init__(self, host: str, port: int, client_id: str, *, keepalive: int = 60) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:  # pragma: no cover - optional deployment dependency
            raise RuntimeError("paho-mqtt is required for MQTT transport") from exc
        self._mqtt = mqtt
        self.host, self.port, self.keepalive = host, int(port), int(keepalive)
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            clean_session=False,
            protocol=mqtt.MQTTv311,
        )
        self._callbacks: dict[str, list[Callable[[str, bytes], None]]] = {}
        self._deliveries: queue.Queue[tuple[str, bytes] | None] = queue.Queue()
        self._delivery_thread: threading.Thread | None = None
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def subscribe(self, topic: str, callback: Callable[[str, bytes], None], *, qos: int = 1) -> None:
        first = topic not in self._callbacks
        self._callbacks.setdefault(topic, []).append(callback)
        if first:
            self._client.subscribe(topic, qos=qos)

    def publish(self, topic: str, payload: bytes, *, qos: int = 1, retain: bool = False) -> None:
        info = self._client.publish(topic, payload, qos=qos, retain=retain)
        if info.rc != self._mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed for {topic}: rc={info.rc}")

    def start(self) -> None:
        if self._delivery_thread is None:
            self._delivery_thread = threading.Thread(
                target=self._deliver_messages,
                name=f"mqtt-delivery-{self._client._client_id!r}",
                daemon=True,
            )
            self._delivery_thread.start()
        self._client.connect_async(self.host, self.port, self.keepalive)
        self._client.loop_start()

    def stop(self) -> None:
        self._client.disconnect()
        self._client.loop_stop()
        self._deliveries.put(None)
        if self._delivery_thread is not None:
            self._delivery_thread.join(timeout=2)
            self._delivery_thread = None

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if int(getattr(reason_code, "value", reason_code)) != 0:
            return
        for topic in self._callbacks:
            client.subscribe(topic, qos=1)

    def _on_message(self, client, userdata, message) -> None:
        # Provider startup and replanning can take tens of seconds. Never run
        # those callbacks on Paho's network thread or keepalives will stop.
        self._deliveries.put((message.topic, bytes(message.payload)))

    def _deliver_messages(self) -> None:
        while True:
            delivery = self._deliveries.get()
            if delivery is None:
                return
            topic, payload = delivery
            try:
                self._deliver_message(topic, payload)
            except Exception:
                logging.exception("MQTT callback failed for %s", topic)

    def _deliver_message(self, topic: str, payload: bytes) -> None:
        for topic_filter, callbacks in tuple(self._callbacks.items()):
            if self._mqtt.topic_matches_sub(topic_filter, topic):
                for callback in tuple(callbacks):
                    callback(topic, payload)


def command_topic(node_id: str) -> str:
    return f"fable/v1/command/{node_id}"


def command_reply_topic(client_id: str) -> str:
    return f"fable/v1/reply/{client_id}"


def result_topic() -> str:
    return "fable/v1/result"


def video_source_topic(source_id: str) -> str:
    return f"fable/v1/source/video/{source_id}"


def audio_source_topic(source_id: str) -> str:
    return f"fable/v1/source/audio/{source_id}"


def vision_monitor_topic(source_id: str) -> str:
    return f"fable/v1/monitor/vision/{source_id}"


def event_evidence_topic(source_id: str) -> str:
    return f"fable/v1/evidence/{source_id}"


class MQTTVideoFrameRelay:
    """Opt-in JPEG relay from source-owning buses to remote node buses."""

    def __init__(self, client: MQTTClient, bus: StreamBus, *, node_id: str,
                 local_source_ids: tuple[str, ...] = (), jpeg_quality: int = 85,
                 max_fps: float | None = None) -> None:
        self.client, self.bus, self.node_id = client, bus, str(node_id)
        self.local_source_ids = frozenset(local_source_ids)
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.max_fps = None if max_fps is None else float(max_fps)
        if self.max_fps is not None and self.max_fps <= 0:
            raise ValueError("max_fps must be greater than zero")
        self._last_published_at: dict[str, float] = {}
        self._publish_lock = threading.Lock()
        self._subscription: Subscription | None = None

    def start(self) -> None:
        if self._subscription is not None:
            return
        self._subscription = self.bus.subscribe("video_frame", self._on_local_frame)
        self.client.subscribe("fable/v1/source/video/+", self._on_remote_frame, qos=0)

    def close(self) -> None:
        if self._subscription is not None:
            self.bus.unsubscribe(self._subscription)
            self._subscription = None

    def _on_local_frame(self, key: StreamKey, value: object) -> None:
        if not isinstance(value, VideoFrame) or value.source_id not in self.local_source_ids:
            return
        if self.max_fps is not None:
            now = time.monotonic()
            with self._publish_lock:
                previous = self._last_published_at.get(value.source_id)
                if previous is not None and now - previous < 1.0 / self.max_fps:
                    return
                self._last_published_at[value.source_id] = now
        try:
            import cv2
            ok, encoded = cv2.imencode(
                ".jpg", value.image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
        except Exception as exc:
            raise RuntimeError("OpenCV JPEG encoding failed for MQTT frame relay") from exc
        if not ok:
            raise RuntimeError("OpenCV could not JPEG-encode relayed video frame")
        payload = _json({
            "origin_node": self.node_id,
            "source_id": value.source_id,
            "event_time": value.event_time.isoformat(),
            "frame_id": value.frame_id,
            "jpeg": base64.b64encode(encoded.tobytes()).decode("ascii"),
        })
        self.client.publish(video_source_topic(value.source_id), payload, qos=0)

    def _on_remote_frame(self, topic: str, payload: bytes) -> None:
        row = json.loads(payload)
        if row.get("origin_node") == self.node_id:
            return
        try:
            import cv2
            import numpy as np
            encoded = np.frombuffer(base64.b64decode(row["jpeg"]), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise RuntimeError("could not decode MQTT-relayed video frame") from exc
        if image is None:
            raise RuntimeError("MQTT-relayed video frame contained invalid JPEG")
        source_id = str(row["source_id"])
        self.bus.publish(
            StreamKey("video_frame", (source_id,)),
            VideoFrame(source_id, datetime.fromisoformat(str(row["event_time"])), image, str(row.get("frame_id", ""))),
        )


class MQTTAudioWindowRelay:
    """Relay compact PCM audio windows when a provider is placed remotely."""

    def __init__(self, client: MQTTClient, bus: StreamBus, *, node_id: str,
                 local_source_ids: tuple[str, ...] = ()) -> None:
        self.client, self.bus, self.node_id = client, bus, str(node_id)
        self.local_source_ids = frozenset(local_source_ids)
        self._subscription: Subscription | None = None

    def start(self) -> None:
        if self._subscription is not None:
            return
        self._subscription = self.bus.subscribe("audio_window", self._on_local)
        self.client.subscribe("fable/v1/source/audio/+", self._on_remote, qos=1)

    def close(self) -> None:
        if self._subscription is not None:
            self.bus.unsubscribe(self._subscription)
            self._subscription = None

    def _on_local(self, key: StreamKey, value: object) -> None:
        if not isinstance(value, AudioWindow) or value.source_id not in self.local_source_ids:
            return
        pcm = struct.pack("<" + "h" * len(value.samples), *(
            max(-32768, min(32767, round(sample * 32768))) for sample in value.samples
        ))
        self.client.publish(audio_source_topic(value.source_id), _json({
            "origin_node": self.node_id,
            "source_id": value.source_id,
            "event_time": value.event_time.isoformat(),
            "sample_rate_hz": value.sample_rate_hz,
            "pcm_s16le": base64.b64encode(pcm).decode("ascii"),
        }), qos=1)

    def _on_remote(self, topic: str, payload: bytes) -> None:
        row = json.loads(payload)
        if row.get("origin_node") == self.node_id:
            return
        pcm = base64.b64decode(row["pcm_s16le"])
        ints = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)
        value = AudioWindow(
            str(row["source_id"]), datetime.fromisoformat(str(row["event_time"])),
            tuple(sample / 32768.0 for sample in ints), int(row["sample_rate_hz"]),
        )
        self.bus.publish(StreamKey("audio_window", (value.source_id,)), value)


class MQTTVisionMetadataRelay:
    """Publish local detector/tracker boxes for the live device dashboard."""

    def __init__(self, client: MQTTClient, bus: StreamBus, *, node_id: str) -> None:
        self.client, self.bus, self.node_id = client, bus, str(node_id)
        self._subscriptions: list[Subscription] = []

    def start(self) -> None:
        if self._subscriptions:
            return
        self._subscriptions = [
            self.bus.subscribe("detections", self._on_value),
            self.bus.subscribe("tracks", self._on_value),
        ]

    def close(self) -> None:
        for subscription in self._subscriptions:
            self.bus.unsubscribe(subscription)
        self._subscriptions = []

    def _on_value(self, key: StreamKey, value: object) -> None:
        if isinstance(value, TrackFrame):
            kind = "tracks"
            items = [
                {"id": track.object_id, "label": track.class_name,
                 "confidence": track.confidence, "bbox": asdict(track.bbox)}
                for track in value.tracks
            ]
        elif isinstance(value, DetectionFrame):
            kind = "detections"
            items = [
                {"id": detection.detection_id, "label": detection.class_name,
                 "confidence": detection.confidence, "bbox": asdict(detection.bbox)}
                for detection in value.detections
            ]
        else:
            return
        self.client.publish(vision_monitor_topic(value.source_id), _json({
            "origin_node": self.node_id,
            "source_id": value.source_id,
            "event_time": value.event_time.isoformat(),
            "kind": kind,
            "items": items,
        }), qos=0)


class MQTTEventEvidenceRelay:
    """Upload one deduplicated context JPEG when a visual predicate fires."""

    DEFAULT_PREDICATES = (
        "present", "enters", "exits", "moving", "near", "follows",
        "boards", "disembarks", "transfer_custody", "track_observation",
    )

    def __init__(
        self, client: MQTTClient, bus: StreamBus, *, node_id: str,
        local_source_ids: tuple[str, ...] = (), jpeg_quality: int = 80,
        minimum_interval_seconds: float = 2.0, metrics_path: str | None = None,
        predicates: tuple[str, ...] = DEFAULT_PREDICATES,
        include_track_observations: bool = False,
    ) -> None:
        self.client, self.bus, self.node_id = client, bus, str(node_id)
        self.local_source_ids = frozenset(local_source_ids)
        self.jpeg_quality = int(jpeg_quality)
        self.minimum_interval_seconds = float(minimum_interval_seconds)
        self.metrics_path = None if metrics_path is None else str(metrics_path)
        self.predicates = tuple(predicates)
        self.include_track_observations = bool(include_track_observations)
        self._latest_frames: dict[str, VideoFrame] = {}
        self._last_sent: dict[tuple[str, str, tuple[tuple[str, str], ...]], datetime] = {}
        self._subscriptions: list[Subscription] = []
        self._lock = threading.Lock()
        self.attempted_images = 0
        self.attempted_payload_bytes = 0
        self.skipped_no_frame = 0
        self.suppressed_duplicates = 0

    def start(self) -> None:
        if self._subscriptions:
            return
        self._subscriptions.append(self.bus.subscribe("video_frame", self._on_frame))
        for predicate in self.predicates:
            self._subscriptions.append(
                self.bus.subscribe(f"predicate_match:{predicate}", self._on_match)
            )
        if self.include_track_observations:
            self._subscriptions.append(self.bus.subscribe("tracks", self._on_tracks))
        self._write_metrics()

    def close(self) -> None:
        for subscription in self._subscriptions:
            self.bus.unsubscribe(subscription)
        self._subscriptions = []
        self._write_metrics()

    def _on_frame(self, key: StreamKey, value: object) -> None:
        if isinstance(value, VideoFrame) and value.source_id in self.local_source_ids:
            with self._lock:
                self._latest_frames[value.source_id] = value

    def _on_match(self, key: StreamKey, value: object) -> None:
        if not isinstance(value, PredicateMatch):
            return
        source_id = next(
            (source for source in value.source_ids if source in self.local_source_ids), None
        )
        if source_id is None:
            return
        identity = tuple(sorted(
            (str(name), str(object_id))
            for name, object_id in value.arguments.items()
            if object_id is not None
        ))
        dedupe_key = (value.predicate, source_id, identity)
        with self._lock:
            previous = self._last_sent.get(dedupe_key)
            if previous is not None and (
                value.event_time - previous
            ).total_seconds() < self.minimum_interval_seconds:
                self.suppressed_duplicates += 1
                self._write_metrics_locked()
                return
            frame = self._latest_frames.get(source_id)
            if frame is None:
                self.skipped_no_frame += 1
                self._write_metrics_locked()
                return
            self._last_sent[dedupe_key] = value.event_time
        try:
            import cv2
            ok, encoded = cv2.imencode(
                ".jpg", frame.image,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
        except Exception as exc:
            raise RuntimeError("OpenCV JPEG encoding failed for event evidence") from exc
        if not ok:
            raise RuntimeError("OpenCV could not encode event evidence")
        payload = _json({
            "origin_node": self.node_id,
            "source_id": source_id,
            "predicate": value.predicate,
            "event_time": value.event_time.isoformat(),
            "frame_event_time": frame.event_time.isoformat(),
            "arguments": dict(value.arguments),
            "jpeg": base64.b64encode(encoded.tobytes()).decode("ascii"),
        })
        self.client.publish(event_evidence_topic(source_id), payload, qos=1)
        with self._lock:
            self.attempted_images += 1
            self.attempted_payload_bytes += len(payload)
            self._write_metrics_locked()

    def _on_tracks(self, key: StreamKey, value: object) -> None:
        if not isinstance(value, TrackFrame):
            return
        for track in value.tracks:
            self._on_match(
                key,
                PredicateMatch(
                    "track_observation", value.event_time,
                    arguments={"object": track.object_id},
                    classes={"object": track.class_name},
                    provider_id="b0_track_observation",
                    source_ids=(value.source_id,),
                    confidence=track.confidence,
                ),
            )

    def _write_metrics(self) -> None:
        with self._lock:
            self._write_metrics_locked()

    def _write_metrics_locked(self) -> None:
        if self.metrics_path is None:
            return
        from pathlib import Path
        path = Path(self.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps({
            "attempted_images": self.attempted_images,
            "attempted_payload_bytes": self.attempted_payload_bytes,
            "skipped_no_frame": self.skipped_no_frame,
            "suppressed_duplicates": self.suppressed_duplicates,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)


class MqttCommandTransport(CommandTransport):
    """Controller-side request/reply command transport over MQTT QoS 1."""

    def __init__(self, client: MQTTClient, *, client_id: str, timeout_s: float = 10.0) -> None:
        self.client, self.client_id, self.timeout_s = client, client_id, float(timeout_s)
        self._pending: dict[str, tuple[threading.Event, dict]] = {}
        self._lock = threading.Lock()
        client.subscribe(command_reply_topic(client_id), self._on_reply)

    def start_service(self) -> None:
        self.client.start()

    def close(self) -> None:
        self.client.stop()

    def start(self, spec: ProviderInstanceSpec) -> CommandResult:
        row = self._request(spec.key.node_id, {"action": "start", "spec": _spec_to_dict(spec)})
        return CommandResult(bool(row.get("ok")), str(row.get("message", "")))

    def stop(self, key: ProviderInstanceKey) -> CommandResult:
        row = self._request(key.node_id, {"action": "stop", "key": _key_to_dict(key)})
        return CommandResult(bool(row.get("ok")), str(row.get("message", "")))

    def status(self, node_id: str) -> NodeStatus:
        row = self._request(node_id, {"action": "status"})
        if not row.get("ok"):
            raise RuntimeError(str(row.get("message", "node status failed")))
        return _status_from_dict(row["status"])

    def _request(self, node_id: str, row: dict) -> dict:
        request_id = str(uuid.uuid4())
        event, response = threading.Event(), {}
        with self._lock:
            self._pending[request_id] = (event, response)
        row.update({"request_id": request_id, "reply_to": command_reply_topic(self.client_id)})
        try:
            self.client.publish(command_topic(node_id), _json(row), qos=1)
            if not event.wait(self.timeout_s):
                raise TimeoutError(f"MQTT command to {node_id!r} timed out")
            return response
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def _on_reply(self, topic: str, payload: bytes) -> None:
        row = json.loads(payload)
        with self._lock:
            pending = self._pending.get(str(row.get("request_id", "")))
        if pending is not None:
            pending[1].update(row)
            pending[0].set()


class NodeAgentMQTTService:
    """Agent-side MQTT lifecycle command consumer."""

    def __init__(self, agent: NodeAgent, client: MQTTClient) -> None:
        self.agent, self.client = agent, client
        client.subscribe(command_topic(agent.node.node_id), self._on_command)

    def start(self) -> None:
        self.client.start()

    def close(self) -> None:
        self.client.stop()

    def _on_command(self, topic: str, payload: bytes) -> None:
        try:
            row = json.loads(payload)
            action = row.get("action")
            if action == "start":
                result = self.agent.start(_spec_from_dict(row["spec"]))
                response = {"ok": result.ok, "message": result.message}
            elif action == "stop":
                result = self.agent.stop(_key_from_dict(row["key"]))
                response = {"ok": result.ok, "message": result.message}
            elif action == "status":
                response = {"ok": True, "status": _status_to_dict(self.agent.status())}
            else:
                response = {"ok": False, "message": f"unknown action {action!r}"}
        except Exception as exc:
            response = {"ok": False, "message": str(exc)}
            row = locals().get("row", {})
        response["request_id"] = row.get("request_id")
        reply_to = row.get("reply_to")
        if reply_to:
            self.client.publish(str(reply_to), _json(response), qos=1)


class MqttResultTransport:
    def __init__(self, client: MQTTClient) -> None:
        self.client = client

    def send_predicate_match(self, match: PredicateMatch) -> None:
        self.client.publish(result_topic(), _json({"type": "predicate_match", "value": match.to_dict()}), qos=1)

    def send_identity_association(self, association: IdentityAssociation) -> None:
        self.client.publish(result_topic(), _json({"type": "identity_association", "value": asdict(association)}), qos=1)


class ControllerMQTTResultService:
    def __init__(
        self,
        client: MQTTClient,
        on_predicate_match: Callable[[PredicateMatch], object],
        on_identity_association: Callable[[IdentityAssociation], object],
    ) -> None:
        self.client = client
        self.on_predicate_match = on_predicate_match
        self.on_identity_association = on_identity_association
        client.subscribe(result_topic(), self._on_result)

    def _on_result(self, topic: str, payload: bytes) -> None:
        row = json.loads(payload)
        value = row.get("value", {})
        if row.get("type") == "predicate_match":
            self.on_predicate_match(PredicateMatch.from_dict(value))
        elif row.get("type") == "identity_association":
            self.on_identity_association(IdentityAssociation(**value))


def _json(row: Mapping) -> bytes:
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
