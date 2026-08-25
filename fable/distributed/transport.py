"""MQTT transport adapters and reliable application-level messaging."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import timedelta
from itertools import count
import json
import logging
import queue
import threading
import time
from typing import Any, Protocol

from pydantic import BaseModel

from .codec import decode_model, encode_model
from .models import ApplicationAck
from .outbox import SQLiteOutbox

LOGGER = logging.getLogger(__name__)
MessageCallback = Callable[[str, bytes], None]


class Transport(Protocol):
    def subscribe(self, topic_filter: str, callback: MessageCallback, *, qos: int = 1) -> None: ...

    def publish(self, topic: str, payload: bytes, *, qos: int = 1, retain: bool = False) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


class InMemoryBroker:
    """Small deterministic MQTT-like broker used by Phase-6 tests.

    It supports ``+`` and terminal ``#`` wildcards and can duplicate or drop the
    next publications to exercise at-least-once and interruption behavior.
    """

    def __init__(self) -> None:
        self._subscriptions: list[tuple[str, MessageCallback]] = []
        self._retained: dict[str, bytes] = {}
        self._lock = threading.RLock()
        self.duplicate_next = 0
        self.drop_next = 0
        self.published: list[tuple[str, bytes, int, bool]] = []

    def subscribe(self, topic_filter: str, callback: MessageCallback) -> None:
        with self._lock:
            self._subscriptions.append((topic_filter, callback))
            retained = [
                (topic, payload)
                for topic, payload in self._retained.items()
                if mqtt_topic_matches(topic_filter, topic)
            ]
        for topic, payload in retained:
            callback(topic, payload)

    def publish(self, topic: str, payload: bytes, *, qos: int, retain: bool) -> None:
        with self._lock:
            self.published.append((topic, payload, qos, retain))
            if retain:
                self._retained[topic] = payload
            if self.drop_next > 0:
                self.drop_next -= 1
                return
            callbacks = [
                callback
                for topic_filter, callback in self._subscriptions
                if mqtt_topic_matches(topic_filter, topic)
            ]
            copies = 2 if self.duplicate_next > 0 else 1
            if self.duplicate_next > 0:
                self.duplicate_next -= 1
        for _ in range(copies):
            for callback in callbacks:
                callback(topic, payload)


class InMemoryTransport:
    def __init__(self, broker: InMemoryBroker) -> None:
        self.broker = broker
        self._subscriptions: list[tuple[str, MessageCallback, int]] = []
        self.started = False

    def subscribe(self, topic_filter: str, callback: MessageCallback, *, qos: int = 1) -> None:
        self._subscriptions.append((topic_filter, callback, qos))
        if self.started:
            self.broker.subscribe(topic_filter, callback)

    def publish(self, topic: str, payload: bytes, *, qos: int = 1, retain: bool = False) -> None:
        self.broker.publish(topic, payload, qos=qos, retain=retain)

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        for topic_filter, callback, _qos in self._subscriptions:
            self.broker.subscribe(topic_filter, callback)

    def stop(self) -> None:
        self.started = False


class PahoMQTTTransport:
    """Paho MQTT v3.1.1 transport with a persistent client session."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        client_id: str,
        keepalive: int = 60,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:  # pragma: no cover - exercised only without optional runtime deps
            raise RuntimeError("paho-mqtt is required for PahoMQTTTransport") from exc
        self._mqtt = mqtt
        self.host = host
        self.port = port
        self.keepalive = keepalive
        self.client_id = client_id
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            clean_session=False,
            protocol=mqtt.MQTTv311,
        )
        if username:
            self._client.username_pw_set(username, password=password)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect
        self._subscriptions: dict[str, tuple[int, list[MessageCallback]]] = {}
        self._connected = threading.Event()
        self._started = False
        self._callback_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._control_callback_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._callback_sequence = count()
        self._callback_threads: list[threading.Thread] = []
        self._pending_callback_tokens: set[tuple[str, int]] = set()
        self._callback_tokens: dict[int, tuple[str, int]] = {}
        self._callback_token_lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def subscribe(self, topic_filter: str, callback: MessageCallback, *, qos: int = 1) -> None:
        existing = self._subscriptions.get(topic_filter)
        if existing is None:
            self._subscriptions[topic_filter] = (qos, [callback])
            if self.connected:
                self._client.subscribe(topic_filter, qos=qos)
        else:
            existing[1].append(callback)
            # MQTT retained messages are delivered on SUBSCRIBE, not when a
            # local callback is appended. Logical provider instances may share
            # one readiness filter while being created at different graph
            # checkpoints. Re-subscribe so a newly registered callback also
            # receives the worker's retained readiness state.
            if self.connected:
                self._client.subscribe(topic_filter, qos=existing[0])

    def publish(self, topic: str, payload: bytes, *, qos: int = 1, retain: bool = False) -> None:
        info = self._client.publish(topic, payload=payload, qos=qos, retain=retain)
        if info.rc != self._mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed for {topic}: rc={info.rc}")

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if not self._callback_threads:
            for name, callback_queue in (
                ("control", self._control_callback_queue),
                ("evidence", self._callback_queue),
            ):
                worker = threading.Thread(
                    target=self._run_callbacks,
                    args=(callback_queue,),
                    name=f"mqtt-{name}-{self.client_id}",
                    daemon=True,
                )
                worker.start()
                self._callback_threads.append(worker)
        self._client.connect_async(self.host, self.port, keepalive=self.keepalive)
        self._client.loop_start()

    def wait_connected(self, timeout: float = 10.0) -> bool:
        return self._connected.wait(timeout)

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            self._client.disconnect()
        finally:
            self._client.loop_stop()
            self._connected.clear()
            for index, callback_queue in enumerate(
                (self._control_callback_queue, self._callback_queue)
            ):
                callback_queue.put((-100, next(self._callback_sequence), None, "", b""))
            for worker in self._callback_threads:
                worker.join(timeout=2.0)
            self._callback_threads.clear()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        # Paho 2.x supplies a ReasonCode object which deliberately does not
        # implement ``int()``; Paho 1.x supplied a plain integer.
        code = getattr(reason_code, "value", reason_code)
        if int(code) != 0:
            LOGGER.error("MQTT connection failed client=%s rc=%s", self.client_id, reason_code)
            return
        self._connected.set()
        for topic_filter, (qos, _callbacks) in self._subscriptions.items():
            client.subscribe(topic_filter, qos=qos)
        LOGGER.info("MQTT connected client=%s host=%s:%s", self.client_id, self.host, self.port)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None) -> None:
        self._connected.clear()
        LOGGER.warning("MQTT disconnected client=%s rc=%s", self.client_id, reason_code)

    def _on_message(self, client, userdata, message) -> None:
        payload = bytes(message.payload)
        callbacks: list[MessageCallback] = []
        for topic_filter, (_qos, registered) in self._subscriptions.items():
            if mqtt_topic_matches(topic_filter, message.topic):
                callbacks.extend(registered)
        for callback in callbacks:
            callback_queue = (
                getattr(self, "_control_callback_queue", self._callback_queue)
                if self._is_control_topic(message.topic)
                else self._callback_queue
            )
            sequence = next(self._callback_sequence)
            token = self._pending_result_token(message.topic, payload, callback)
            if token is not None and hasattr(self, "_callback_token_lock"):
                with self._callback_token_lock:
                    if token in self._pending_callback_tokens:
                        # QoS/application retries can arrive while the first
                        # copy is still waiting for durable processing.  One
                        # callback is sufficient: its application ACK clears
                        # every retransmission of the same message ID.
                        continue
                    self._pending_callback_tokens.add(token)
                    self._callback_tokens[sequence] = token
            callback_queue.put(
                (
                    self._message_priority(message.topic, payload),
                    sequence,
                    callback,
                    message.topic,
                    payload,
                )
            )

    def _run_callbacks(self, callback_queue: queue.PriorityQueue) -> None:
        while True:
            _priority, sequence, callback, topic, payload = callback_queue.get()
            if callback is None:
                callback_queue.task_done()
                return
            started = time.monotonic()
            try:
                callback(topic, payload)
            except Exception:
                LOGGER.exception("MQTT handler failed topic=%s client=%s", topic, self.client_id)
            finally:
                elapsed_ms = (time.monotonic() - started) * 1000.0
                if elapsed_ms >= 250.0:
                    LOGGER.warning(
                        "diagnostic slow MQTT handler client=%s topic=%s callback=%s "
                        "elapsed_ms=%.1f payload_bytes=%d",
                        self.client_id,
                        topic,
                        getattr(callback, "__qualname__", repr(callback)),
                        elapsed_ms,
                        len(payload),
                    )
                if hasattr(self, "_callback_token_lock"):
                    with self._callback_token_lock:
                        token = self._callback_tokens.pop(sequence, None)
                        if token is not None:
                            self._pending_callback_tokens.discard(token)
                callback_queue.task_done()

    @staticmethod
    def _pending_result_token(
        topic: str,
        payload: bytes,
        callback: MessageCallback,
    ) -> tuple[str, int] | None:
        """Return an exact in-flight deduplication key for typed results.

        This intentionally uses the reliable wrapper's ``message_id``, not an
        occurrence or semantic identity.  Distinct observations and distinct
        provider attempts therefore remain independently ordered and applied.
        """

        if not topic.startswith("fable/v1/result/"):
            return None
        try:
            message_id = str(json.loads(payload).get("message_id", ""))
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            return None
        if not message_id:
            return None
        return message_id, id(callback)

    @staticmethod
    def _is_control_topic(topic: str) -> bool:
        return any(
            marker in topic
            for marker in (
                "/status/",
                "/heartbeat",
                "/command/",
                "/cancel/",
                "/ack/",
                "/artifact/",
                "/provider_status/",
                "/dispatch/",
                "/evaluation/resource_change",
            )
        )

    @staticmethod
    def _message_priority(topic: str, payload: bytes) -> int:
        if "/evaluation/resource_change" in topic:
            return -2
        if PahoMQTTTransport._is_control_topic(topic):
            return -1
        if topic.startswith("fable/v1/result/"):
            return 0
        if topic.endswith("/predicates"):
            try:
                predicate_id = str(json.loads(payload).get("predicate_id", "")).upper()
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                predicate_id = ""
            if predicate_id in {
                "PASSES",
                "ENTERS",
                "EXITS",
                "GUNSHOT",
                "ALARM",
                "SAME_ENTITY",
            }:
                return 1
            return 2
        return 3


class ReliableMessenger:
    """Persists outgoing QoS-1 messages until application acknowledgment."""

    def __init__(
        self,
        *,
        entity_id: str,
        transport: Transport,
        outbox: SQLiteOutbox,
        retry_interval: float = 1.0,
        retry_after: timedelta = timedelta(seconds=1),
    ) -> None:
        self.entity_id = entity_id
        self.transport = transport
        self.outbox = outbox
        self.retry_interval = retry_interval
        self.retry_after = retry_after
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def send_model(
        self,
        topic: str,
        model: BaseModel,
        *,
        message_id: str,
        qos: int = 1,
        retain: bool = False,
        require_application_ack: bool = True,
    ) -> None:
        payload = encode_model(model)
        self.outbox.enqueue(
            message_id=message_id,
            topic=topic,
            payload=payload,
            qos=qos,
            retain=retain,
            requires_ack=require_application_ack,
        )
        self._publish_item(message_id)
        if not require_application_ack:
            self.outbox.mark_application_acked(message_id)

    def send_raw(
        self,
        topic: str,
        payload: bytes,
        *,
        message_id: str,
        qos: int = 1,
        retain: bool = False,
        require_application_ack: bool = True,
    ) -> None:
        self.outbox.enqueue(
            message_id=message_id,
            topic=topic,
            payload=payload,
            qos=qos,
            retain=retain,
            requires_ack=require_application_ack,
        )
        self._publish_item(message_id)
        if not require_application_ack:
            self.outbox.mark_application_acked(message_id)

    def accept_ack(self, payload: bytes) -> ApplicationAck:
        ack = decode_model(payload, ApplicationAck)
        self.outbox.mark_application_acked(str(ack.acked_message_id))
        return ack

    def flush(self, *, limit: int = 100) -> int:
        count = 0
        for item in self.outbox.pending(limit=limit, retry_after=self.retry_after):
            self.transport.publish(item.topic, item.payload, qos=item.qos, retain=item.retain)
            self.outbox.mark_attempt(item.message_id)
            count += 1
            if not item.requires_ack:
                self.outbox.mark_application_acked(item.message_id)
        return count

    def start_retry_loop(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._retry_loop,
            name=f"fable-outbox-{self.entity_id}",
            daemon=True,
        )
        self._thread.start()

    def stop_retry_loop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.retry_interval * 2))

    def _publish_item(self, message_id: str) -> None:
        item = self.outbox.get(message_id)
        if item is None:
            raise KeyError(message_id)
        self.transport.publish(item.topic, item.payload, qos=item.qos, retain=item.retain)
        self.outbox.mark_attempt(message_id)

    def _retry_loop(self) -> None:
        while not self._stop_event.wait(self.retry_interval):
            try:
                self.flush()
            except Exception:
                LOGGER.exception("outbox flush failed entity=%s", self.entity_id)


def mqtt_topic_matches(topic_filter: str, topic: str) -> bool:
    filter_parts = topic_filter.strip("/").split("/")
    topic_parts = topic.strip("/").split("/")
    for index, part in enumerate(filter_parts):
        if part == "#":
            return index == len(filter_parts) - 1
        if index >= len(topic_parts):
            return False
        if part != "+" and part != topic_parts[index]:
            return False
    return len(topic_parts) == len(filter_parts)
