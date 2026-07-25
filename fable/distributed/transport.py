"""MQTT transport adapters and reliable application-level messaging."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import timedelta
import logging
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

    def publish(self, topic: str, payload: bytes, *, qos: int = 1, retain: bool = False) -> None:
        info = self._client.publish(topic, payload=payload, qos=qos, retain=retain)
        if info.rc != self._mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed for {topic}: rc={info.rc}")

    def start(self) -> None:
        if self._started:
            return
        self._started = True
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

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if int(reason_code) != 0:
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
            try:
                callback(message.topic, payload)
            except Exception:
                LOGGER.exception("MQTT handler failed topic=%s client=%s", message.topic, self.client_id)


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
