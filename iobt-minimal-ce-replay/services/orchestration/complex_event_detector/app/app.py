#!/usr/bin/env python3
"""Minimal complex-event detector.

This subscribes to compact MQTT event streams produced by per-node detectors. It is
intentionally simple: if a person/car/truck/bus detection and an audio detection
occur within CE_WINDOW_SEC, it publishes a fused example event.

This is meant as a starting point for your own distributed CE logic.
"""

import json
import os
import time
from collections import deque

import paho.mqtt.client as mqtt

MQTT_HOST = os.environ.get("MQTT_HOST_IP", "host.docker.internal")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
WINDOW_SEC = float(os.environ.get("CE_WINDOW_SEC", "5.0"))
OUTPUT_TOPIC = os.environ.get("CE_OUTPUT_TOPIC", "/complex_events/demo")

YOLO_TOPIC_FILTER = os.environ.get("YOLO_TOPIC_FILTER", "/+/analytics/yolo/bbox")
AUDIO_TOPIC_FILTER = os.environ.get("AUDIO_TOPIC_FILTER", "/+/audio_detector/detections")
GPS_TOPIC_FILTER = os.environ.get("GPS_TOPIC_FILTER", "/+/gps")

RECENT_VISION = deque(maxlen=200)
RECENT_AUDIO = deque(maxlen=200)
RECENT_GPS = deque(maxlen=200)
LAST_FUSED = 0.0


def now() -> float:
    return time.time()


def prune() -> None:
    cutoff = now() - max(WINDOW_SEC * 4, 30.0)
    for q in (RECENT_VISION, RECENT_AUDIO, RECENT_GPS):
        while q and q[0].get("received_at", 0.0) < cutoff:
            q.popleft()


def parse_json_payload(payload: bytes):
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception as exc:
        print(f"[CE] Dropping unparsable payload: {exc}", flush=True)
        return None


def normalize_yolo(topic: str, payload) -> list[dict]:
    # YOLO service publishes a JSON list of detections.
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    out = []
    node = topic.split("/")[1] if topic.startswith("/") else "unknown"
    for det in payload:
        if not isinstance(det, dict):
            continue
        event = dict(det)
        event.setdefault("source_node", node)
        event["received_at"] = now()
        out.append(event)
    return out


def normalize_audio(topic: str, payload) -> dict | None:
    if not isinstance(payload, dict):
        return None
    node = topic.split("/")[1] if topic.startswith("/") else "unknown"
    event = dict(payload)
    event.setdefault("source_node", node)
    event["received_at"] = now()
    return event


def maybe_emit(client: mqtt.Client) -> None:
    global LAST_FUSED
    prune()

    t_now = now()
    if t_now - LAST_FUSED < 1.0:
        return

    interesting_classes = {"person", "car", "truck", "bus", "bicycle", "motorcycle"}
    vision = [v for v in RECENT_VISION if v.get("class") in interesting_classes]
    audio = list(RECENT_AUDIO)

    if not vision or not audio:
        return

    # Simple temporal join over arrival time. Replace this with event-time logic
    # once you have stable timestamps from every primitive detector.
    best = None
    for v in vision:
        for a in audio:
            dt = abs(v["received_at"] - a["received_at"])
            if dt <= WINDOW_SEC and (best is None or dt < best[0]):
                best = (dt, v, a)

    if best is None:
        return

    dt, v, a = best
    event = {
        "type": "vision_audio_cooccurrence",
        "window_sec": WINDOW_SEC,
        "delta_sec": round(dt, 3),
        "vision": v,
        "audio": a,
        "emitted_at": t_now,
    }
    client.publish(OUTPUT_TOPIC, json.dumps(event))
    LAST_FUSED = t_now
    print(f"[CE] Published {OUTPUT_TOPIC}: {json.dumps(event)[:500]}", flush=True)


def on_connect(client, userdata, flags, reason_code, properties=None):
    print(f"[CE] Connected to MQTT {MQTT_HOST}:{MQTT_PORT} rc={reason_code}", flush=True)
    for topic_filter in [YOLO_TOPIC_FILTER, AUDIO_TOPIC_FILTER, GPS_TOPIC_FILTER]:
        client.subscribe(topic_filter)
        print(f"[CE] Subscribed: {topic_filter}", flush=True)


def on_message(client, userdata, msg):
    payload = parse_json_payload(msg.payload)
    if payload is None:
        return

    topic = msg.topic
    if topic.endswith("/analytics/yolo/bbox"):
        events = normalize_yolo(topic, payload)
        RECENT_VISION.extend(events)
        if events:
            print(f"[CE] vision {topic}: {len(events)} detections", flush=True)
    elif topic.endswith("/audio_detector/detections"):
        event = normalize_audio(topic, payload)
        if event:
            RECENT_AUDIO.append(event)
            print(f"[CE] audio {topic}: db={event.get('db')}", flush=True)
    elif topic.endswith("/gps"):
        if isinstance(payload, dict):
            payload["received_at"] = now()
            RECENT_GPS.append(payload)
    else:
        return

    maybe_emit(client)


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    while True:
        try:
            client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            break
        except Exception as exc:
            print(f"[CE] MQTT connect failed: {exc}; retrying", flush=True)
            time.sleep(1)
    client.loop_forever()


if __name__ == "__main__":
    main()
