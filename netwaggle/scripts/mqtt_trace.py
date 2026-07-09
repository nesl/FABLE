#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import paho.mqtt.client as mqtt


def classify(topic: str) -> str:
    if topic.startswith("/replay/") or topic == "control" or topic.endswith("/control"):
        return "control"
    if "/analytics/yolo/bbox" in topic or "/audio_detector/detections" in topic or topic.startswith("/complex_events"):
        return "predicate_or_event"
    if "/continuation/" in topic:
        return "continuation"
    if "/artifact/" in topic:
        return "artifact"
    if topic.startswith("/debug/"):
        return "debug"
    if "/gps" in topic or topic.endswith("/gps"):
        return "predicate_or_event"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description="Trace MQTT topic/payload sizes to JSONL.")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--topic", default="#")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("a", encoding="utf-8")

    def on_connect(client, _userdata, _flags, rc, *_extra):
        print(f"connected rc={rc}; subscribing to {args.topic}", flush=True)
        client.subscribe(args.topic)

    def on_message(_client, _userdata, msg):
        row = {
            "ts": time.time(),
            "topic": msg.topic,
            "payload_bytes": len(msg.payload or b""),
            "category": classify(msg.topic),
            "qos": msg.qos,
            "retain": bool(msg.retain),
        }
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, 60)
    try:
        client.loop_forever()
    finally:
        f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
