#!/usr/bin/env python3
"""Publish timestamped MQTT latency probes from the active network namespace.

Run this through nsenter into a NetWaggle node namespace. Because the process
uses the same host clock as the web UI container, the UI can estimate one-way
node->broker->UI latency as receive_time - sent_ts.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from typing import Any, Dict

try:
    import paho.mqtt.client as mqtt
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing paho-mqtt. Install with: /usr/bin/python3 -m pip install paho-mqtt") from exc


def make_client(client_id: str) -> mqtt.Client:
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:
        return mqtt.Client(client_id=client_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--node", required=True)
    ap.add_argument("--anchor-container", default="")
    ap.add_argument("--profile", default="")
    ap.add_argument("--mqtt-host", default="10.255.0.1")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--count", type=int, default=0, help="0 means run forever")
    ap.add_argument("--qos", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--payload-bytes", type=int, default=0)
    ap.add_argument("--configured-one-way-ms", type=float, default=None)
    ap.add_argument("--log-every-n", type=int, default=10, help="Print a success line every N probes; 0 disables success logs")
    args = ap.parse_args()

    topic = f"/netwaggle/probe/{args.node}"
    client_id = f"netwaggle-latency-probe-{args.node}-{os.getpid()}"
    client = make_client(client_id)
    client.connect(args.mqtt_host, args.mqtt_port, keepalive=30)
    client.loop_start()

    filler = "x" * max(0, int(args.payload_bytes))
    seq = 0
    try:
        while args.count <= 0 or seq < args.count:
            seq += 1
            now = time.time()
            payload: Dict[str, Any] = {
                "source": "netwaggle_latency_probe",
                "node": args.node,
                "anchor_container": args.anchor_container,
                "profile": args.profile,
                "seq": seq,
                "sent_ts": now,
                "sent_ts_ms": int(now * 1000),
                "monotonic_ts": time.monotonic(),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "mqtt_host": args.mqtt_host,
                "mqtt_port": args.mqtt_port,
            }
            if args.configured_one_way_ms is not None:
                payload["configured_one_way_ms"] = args.configured_one_way_ms
            if filler:
                payload["padding"] = filler
            info = client.publish(topic, json.dumps(payload, separators=(",", ":")), qos=args.qos, retain=False)
            if args.qos:
                info.wait_for_publish(timeout=5)
            if args.log_every_n and (seq == 1 or seq % max(1, args.log_every_n) == 0):
                print(json.dumps({
                    "event": "probe_publish",
                    "node": args.node,
                    "seq": seq,
                    "topic": topic,
                    "mqtt_host": args.mqtt_host,
                    "configured_one_way_ms": args.configured_one_way_ms,
                    "sent_ts": now,
                }, separators=(",", ":")), flush=True)
            time.sleep(max(args.interval, 0.05))
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
