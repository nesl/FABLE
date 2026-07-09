#!/usr/bin/env python3
"""Send replay config/sync messages to the local MQTT broker.

The web UI and this tool use the same MQTT control path:
  /replay/config  -> scenario/window/timing, retained by default
  /replay/sync    -> synchronized start time, non-retained by default

The generated Docker compose file mounts the data roots; this tool only selects
which scenario prefix to replay.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid

import paho.mqtt.client as mqtt


def normalize_timing(mode: str, speed: float | None) -> tuple[str, float]:
    mode = (mode or "max").strip().lower()
    if mode in {"fast", "asap", "unlimited"}:
        mode = "max"
    if mode not in {"max", "realtime", "scaled"}:
        raise SystemExit("--playback-mode must be max, realtime, or scaled")
    if mode in {"max", "realtime"}:
        return mode, 1.0
    value = 1.0 if speed is None else float(speed)
    if value <= 0:
        raise SystemExit("--speed must be > 0 for scaled mode")
    return mode, value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mqtt-host", default="localhost")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--scenario", help="Scenario/file prefix, e.g. 20260414_111951")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=-1.0)
    ap.add_argument("--sync-delay", type=float, default=1.0, help="Seconds in the future for synchronized start")
    ap.add_argument("--playback-mode", choices=["max", "realtime", "scaled"], default="max",
                    help="max = no intentional timing sleeps; realtime = 1.0x; scaled = use --speed")
    ap.add_argument("--speed", type=float, default=None, help="Multiplier for --playback-mode scaled, e.g. 0.25, 1, 2")
    ap.add_argument("--send-control", action="store_true", help="Also publish collection-start on the global control topic")
    ap.add_argument("--retain-config", action=argparse.BooleanOptionalAction, default=True,
                    help="Retain /replay/config so later-starting supervisors know the selected scenario")
    ap.add_argument("--retain-sync", action=argparse.BooleanOptionalAction, default=False,
                    help="Retain /replay/sync. Usually leave false to avoid stale replay starts")
    ap.add_argument("--sync-burst", action=argparse.BooleanOptionalAction, default=True,
                    help="Publish several identical /replay/sync messages to avoid child-startup races")
    ap.add_argument("--clear-retained", action="store_true", help="Clear retained /replay/config and /replay/sync and exit")
    args = ap.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
    client.loop_start()
    time.sleep(0.2)

    if args.clear_retained:
        client.publish("/replay/config", payload=None, qos=1, retain=True)
        client.publish("/replay/sync", payload=None, qos=1, retain=True)
        print("cleared retained /replay/config and /replay/sync")
        time.sleep(0.2)
        client.loop_stop()
        client.disconnect()
        return

    if not args.scenario:
        raise SystemExit("--scenario is required unless --clear-retained is used")

    mode, speed = normalize_timing(args.playback_mode, args.speed)
    replay_id = f"{args.scenario}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    config = {
        "scenario": args.scenario,
        "start_time": args.start,
        "end_time": args.end,
        "playback_mode": mode,
        "speed": speed,
        "replay_id": replay_id,
    }
    start_at = time.time() + max(args.sync_delay, 0.0)
    sync = {
        "scenario": args.scenario,
        "start_at": start_at,
        "playback_mode": mode,
        "speed": speed,
        "replay_id": replay_id,
    }

    client.publish("/replay/config", json.dumps(config), qos=1, retain=args.retain_config)
    print(f"published /replay/config (retain={args.retain_config}): {json.dumps(config)}")
    time.sleep(0.2)

    if args.send_control:
        client.publish("control", "collection-start", qos=1)
        print("published control: collection-start")
        time.sleep(0.1)

    if args.sync_burst:
        delays = [0.0, max(args.sync_delay * 0.5, 0.25), max(args.sync_delay, 0.0) + 0.25, max(args.sync_delay, 0.0) + 1.0, max(args.sync_delay, 0.0) + 2.0]
        last_delay = 0.0
        for i, delay in enumerate(delays):
            sleep_for = max(delay - last_delay, 0.0)
            if sleep_for:
                time.sleep(sleep_for)
            payload = dict(sync)
            payload["burst_index"] = i
            payload["note"] = "tools_replay_control"
            client.publish("/replay/sync", json.dumps(payload), qos=1, retain=args.retain_sync)
            print(f"published /replay/sync burst[{i}] (retain={args.retain_sync}): {json.dumps(payload)}")
            last_delay = delay
    else:
        client.publish("/replay/sync", json.dumps(sync), qos=1, retain=args.retain_sync)
        print(f"published /replay/sync (retain={args.retain_sync}): {json.dumps(sync)}")
    print(f"replay should start at host time {start_at:.3f} (in ~{max(args.sync_delay, 0.0):.1f}s)")

    time.sleep(0.5)
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
