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
from threading import Lock
from typing import Any

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


def add_boolean_optional_argument(parser: argparse.ArgumentParser, name: str, *, default: bool, help: str) -> None:
    """Add a --foo/--no-foo boolean flag on Python 3.8+.

    argparse.BooleanOptionalAction was added in Python 3.9, but the replay
    desktop is currently using Python 3.8. Keep the same CLI surface without
    requiring a Python upgrade.
    """
    action = getattr(argparse, "BooleanOptionalAction", None)
    if action is not None:
        parser.add_argument(name, action=action, default=default, help=help)
        return

    parser.add_argument(name, dest=name.lstrip("-").replace("-", "_"), action="store_true", default=default, help=help)
    parser.add_argument(f"--no-{name.lstrip('-')}", dest=name.lstrip("-").replace("-", "_"), action="store_false", help=argparse.SUPPRESS)


def make_mqtt_client() -> mqtt.Client:
    """Create a paho-mqtt client across paho 1.x and 2.x."""
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        return mqtt.Client()



def readiness_key(topic: str, payload: Any) -> tuple[str, str] | None:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return None
    if not isinstance(payload, dict):
        return None
    node = str(payload.get("node") or "").strip()
    service = str(payload.get("service") or "").strip()
    if topic.startswith("/readiness/"):
        parts = topic.strip("/").split("/")
        if len(parts) >= 3:
            node = node or parts[1]
            service = service or parts[2]
    if not node or not service:
        return None
    return node, service


def readiness_matches(row: dict[str, Any], service: str, scenario: str, replay_id: str) -> bool:
    if row.get("service") != service or not row.get("ready"):
        return False
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    if service in {"zed", "respeaker", "gps"}:
        p_scenario = str(payload.get("scenario") or "").strip()
        if p_scenario and p_scenario != scenario:
            return False
    p_replay_id = str(payload.get("replay_id") or "").strip()
    if p_replay_id and p_replay_id != replay_id:
        return False
    return True


def wait_for_readiness(client: mqtt.Client, *, scenario: str, replay_id: str, required: list[str], timeout: float) -> dict[str, Any]:
    lock = Lock()
    rows: dict[str, dict[str, Any]] = {}

    def on_message(_client, _userdata, msg):
        raw = msg.payload.decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except Exception:
            payload = raw
        key = readiness_key(msg.topic, payload)
        if not key:
            return
        node, service = key
        with lock:
            rows[f"{node}:{service}"] = {
                "node": node,
                "service": service,
                "ready": bool(payload.get("ready")) if isinstance(payload, dict) else False,
                "payload": payload,
                "topic": msg.topic,
                "ts": time.time(),
            }

    old_on_message = getattr(client, "on_message", None)
    client.on_message = on_message
    client.subscribe("/readiness/#", qos=0)
    deadline = time.time() + max(float(timeout), 0.0)
    last_missing = list(required)
    try:
        while time.time() < deadline:
            with lock:
                current = list(rows.values())
            missing = []
            for service in required:
                if not any(readiness_matches(r, service, scenario, replay_id) for r in current):
                    missing.append(service)
            last_missing = missing
            if not missing:
                return {"ok": True, "missing": [], "services": current}
            time.sleep(0.25)
        with lock:
            current = list(rows.values())
        return {"ok": False, "missing": last_missing, "services": current, "timeout_sec": timeout}
    finally:
        try:
            client.unsubscribe("/readiness/#")
        except Exception:
            pass
        client.on_message = old_on_message

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
    add_boolean_optional_argument(
        ap, "--retain-config", default=True,
        help="Retain /replay/config so later-starting supervisors know the selected scenario",
    )
    add_boolean_optional_argument(
        ap, "--retain-sync", default=False,
        help="Retain /replay/sync. Usually leave false to avoid stale replay starts",
    )
    add_boolean_optional_argument(
        ap, "--sync-burst", default=False,
        help="Publish several identical /replay/sync messages. Default false because readiness gating should make this unnecessary.",
    )
    add_boolean_optional_argument(
        ap, "--wait-ready", default=True,
        help="After /replay/config, wait for required service readiness before /replay/sync",
    )
    ap.add_argument("--ready-timeout", type=float, default=90.0)
    ap.add_argument("--required-ready-services", default="zed,respeaker,yolo,audio_detector")
    ap.add_argument("--clear-retained", action="store_true", help="Clear retained /replay/config and /replay/sync and exit")
    args = ap.parse_args()

    client = make_mqtt_client()
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

    required_ready = [x.strip() for x in str(args.required_ready_services or "").split(",") if x.strip()]
    if args.wait_ready and required_ready:
        print(f"waiting up to {args.ready_timeout:g}s for readiness: {required_ready}")
        report = wait_for_readiness(client, scenario=args.scenario, replay_id=replay_id, required=required_ready, timeout=args.ready_timeout)
        if not report.get("ok"):
            print(json.dumps(report, indent=2, default=str))
            raise SystemExit(f"required services not ready: {report.get('missing')}")
        print("readiness satisfied")

    start_at = time.time() + max(args.sync_delay, 0.0)
    sync["start_at"] = start_at

    if args.sync_burst:
        delays = [0.0, max(args.sync_delay * 0.5, 0.25), max(args.sync_delay, 0.0) + 0.25]
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
