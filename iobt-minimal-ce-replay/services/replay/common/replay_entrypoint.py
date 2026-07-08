#!/usr/bin/env python3
"""Persistent replay-container supervisor.

The original replay apps expect /output (or /data/<scenario>/GPS) to already be
bound to one scenario/date folder. This wrapper lets the replay containers stay
alive across scenarios:

  1. Subscribe to /replay/config.
  2. Resolve the requested scenario against mounted data roots.
  3. Create /output or /data symlinks for the selected node/date files.
  4. Start/restart the original replay app as a child process.
  5. Re-broadcast /replay/sync once after child startup so the child does not
     miss sync messages published immediately after config.

It intentionally leaves the original replay app mostly unchanged.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import paho.mqtt.client as mqtt

CONFIG_TOPIC = "/replay/config"
SYNC_TOPIC = "/replay/sync"

DEFAULT_ROOTS = [
    "/data_roots/west_point",
    "/data_roots/gq",
]


def scenario_date(scenario: str) -> str:
    return scenario.split("_")[0]


def env_roots() -> list[Path]:
    raw = os.environ.get("IOBT_DATA_ROOTS", os.pathsep.join(DEFAULT_ROOTS))
    roots: list[Path] = []
    seen: set[str] = set()
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        p = Path(part)
        key = str(p)
        if key not in seen:
            seen.add(key)
            roots.append(p)
    return roots


def publish_json(client: mqtt.Client, topic: str, payload: dict[str, Any]) -> None:
    try:
        client.publish(topic, json.dumps(payload), qos=1)
    except Exception as exc:
        print(f"[supervisor] MQTT publish failed topic={topic}: {exc}", flush=True)


def clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def find_first(patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[0])
    return None


class ReplaySupervisor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.service = args.service
        self.node_folder = args.node_folder
        self.node_name = args.node_name
        self.child: subprocess.Popen[str] | None = None
        self.last_sync_payload: dict[str, Any] | None = None
        self.last_config_payload: dict[str, Any] | None = None
        self.lock = threading.RLock()
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    @property
    def status_topic(self) -> str:
        if self.service == "gps":
            return "/replay/status/supervisor/gps"
        return f"/replay/status/supervisor/{self.service}/{self.node_name}"

    @property
    def error_topic(self) -> str:
        if self.service == "gps":
            return "/replay/error/supervisor/gps"
        return f"/replay/error/supervisor/{self.service}/{self.node_name}"

    def status(self, **payload: Any) -> None:
        payload.setdefault("service", f"{self.service}_supervisor")
        payload.setdefault("node", self.node_name)
        payload.setdefault("t", time.time())
        publish_json(self.client, self.status_topic, payload)

    def error(self, message: str, **payload: Any) -> None:
        payload.setdefault("service", f"{self.service}_supervisor")
        payload.setdefault("node", self.node_name)
        payload.setdefault("error", message)
        payload.setdefault("t", time.time())
        print(f"[supervisor:{self.service}:{self.node_name}] ERROR: {message}", flush=True)
        publish_json(self.client, self.error_topic, payload)

    def on_connect(self, client: mqtt.Client, _userdata: Any, _flags: Any, rc: int, *_extra: Any) -> None:
        print(f"[supervisor:{self.service}:{self.node_name}] MQTT connected rc={rc}", flush=True)
        client.subscribe(CONFIG_TOPIC, qos=1)
        client.subscribe(SYNC_TOPIC, qos=1)
        self.status(state="idle", message="connected and waiting for /replay/config")

    def on_message(self, _client: mqtt.Client, _userdata: Any, msg: mqtt.MQTTMessage) -> None:
        payload_raw = msg.payload.decode("utf-8", errors="replace")
        try:
            payload = json.loads(payload_raw)
        except Exception:
            payload = {"raw": payload_raw}

        if msg.topic == CONFIG_TOPIC:
            self.handle_config(payload)
        elif msg.topic == SYNC_TOPIC:
            # Ignore sync messages that this supervisor re-broadcasts, otherwise
            # we can create an infinite rebroadcast loop.
            if isinstance(payload, dict) and payload.get("_supervisor_rebroadcast"):
                return
            with self.lock:
                self.last_sync_payload = payload
            self.rebroadcast_sync_after_child_ready(delay=1.0)

    def stop_child(self) -> None:
        with self.lock:
            child = self.child
            self.child = None
        if child and child.poll() is None:
            print(f"[supervisor:{self.service}:{self.node_name}] stopping child pid={child.pid}", flush=True)
            child.terminate()
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)

    def resolve_node_files(self, scenario: str) -> tuple[Path, dict[str, Path]] | None:
        date = scenario_date(scenario)
        roots = env_roots()
        searched: list[str] = []
        for root in roots:
            node_dir = root / date / self.node_folder
            searched.append(str(node_dir))
            if not node_dir.is_dir():
                continue
            if self.service == "zed":
                video = find_first([
                    str(node_dir / f"{scenario}_*zed*.svo2"),
                    str(node_dir / f"{scenario}_*zed*.svo"),
                ])
                csv = find_first([str(node_dir / f"{scenario}_*zed*.csv")])
                if video and csv:
                    return node_dir, {"video": video, "csv": csv}
            elif self.service == "respeaker":
                flac = find_first([str(node_dir / f"{scenario}_*respeaker*.flac")])
                csv = find_first([str(node_dir / f"{scenario}_*respeaker*.csv")])
                if flac and csv:
                    return node_dir, {"flac": flac, "csv": csv}
        self.error(
            f"no {self.service} files for scenario={scenario} node_folder={self.node_folder}",
            searched=searched,
        )
        return None

    def resolve_gps_files(self, scenario: str) -> tuple[Path, list[Path]] | None:
        date = scenario_date(scenario)
        searched: list[str] = []
        for root in env_roots():
            gps_dir = root / date / "GPS"
            searched.append(str(gps_dir))
            if not gps_dir.is_dir():
                continue
            files = sorted(Path(p) for p in glob.glob(str(gps_dir / "*" / f"{scenario}_*gps.csv")))
            if files:
                return gps_dir, files
        self.error(f"no GPS files for scenario={scenario}", searched=searched)
        return None

    def prepare_symlinks(self, scenario: str) -> bool:
        if self.service in {"zed", "respeaker"}:
            resolved = self.resolve_node_files(scenario)
            if not resolved:
                return False
            _node_dir, files = resolved
            out = Path("/output")
            clean_dir(out)
            if self.service == "zed":
                symlink_force(files["video"], out / f"{scenario}_{self.node_name}_zed.svo2")
                symlink_force(files["csv"], out / f"{scenario}_{self.node_name}_zed.csv")
            else:
                symlink_force(files["flac"], out / f"{scenario}_{self.node_name}_respeaker.flac")
                symlink_force(files["csv"], out / f"{scenario}_{self.node_name}_respeaker.csv")
            self.status(state="prepared", scenario=scenario, output=str(out), files={k: str(v) for k, v in files.items()})
            return True

        if self.service == "gps":
            resolved = self.resolve_gps_files(scenario)
            if not resolved:
                return False
            _gps_dir, files = resolved
            base = Path("/data") / scenario / "GPS"
            if base.exists():
                shutil.rmtree(base)
            for src in files:
                object_name = src.parent.name
                symlink_force(src, base / object_name / src.name)
            self.status(state="prepared", scenario=scenario, files=[str(f) for f in files], output=str(base))
            return True

        self.error(f"unknown service: {self.service}")
        return False

    def _format_child_offset(self, value: float) -> str:
        """Format replay offsets for the original child apps.

        The persistent supervisor accepts float-valued /replay/config payloads
        because the web UI uses numeric fields. The original ZED and GPS replay
        apps, however, define --start/--end as int argparse fields, so passing
        "0.0" crashes before replay begins. ReSpeaker accepts floats, so keep
        fractional values there.
        """
        if self.service in {"zed", "gps"}:
            if value == -1 or value < 0:
                return "-1"
            return str(int(value))
        return f"{value:g}"

    def child_command(self, scenario: str, start: float, end: float) -> list[str]:
        cmd = [
            sys.executable,
            "-u",
            self.args.child_app,
            "--scenario",
            scenario,
            "--start",
            self._format_child_offset(start),
            "--end",
            self._format_child_offset(end),
        ]
        if self.service == "zed":
            if not self.args.debug_raw_mqtt:
                cmd.extend(["--no_rgb_mqtt", "--no_depth_mqtt"])
            cmd.extend(["--downsample_for_mqtt", str(self.args.downsample_for_mqtt)])
        return cmd

    def start_child(self, scenario: str, start: float, end: float) -> None:
        cmd = self.child_command(scenario, start, end)
        env = os.environ.copy()
        env["MCP_NODE_NAME"] = self.node_name
        env.setdefault("MCP_CONTAINER_OUTPUT_DIR", "/output")
        print(f"[supervisor:{self.service}:{self.node_name}] starting child: {' '.join(cmd)}", flush=True)
        child = subprocess.Popen(cmd, env=env, text=True)
        with self.lock:
            self.child = child
        self.status(state="running", scenario=scenario, pid=child.pid, command=cmd)
        self.rebroadcast_sync_after_child_ready(delay=self.args.sync_rebroadcast_delay)

        def watcher() -> None:
            rc = child.wait()
            with self.lock:
                if self.child is child:
                    self.child = None
            self.status(state="child_exited", scenario=scenario, returncode=rc)

        threading.Thread(target=watcher, daemon=True).start()

    def rebroadcast_sync_after_child_ready(self, delay: float) -> None:
        with self.lock:
            has_child = self.child is not None and self.child.poll() is None
            sync_payload = dict(self.last_sync_payload) if isinstance(self.last_sync_payload, dict) else None
        if not has_child or not sync_payload:
            return

        def send() -> None:
            with self.lock:
                child_ok = self.child is not None and self.child.poll() is None
                payload = dict(self.last_sync_payload) if isinstance(self.last_sync_payload, dict) else None
            if not child_ok or not payload:
                return
            payload["_supervisor_rebroadcast"] = True
            payload["_rebroadcast_for"] = f"{self.service}:{self.node_name}"
            print(f"[supervisor:{self.service}:{self.node_name}] rebroadcasting sync for child", flush=True)
            publish_json(self.client, SYNC_TOPIC, payload)

        threading.Timer(delay, send).start()

    def handle_config(self, payload: dict[str, Any]) -> None:
        scenario = str(payload.get("scenario") or "").strip()
        if not scenario:
            self.error("/replay/config missing non-empty scenario", payload=payload)
            return
        try:
            start = float(payload.get("start_time", payload.get("start", 0.0)))
            end = float(payload.get("end_time", payload.get("end", -1.0)))
        except Exception as exc:
            self.error(f"invalid start/end in config: {exc}", payload=payload)
            return

        with self.lock:
            self.last_config_payload = payload

        self.status(state="config_received", scenario=scenario, start=start, end=end)
        self.stop_child()
        if not self.prepare_symlinks(scenario):
            self.status(state="idle_no_data", scenario=scenario)
            return
        self.start_child(scenario, start, end)

    def run(self) -> None:
        host = os.environ.get("MQTT_HOST_IP", "host.docker.internal")
        port = int(os.environ.get("MQTT_PORT", "1883"))
        print(f"[supervisor:{self.service}:{self.node_name}] connecting to MQTT {host}:{port}", flush=True)
        while True:
            try:
                self.client.connect(host, port, 60)
                break
            except Exception as exc:
                print(f"[supervisor:{self.service}:{self.node_name}] MQTT connect failed: {exc}; retrying", flush=True)
                time.sleep(2)
        self.client.loop_start()

        def shutdown(_signum: int, _frame: Any) -> None:
            self.stop_child()
            self.client.loop_stop()
            self.client.disconnect()
            sys.exit(0)

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)

        while True:
            time.sleep(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Persistent replay service supervisor")
    ap.add_argument("--service", required=True, choices=["zed", "respeaker", "gps"])
    ap.add_argument("--node-folder", default="", help="Host data folder name, e.g. orin11. Not used for GPS.")
    ap.add_argument("--node-name", default="x86server", help="MCP node name, e.g. dvpg_gq_orin_11")
    ap.add_argument("--child-app", default="/app/app.py")
    ap.add_argument("--debug-raw-mqtt", action="store_true", help="For ZED, allow RGB/depth MQTT debug streams")
    ap.add_argument("--downsample-for-mqtt", type=float, default=1.0)
    ap.add_argument("--sync-rebroadcast-delay", type=float, default=1.5)
    args = ap.parse_args()

    if args.service != "gps" and not args.node_folder:
        ap.error("--node-folder is required for zed/respeaker")

    ReplaySupervisor(args).run()


if __name__ == "__main__":
    main()
