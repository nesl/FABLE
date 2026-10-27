#!/usr/bin/env python3
"""Serve the persistent dashboard from the active experiment state file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fable.execution import PahoMQTTClient
from fable.visualization import LiveDeviceMonitor, VisualizationServer
from scripts.visualization_contract import IDLE_STATE


def file_state_provider(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return IDLE_STATE


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--mqtt-host", default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1884)
    args = parser.parse_args()
    devices = LiveDeviceMonitor()
    mqtt = PahoMQTTClient(args.mqtt_host, args.mqtt_port, "fable-visualization-devices")
    devices.subscribe(mqtt)
    mqtt.start()
    server = VisualizationServer(
        lambda: file_state_provider(args.state_file), args.host, args.port,
        device_provider=devices,
    )
    print(f"Persistent FABLE dashboard listening on http://{server.address[0]}:{server.address[1]}", flush=True)
    server.start_background()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        mqtt.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
