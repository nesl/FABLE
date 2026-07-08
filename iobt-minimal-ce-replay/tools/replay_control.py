#!/usr/bin/env python3
"""Send replay config/sync messages to the local MQTT broker.

This tool does not choose the data folder. The generated Docker compose file
mounts the chosen <data-root>/<YYYYMMDD>/... date directory into the replay
containers. This only tells those already-running containers which scenario/file
prefix to replay.

Usage:
  python3 tools/replay_control.py --scenario 20260414_134838 --start 0 --end -1 --sync-delay 10
"""

import argparse
import json
import time

import paho.mqtt.client as mqtt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mqtt-host", default="localhost")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=-1.0)
    ap.add_argument("--sync-delay", type=float, default=10.0, help="Seconds in the future for synchronized start")
    ap.add_argument("--send-control", action="store_true", help="Also publish collection-start on the global control topic")
    args = ap.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
    client.loop_start()
    time.sleep(0.2)

    config = {
        "scenario": args.scenario,
        "start_time": args.start,
        "end_time": args.end,
    }
    start_at = time.time() + args.sync_delay
    sync = {"scenario": args.scenario, "start_at": start_at}

    client.publish("/replay/config", json.dumps(config), qos=1)
    print(f"published /replay/config: {json.dumps(config)}")
    time.sleep(0.5)

    if args.send_control:
        client.publish("control", "collection-start", qos=1)
        print("published control: collection-start")
        time.sleep(0.2)

    client.publish("/replay/sync", json.dumps(sync), qos=1)
    print(f"published /replay/sync: {json.dumps(sync)}")
    print(f"replay should start at host time {start_at:.3f} (in ~{args.sync_delay:.1f}s)")

    time.sleep(0.5)
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
