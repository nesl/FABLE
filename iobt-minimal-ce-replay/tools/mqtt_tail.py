#!/usr/bin/env python3
"""Print MQTT messages for quick debugging."""

import argparse
import datetime as dt

import paho.mqtt.client as mqtt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mqtt-host", default="localhost")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--topic", default="#", help="MQTT topic filter")
    ap.add_argument("--max-len", type=int, default=500)
    args = ap.parse_args()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        print(f"connected rc={reason_code}; subscribing to {args.topic}")
        client.subscribe(args.topic)

    def on_message(client, userdata, msg):
        payload = msg.payload.decode("utf-8", errors="replace")
        if len(payload) > args.max_len:
            payload = payload[: args.max_len] + "..."
        ts = dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}] {msg.topic}: {payload}")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.mqtt_host, args.mqtt_port, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
