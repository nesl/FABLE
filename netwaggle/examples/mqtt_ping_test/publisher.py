#!/usr/bin/env python3
import argparse, json, time
import paho.mqtt.client as mqtt

ap = argparse.ArgumentParser()
ap.add_argument('--host', default='10.255.0.1')
ap.add_argument('--port', type=int, default=1883)
ap.add_argument('--topic', default='/netwaggle/ping')
ap.add_argument('--interval', type=float, default=1.0)
args = ap.parse_args()
client = mqtt.Client()
client.connect(args.host, args.port, 60)
seq = 0
while True:
    payload = {'seq': seq, 'ts': time.time()}
    client.publish(args.topic, json.dumps(payload))
    print(payload, flush=True)
    seq += 1
    time.sleep(args.interval)
