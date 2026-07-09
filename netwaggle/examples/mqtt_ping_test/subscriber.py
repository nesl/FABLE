#!/usr/bin/env python3
import argparse, json, time
import paho.mqtt.client as mqtt

ap = argparse.ArgumentParser()
ap.add_argument('--host', default='10.255.0.1')
ap.add_argument('--port', type=int, default=1883)
ap.add_argument('--topic', default='/netwaggle/ping')
args = ap.parse_args()

def on_connect(client, userdata, flags, rc, *extra):
    print('connected', rc, flush=True)
    client.subscribe(args.topic)

def on_message(client, userdata, msg):
    now = time.time()
    try:
        payload = json.loads(msg.payload.decode())
        print({'topic': msg.topic, 'seq': payload.get('seq'), 'latency_s': now - float(payload.get('ts', now))}, flush=True)
    except Exception:
        print({'topic': msg.topic, 'bytes': len(msg.payload)}, flush=True)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(args.host, args.port, 60)
client.loop_forever()
