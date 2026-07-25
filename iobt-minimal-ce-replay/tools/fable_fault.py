#!/usr/bin/env python3
"""Publish a Phase-6 fault-injection command to a node agent."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import paho.mqtt.client as mqtt

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fable.distributed.codec import encode_model
from fable.distributed.models import FaultCommand, FaultKind
from fable.distributed.topics import fault_topic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("kind", choices=[item.value for item in FaultKind])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--provider-instance-id")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=0)
    parser.add_argument("--reason", default="manual Phase-6 fault injection")
    args = parser.parse_args()

    command = FaultCommand(
        target_id=args.target,
        kind=FaultKind(args.kind),
        provider_instance_id=args.provider_instance_id,
        count=args.count,
        duration_ms=args.duration_ms,
        reason=args.reason,
    )
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"fable-fault-{args.target}",
        protocol=mqtt.MQTTv311,
    )
    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()
    info = client.publish(fault_topic(args.target), encode_model(command), qos=1)
    info.wait_for_publish(timeout=5)
    client.disconnect()
    client.loop_stop()
    print(command.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
