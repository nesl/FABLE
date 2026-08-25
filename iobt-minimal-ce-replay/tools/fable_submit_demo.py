#!/usr/bin/env python3
"""Submit a typed Phase-5 candidate to the Phase-6 orchestrator over MQTT."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import threading
import time

import paho.mqtt.client as mqtt

# Run from FABLE/iobt-minimal-ce-replay.  The parent repository contains the
# installed/editable FABLE package when this tool is used outside Docker.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fable.common.time import EventTimeInterval, utc_now
from fable.distributed.codec import decode_model, encode_model
from fable.debug import build_replay_audio_candidate
from fable.distributed.models import PlanDispatchRequest, PlanDispatchResponse
from fable.distributed.topics import dispatch_request_topic, dispatch_response_topic
from fable.planning.provider_registry import ProviderRegistry


def parse_time(value: str) -> datetime:
    text = value.strip()
    try:
        numeric = float(text)
        return datetime.fromtimestamp(numeric, tz=UTC)
    except ValueError:
        pass
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--orchestrator-id", default="orchestrator")
    parser.add_argument("--node", default="dvpg_gq_orin_11")
    parser.add_argument("--source", default="orin11_microphone")
    parser.add_argument("--label", default="loud_audio")
    parser.add_argument("--request-id", default="replay_audio_demo")
    parser.add_argument(
        "--event-start",
        help="Replay event-time lower bound (ISO-8601 or Unix seconds). Default: 2024-01-01 UTC.",
    )
    parser.add_argument(
        "--event-end",
        help="Replay event-time upper bound. Default: 2027-12-31 UTC.",
    )
    parser.add_argument("--deadline-sec", type=float, default=300.0)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    args = parser.parse_args()

    start = parse_time(args.event_start) if args.event_start else datetime(2024, 1, 1, tzinfo=UTC)
    end = parse_time(args.event_end) if args.event_end else datetime(2027, 12, 31, 23, 59, 59, tzinfo=UTC)
    now = utc_now()
    registry = ProviderRegistry.from_files(
        catalog_path=REPO / "providers/registry/catalog.yaml",
        data_types_path=REPO / "providers/registry/data_types.yaml",
    )
    candidate = build_replay_audio_candidate(
        provider_registry=registry,
        node_id=args.node,
        source_id=args.source,
        event_interval=EventTimeInterval(start=start, end=end),
        label=args.label,
        request_id=args.request_id,
        deadline_seconds=args.deadline_sec,
        now=now,
    )
    submitter_id = f"fable-submit-{int(time.time())}"
    request = PlanDispatchRequest(submitter_id=submitter_id, candidates=(candidate,))
    response_event = threading.Event()
    result: dict[str, object] = {}

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=submitter_id,
        protocol=mqtt.MQTTv311,
    )

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if int(getattr(reason_code, "value", reason_code)) != 0:
            result["error"] = f"MQTT connect failed: {reason_code}"
            response_event.set()
            return
        client.subscribe(dispatch_response_topic(submitter_id), qos=1)
        client.publish(
            dispatch_request_topic(args.orchestrator_id),
            payload=encode_model(request),
            qos=1,
        )

    def on_message(client, userdata, message):
        try:
            response = decode_model(message.payload, PlanDispatchResponse)
            if response.request_message_id != request.message_id:
                return
            result["response"] = response
        except Exception as exc:
            result["error"] = str(exc)
        response_event.set()

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=60)
    client.loop_start()
    try:
        if not response_event.wait(args.timeout_sec):
            print("Timed out waiting for a dispatch response", file=sys.stderr)
            return 2
    finally:
        client.disconnect()
        client.loop_stop()

    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1
    response = result["response"]
    print(json.dumps(response.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if response.admitted_plan_ids else 3


if __name__ == "__main__":
    raise SystemExit(main())
