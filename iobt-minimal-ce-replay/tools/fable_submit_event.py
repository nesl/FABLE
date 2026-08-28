#!/usr/bin/env python3
"""Submit a semantic complex-event request to the closed-loop FABLE controller."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import threading
import time


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from fable.common.time import EventTimeInterval
from fable.distributed.codec import decode_model, encode_model
from fable.distributed.models import EventRequestResponse, EventRequestSubmission
from fable.distributed.topics import event_request_topic, event_response_topic


def parse_time(value: str) -> datetime:
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except ValueError:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--orchestrator-id", default="orchestrator")
    parser.add_argument("--family", required=True, help="Authored event family, e.g. convoy or robbery")
    parser.add_argument("--parameters-json", default="{}")
    parser.add_argument("--request-id", default=f"event-{int(time.time())}")
    parser.add_argument("--event-start")
    parser.add_argument("--event-end")
    parser.add_argument("--horizon-sec", type=float, default=300.0)
    parser.add_argument("--deadline-sec", type=float, default=300.0)
    parser.add_argument("--allow-raw-transfer", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    args = parser.parse_args()

    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise SystemExit(
            "Missing paho-mqtt. Install the replay requirements or run this tool "
            "from iobt-minimal-ce-replay/.venv."
        ) from exc

    parameters = json.loads(args.parameters_json)
    interval = None
    if args.event_start or args.event_end:
        if not (args.event_start and args.event_end):
            parser.error("--event-start and --event-end must be supplied together")
        interval = EventTimeInterval(
            start=parse_time(args.event_start),
            end=parse_time(args.event_end),
        )

    submitter_id = f"fable-event-submit-{int(time.time() * 1000)}"
    request = EventRequestSubmission(
        submitter_id=submitter_id,
        request_id=args.request_id,
        family_id=args.family,
        parameters=parameters,
        event_time_window=interval,
        hypothesis_horizon_ms=max(1, int(args.horizon_sec * 1000)),
        deadline_offset_ms=max(1, int(args.deadline_sec * 1000)),
        raw_data_must_remain_local=not args.allow_raw_transfer,
    )
    done = threading.Event()
    result: dict[str, object] = {}
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=submitter_id,
        protocol=mqtt.MQTTv311,
    )

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if int(getattr(reason_code, "value", reason_code)) != 0:
            result["error"] = f"MQTT connect failed: {reason_code}"
            done.set()
            return
        client.subscribe(event_response_topic(submitter_id), qos=1)
        client.publish(
            event_request_topic(args.orchestrator_id),
            payload=encode_model(request),
            qos=1,
        )

    def on_message(client, userdata, message):
        try:
            response = decode_model(message.payload, EventRequestResponse)
            if response.request_message_id != request.message_id:
                return
            result["response"] = response
        except Exception as exc:
            result["error"] = str(exc)
        done.set()

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.host, args.port, keepalive=60)
    client.loop_start()
    try:
        if not done.wait(args.timeout_sec):
            print("Timed out waiting for an event-request response", file=sys.stderr)
            return 2
    finally:
        client.disconnect()
        client.loop_stop()

    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1
    response = result["response"]
    print(json.dumps(response.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if response.accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
