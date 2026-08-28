#!/usr/bin/env python3
"""Launch one typed E4 controller run against the replay/testbed stack."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
import time
from pathlib import Path

FABLE_ROOT = Path(__file__).resolve().parents[2]
if str(FABLE_ROOT) not in sys.path:
    sys.path.insert(0, str(FABLE_ROOT))

from evaluation.e4_worker import E4Worker, planning_policy_for_baseline
from evaluation.runner import EvaluationRunner
from evaluation.schemas import BaselineId, EvaluationMode
from fable.common.schemas import RuntimeLinkUpdate, RuntimeNodeUpdate
from fable.common.time import EventTimeInterval
from fable.distributed.models import EventRequestSubmission, RuntimeDisturbanceRequest
from fable.distributed.transport import PahoMQTTTransport


def _deployment_node_id(value: str) -> str:
    """Accept replay-friendly orinN names but submit canonical node IDs."""

    value = value.strip()
    suffix = value.removeprefix("orin")
    if value.startswith("orin") and suffix.isdigit():
        return f"dvpg_gq_orin_{int(suffix)}"
    return value


def _load_disturbances(path: Path | None, submitter_id: str):
    if path is None:
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else [raw]
    parsed = []
    for index, row in enumerate(rows):
        delay = float(row.pop("after_seconds", 0.0))
        parsed.append(
            (
                delay,
                RuntimeDisturbanceRequest(
                    submitter_id=submitter_id,
                    disturbance_id=str(row.pop("disturbance_id", f"e4-{index}")),
                    reason=str(row.pop("reason", "E4 disturbance")),
                    node_updates=tuple(
                        RuntimeNodeUpdate.model_validate(item)
                        for item in row.pop("node_updates", ())
                    ),
                    link_updates=tuple(
                        RuntimeLinkUpdate.model_validate(item)
                        for item in row.pop("link_updates", ())
                    ),
                ),
            )
        )
        if row:
            raise ValueError(f"unknown disturbance fields: {sorted(row)}")
    return tuple(parsed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--orchestrator-id", default="orchestrator")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--parameters-json", default="{}")
    parser.add_argument("--allowed-node", action="append", default=[])
    parser.add_argument("--event-start")
    parser.add_argument("--event-end")
    parser.add_argument(
        "--baseline",
        choices=[
            BaselineId.B1_HANDWRITTEN_STATIC.value,
            BaselineId.B3_TASK_RESOURCE_ADAPTIVE.value,
            BaselineId.FABLE.value,
        ],
        required=True,
    )
    parser.add_argument("--disturbance-file", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=Path("evaluation-runs"))
    parser.add_argument("--replay-scenario")
    parser.add_argument("--replay-node", action="append", default=[])
    parser.add_argument("--replay-start", type=float, default=0.0)
    parser.add_argument("--replay-end", type=float, default=-1.0)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--replay-id")
    parser.add_argument("--replay-drain-seconds", type=float, default=5.0)
    args = parser.parse_args()

    if bool(args.event_start) != bool(args.event_end):
        parser.error("--event-start and --event-end must be supplied together")
    if args.playback_speed <= 0:
        parser.error("--playback-speed must be positive")
    if args.replay_scenario and not args.replay_node:
        parser.error("--replay-scenario requires at least one --replay-node")
    try:
        parameters = json.loads(args.parameters_json)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --parameters-json: {exc}")
    if not isinstance(parameters, dict):
        parser.error("--parameters-json must decode to an object")
    event_window = (
        EventTimeInterval(
            start=datetime.fromisoformat(args.event_start.replace("Z", "+00:00")),
            end=datetime.fromisoformat(args.event_end.replace("Z", "+00:00")),
        )
        if args.event_start
        else None
    )

    baseline = BaselineId(args.baseline)
    submitter_id = f"e4-{args.run_id}-{baseline.value}"
    transport = PahoMQTTTransport(
        host=args.host,
        port=args.port,
        client_id=submitter_id,
    )
    worker = E4Worker(
        transport=transport,
        submitter_id=submitter_id,
        orchestrator_id=args.orchestrator_id,
    )
    worker.bind()
    transport.start()
    if not transport.wait_connected(timeout=15.0):
        raise RuntimeError("E4 worker could not connect to MQTT")

    replay_id = args.replay_id or f"e4-{args.run_id}"
    replay_started = False
    try:
        response = worker.submit_event(
            EventRequestSubmission(
                submitter_id=submitter_id,
                request_id=args.request_id,
                family_id=args.family,
                parameters=parameters,
                event_time_window=event_window,
                allowed_node_ids=tuple(_deployment_node_id(item) for item in args.allowed_node),
                planning_policy_id=planning_policy_for_baseline(baseline),
            ),
            timeout=15.0,
        )
        if not response.accepted:
            raise RuntimeError(f"event request rejected: {response.reason}")
        print(json.dumps({"event_request": response.model_dump(mode="json")}, default=str))

        if args.replay_scenario:
            replay_payload = {
                "scenario": args.replay_scenario,
                "start_time": args.replay_start,
                "end_time": args.replay_end,
                "playback_mode": "realtime",
                "speed": args.playback_speed,
                "replay_id": replay_id,
                "target_nodes": sorted(set(args.replay_node)),
            }
            transport.publish(
                "/replay/config",
                json.dumps(replay_payload).encode("utf-8"),
                qos=1,
                retain=True,
            )
            transport.publish(
                "/replay/sync",
                json.dumps(replay_payload).encode("utf-8"),
                qos=1,
                retain=True,
            )
            replay_started = True
            print(json.dumps({"replay_start": replay_payload}, default=str))

        for delay, disturbance in _load_disturbances(args.disturbance_file, submitter_id):
            if delay > 0:
                time.sleep(delay)
            ack = worker.inject_disturbance(disturbance, timeout=15.0)
            if not ack.accepted:
                raise RuntimeError(f"disturbance rejected: {ack.reason}")
            print(json.dumps({"disturbance_ack": ack.model_dump(mode="json")}, default=str))

        terminal = worker.wait_terminal(args.request_id, timeout=args.timeout)
        run_dir = args.output / args.run_id / baseline.value
        EvaluationRunner(run_dir, mode=EvaluationMode.FULL_STACK).record_terminal_event(
            terminal,
            run_id=args.run_id,
            trace_id=args.trace_id,
            baseline_id=baseline,
        )
        print(json.dumps({"terminal_event": terminal.model_dump(mode="json")}, default=str))
        return 0
    finally:
        if replay_started:
            stop_payload = {
                "action": "STOP",
                "replay_id": replay_id,
                "target_nodes": sorted(set(args.replay_node)),
            }
            transport.publish(
                "/replay/config",
                json.dumps(stop_payload).encode("utf-8"),
                qos=1,
                retain=True,
            )
            transport.publish(
                "/replay/sync",
                json.dumps(stop_payload).encode("utf-8"),
                qos=1,
                retain=True,
            )
            if args.replay_drain_seconds > 0:
                time.sleep(args.replay_drain_seconds)
            transport.publish("/replay/config", b"", qos=1, retain=True)
            transport.publish("/replay/sync", b"", qos=1, retain=True)
        transport.stop()


if __name__ == "__main__":
    raise SystemExit(main())
