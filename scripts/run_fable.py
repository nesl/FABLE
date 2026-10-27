#!/usr/bin/env python3
"""Run the distributed FABLE controller with an asynchronous result receiver."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
import uuid
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import yaml

from fable.execution import (
    ControllerMQTTResultService, FableRuntime, MqttCommandTransport,
    PahoMQTTClient, ResultTCPServer, TcpCommandTransport,
)
from fable.language import load_and_compile_event
from fable.planning import LinkState, NodeState, RuntimeState, SourceState, load_provider_profiles
from replay.evaluation.artifacts import CompletionArtifact, CompletionWriter
from fable.visualization import VisualizationServer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("event", type=Path)
    parser.add_argument("deployment", type=Path)
    parser.add_argument("--result-host", default="0.0.0.0")
    parser.add_argument("--result-port", type=int, default=8766)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--cell-id")
    parser.add_argument(
        "--visualization-port", type=int, default=0,
        help="Serve the read-only CE progress dashboard on this port (0 disables it).",
    )
    parser.add_argument("--visualization-host", default="0.0.0.0")
    parser.add_argument(
        "--visualization-state-file", type=Path,
        help="publish dashboard state for a separate persistent visualization service",
    )
    parser.add_argument("--visualization-session-name")
    parser.add_argument(
        "--command-timeout-seconds", type=float, default=30.0,
        help="Bound for one NodeAgent lifecycle acknowledgement.",
    )
    parser.add_argument(
        "--frontier-mode", choices=("fable", "whole_event"), default="fable",
        help="Use dynamic FABLE frontiers or the always-on whole-event B0 evaluation policy.",
    )
    args = parser.parse_args()

    event = load_and_compile_event(args.event)
    raw = yaml.safe_load(args.deployment.read_text(encoding="utf-8"))
    nodes = {
        str(node_id): NodeState(
            str(node_id), str(spec.get("node_type", "unknown")), bool(spec.get("available", True)),
            float(spec.get("cpu_free", float("inf"))),
            float(spec.get("memory_mb_free", float("inf"))),
            float(spec.get("gpu_memory_mb_free", float("inf"))),
        )
        for node_id, spec in raw.get("nodes", {}).items()
    }
    sources = {
        str(source_id): SourceState(
            str(source_id), str(spec["node_id"]), str(spec["data_type"]),
            spec.get("site_id"), int(spec.get("sample_bytes", 0)), bool(spec.get("available", True)),
        )
        for source_id, spec in raw.get("sources", {}).items()
    }
    links = tuple(
        LinkState(
            str(spec["source_node"]), str(spec["destination_node"]),
            float(spec.get("latency_ms", 0)),
            None if spec.get("bandwidth_mbps") is None else float(spec["bandwidth_mbps"]),
            bool(spec.get("available", True)),
        )
        for spec in raw.get("links", ())
    )
    endpoints = {
        str(node_id): (str(spec["agent_host"]), int(spec.get("agent_port", 8765)))
        for node_id, spec in raw.get("nodes", {}).items()
        if "agent_host" in spec
    }
    profile_path = raw.get("provider_profiles")
    if profile_path is not None:
        profile_path = Path(str(profile_path))
        if not profile_path.is_absolute():
            profile_path = (args.deployment.parent / profile_path).resolve()
    runtime_state = RuntimeState(
        nodes=nodes, sources=sources, links=links, profiles=load_provider_profiles(profile_path)
    )
    transport_kind = str(raw.get("transport", "tcp")).lower()
    mqtt_clients = []
    if transport_kind == "mqtt":
        mqtt_raw = raw.get("mqtt", {})
        command_client = PahoMQTTClient(
            str(mqtt_raw.get("host", "localhost")), int(mqtt_raw.get("port", 1883)),
            "fable-controller-commands",
        )
        command_transport = MqttCommandTransport(
            command_client, client_id="controller", timeout_s=args.command_timeout_seconds
        )
        command_transport.start_service()
        mqtt_clients.append(command_client)
    else:
        command_transport = TcpCommandTransport(endpoints, timeout_s=args.command_timeout_seconds)
    runtime = FableRuntime(
        event, runtime_state, command_transport, frontier_mode=args.frontier_mode
    )
    writer = CompletionWriter(args.output_jsonl) if args.output_jsonl else None

    def publish_visualization_state() -> None:
        if args.visualization_state_file is None:
            return
        state = runtime.visualization_state()
        replay_barrier = os.environ.get("FABLE_REPLAY_START_BARRIER_FILE")
        state["session"] = {
            "status": (
                "waiting_for_replay"
                if replay_barrier and not Path(replay_barrier).exists()
                else "replay_started"
            ),
            "name": args.visualization_session_name or event.name,
        }
        path = args.visualization_state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    def on_match(match):
        print(json.dumps({
            "type": "predicate_match_received",
            "received_at": datetime.now(timezone.utc).isoformat(),
            "match": match.to_dict(),
        }, sort_keys=True), flush=True)
        update = runtime.handle_predicate_match(match)
        print(json.dumps({
            "type": "predicate_match_outcome",
            "predicate": match.predicate,
            "event_time": match.event_time.isoformat(),
            "produced_instances": len(update.produced_instances),
            "completed_instances": len(update.completed_instances),
        }, sort_keys=True), flush=True)
        print(json.dumps(plan_row("predicate_match", update), sort_keys=True), flush=True)
        for instance in update.completed_instances:
            artifact = CompletionArtifact(
                event=instance.event_name,
                completed_at=instance.completed_at or match.event_time,
                matched_at=instance.matched_at,
                matched_source=instance.matched_source,
                bindings=dict(instance.bindings),
                cell_id=args.cell_id,
            )
            if writer is not None:
                writer.append(artifact)
            print(json.dumps(artifact.to_dict(), sort_keys=True), flush=True)
        publish_visualization_state()

    def on_identity(association):
        runtime.handle_identity_association(association)
        publish_visualization_state()

    def plan_row(reason: str, update) -> dict:
        return {
            "type": "execution_plan", "reason": reason,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "plan_id": str(uuid.uuid4()),
            "steps": [
                {"provider_id": step.provider_id, "node_id": step.node_id,
                 "source_ids": list(step.source_ids), "output_type": step.output_type}
                for step in update.plan.steps
            ],
            "predicted_completion_ms": update.plan.predicted_completion_ms,
            "transfer_bytes": update.plan.transfer_bytes,
            "frontier_mode": args.frontier_mode,
            "actions": {
                "start": [
                    {"provider_id": spec.key.provider_id, "node_id": spec.key.node_id,
                     "source_ids": list(spec.key.source_ids)}
                    for spec in update.actions.start
                ],
                "keep": [
                    {"provider_id": spec.key.provider_id, "node_id": spec.key.node_id,
                     "source_ids": list(spec.key.source_ids)}
                    for spec in update.actions.keep
                ],
                "stop": [
                    {"provider_id": key.provider_id, "node_id": key.node_id,
                     "source_ids": list(key.source_ids)}
                    for key in update.actions.stop
                ],
            },
        }

    if transport_kind == "mqtt":
        result_client = PahoMQTTClient(
            str(mqtt_raw.get("host", "localhost")), int(mqtt_raw.get("port", 1883)),
            "fable-controller-results",
        )
        result_service = ControllerMQTTResultService(result_client, on_match, on_identity)
        processed_link_requests: dict[str, dict] = {}
        def on_link_state(_topic: str, payload: bytes) -> None:
            request = json.loads(payload)
            request_id = str(request.get("request_id", ""))
            if request_id in processed_link_requests:
                result_client.publish(
                    str(request.get("reply_to", "fable/v1/runtime/link/ack")),
                    json.dumps(processed_link_requests[request_id], sort_keys=True).encode(), qos=1,
                )
                return
            try:
                changed = tuple(LinkState(
                    str(row["source_node"]), str(row["destination_node"]),
                    float(row.get("latency_ms", 0)),
                    None if row.get("bandwidth_mbps") is None else float(row["bandwidth_mbps"]),
                    bool(row.get("available", True)),
                ) for row in request.get("links", ()))
                update = runtime.handle_link_states(changed, now=datetime.now(timezone.utc))
                publish_visualization_state()
                document = plan_row("network_condition", update)
                print(json.dumps(document, sort_keys=True), flush=True)
                reply = {"ok": True, "request_id": request_id, "plan": document}
            except Exception as exc:
                reply = {"ok": False, "request_id": request_id, "error": str(exc)}
            if request_id:
                processed_link_requests[request_id] = reply
            result_client.publish(
                str(request.get("reply_to", "fable/v1/runtime/link/ack")),
                json.dumps(reply, sort_keys=True).encode(), qos=1,
            )
        result_client.subscribe("fable/v1/runtime/link", on_link_state, qos=1)
        result_client.start()
        mqtt_clients.append(result_client)
        result_server = None
    else:
        result_server = ResultTCPServer(on_match, on_identity, args.result_host, args.result_port)
        result_server.start_background()
    visualization_server = None
    if args.visualization_port:
        visualization_server = VisualizationServer(
            runtime.visualization_state,
            args.visualization_host,
            args.visualization_port,
        )
        visualization_server.start_background()
    # Publish the authored CE before waiting for infrastructure/replay release.
    # The persistent dashboard can therefore explain what the current session
    # is looking for even when no predicate result has arrived yet.
    publish_visualization_state()
    start_barrier = os.environ.get("FABLE_CONTROLLER_START_BARRIER_FILE")
    if start_barrier:
        print(f"Waiting for controller start barrier: {start_barrier}", file=sys.stderr, flush=True)
        while not Path(start_barrier).exists():
            time.sleep(0.05)
    try:
        initial = runtime.start(datetime.now(timezone.utc))
        publish_visualization_state()
        print(json.dumps(plan_row("initial", initial), sort_keys=True), flush=True)
        print(
            "FABLE running; result receiver using " + (
                "MQTT" if result_server is None else f"TCP {result_server.address[0]}:{result_server.address[1]}"
            ),
            file=sys.stderr,
        )
        if visualization_server is not None:
            print(
                f"CE progress dashboard listening on "
                f"http://{visualization_server.address[0]}:{visualization_server.address[1]}",
                file=sys.stderr,
            )
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.shutdown()
        if result_server is not None:
            result_server.close()
        for client in mqtt_clients:
            client.stop()
        if visualization_server is not None:
            visualization_server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
