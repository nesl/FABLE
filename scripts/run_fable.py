#!/usr/bin/env python3
"""Run the distributed FABLE controller with an asynchronous result receiver."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import yaml

from fable.execution import FableRuntime, ResultTCPServer, TcpCommandTransport
from fable.language import load_and_compile_event
from fable.planning import LinkState, NodeState, RuntimeState, SourceState, load_provider_profiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("event", type=Path)
    parser.add_argument("deployment", type=Path)
    parser.add_argument("--result-host", default="0.0.0.0")
    parser.add_argument("--result-port", type=int, default=8766)
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
    runtime_state = RuntimeState(
        nodes=nodes, sources=sources, links=links, profiles=load_provider_profiles()
    )
    runtime = FableRuntime(event, runtime_state, TcpCommandTransport(endpoints))

    def on_match(match):
        update = runtime.handle_predicate_match(match)
        for instance in update.completed_instances:
            print(json.dumps({
                "event": instance.event_name,
                "matched_at": instance.matched_at.isoformat(),
                "matched_source": instance.matched_source,
                "bindings": dict(instance.bindings),
                "completed": True,
            }), flush=True)

    def on_identity(association):
        runtime.handle_identity_association(association)

    result_server = ResultTCPServer(
        on_match, on_identity, args.result_host, args.result_port
    )
    result_server.start_background()
    try:
        runtime.start(datetime.now(timezone.utc))
        print(
            f"FABLE running; result receiver listening on "
            f"{result_server.address[0]}:{result_server.address[1]}",
            file=sys.stderr,
        )
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.shutdown()
        result_server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
