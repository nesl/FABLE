#!/usr/bin/env python3
"""Run a FABLE node agent.

The default ``runtime: dataflow`` mode executes recovered provider workers in
this node-agent process and connects them through the same-node StreamBus.
Terminal PredicateMatch/ReID results are sent asynchronously to the controller.

Example:

node:
  node_id: edge1
  node_type: edge
runtime: dataflow
controller_results:
  host: 10.0.0.10
  port: 8766
sources:
  camera3:
    type: video
    uri: rtsp://camera3/stream

For legacy/external provider commands, ``runtime: subprocess`` remains
available and uses the ``providers: <id>: {command: [...]}`` mapping.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import yaml

from fable.execution import (
    DataflowProviderRuntime,
    NodeAgent,
    NodeAgentTCPServer,
    OpenCVVideoSourceAdapter,
    SubprocessProviderRuntime,
    SystemResourceProbe,
    TcpResultTransport,
    WaveAudioSourceAdapter,
)
from fable.planning import NodeState


def _sources(raw: dict) -> dict:
    output = {}
    for source_id, spec in raw.get("sources", {}).items():
        kind = str(spec.get("type", "video")).lower()
        if kind == "video":
            output[str(source_id)] = OpenCVVideoSourceAdapter(
                str(source_id), spec["uri"], realtime=bool(spec.get("realtime", True))
            )
        elif kind in {"wav", "audio"}:
            output[str(source_id)] = WaveAudioSourceAdapter(
                str(source_id), spec["path"],
                window_ms=int(spec.get("window_ms", 1000)),
                realtime=bool(spec.get("realtime", True)),
            )
        else:
            raise ValueError(f"unsupported source adapter type {kind!r}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    node_raw = raw["node"]
    node = NodeState(
        str(node_raw["node_id"]),
        str(node_raw.get("node_type", "unknown")),
        bool(node_raw.get("available", True)),
        float(node_raw.get("cpu_free", float("inf"))),
        float(node_raw.get("memory_mb_free", float("inf"))),
        float(node_raw.get("gpu_memory_mb_free", float("inf"))),
    )

    runtime_kind = str(raw.get("runtime", "dataflow")).lower()
    if runtime_kind == "dataflow":
        result_raw = raw.get("controller_results")
        if not isinstance(result_raw, dict):
            raise ValueError("dataflow node agents require controller_results.host/port")
        provider_runtime = DataflowProviderRuntime(
            result_transport=TcpResultTransport(
                str(result_raw["host"]), int(result_raw.get("port", 8766))
            ),
            source_adapters=_sources(raw),
        )
    elif runtime_kind == "subprocess":
        commands = {
            str(provider_id): tuple(spec["command"])
            for provider_id, spec in raw.get("providers", {}).items()
            if isinstance(spec, dict) and "command" in spec
        }
        provider_runtime = SubprocessProviderRuntime(commands)
    else:
        raise ValueError("runtime must be 'dataflow' or 'subprocess'")

    resource_probe = (
        SystemResourceProbe(node.node_id, node.node_type)
        if raw.get("system_probe", True)
        else None
    )
    agent = NodeAgent(node, provider_runtime, resource_probe=resource_probe)
    server = NodeAgentTCPServer(agent, args.host, args.port)
    print(f"FABLE node agent {node.node_id} listening on {server.address[0]}:{server.address[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
