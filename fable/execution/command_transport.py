"""Command transports for controller -> node-agent lifecycle actions.

The standard-library TCP implementation is intentionally tiny and synchronous:
one JSON request receives one JSON acknowledgement.  It is sufficient for the
START/STOP/STATUS control plane without forcing a particular message broker.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import socket
import socketserver
import threading
from typing import Mapping, Protocol

from .node_agent import CommandResult, NodeAgent, NodeStatus
from .plan_reconciler import ProviderInstanceKey, ProviderInstanceSpec


class CommandTransport(Protocol):
    def start(self, spec: ProviderInstanceSpec) -> CommandResult: ...
    def stop(self, key: ProviderInstanceKey) -> CommandResult: ...
    def status(self, node_id: str) -> NodeStatus: ...


class DirectCommandTransport:
    """Direct calls to in-process NodeAgents; useful locally and in tests."""

    def __init__(self, agents: Mapping[str, NodeAgent]) -> None:
        self.agents = dict(agents)

    def _agent(self, node_id: str) -> NodeAgent:
        try:
            return self.agents[node_id]
        except KeyError as exc:
            raise RuntimeError(f"no node agent registered for {node_id!r}") from exc

    def start(self, spec: ProviderInstanceSpec) -> CommandResult:
        return self._agent(spec.key.node_id).start(spec)

    def stop(self, key: ProviderInstanceKey) -> CommandResult:
        return self._agent(key.node_id).stop(key)

    def status(self, node_id: str) -> NodeStatus:
        return self._agent(node_id).status()


class TcpCommandTransport:
    def __init__(
        self,
        endpoints: Mapping[str, tuple[str, int]],
        *,
        timeout_s: float = 5.0,
    ) -> None:
        self.endpoints = dict(endpoints)
        self.timeout_s = timeout_s

    def start(self, spec: ProviderInstanceSpec) -> CommandResult:
        response = self._request(spec.key.node_id, {
            "action": "start",
            "spec": _spec_to_dict(spec),
        })
        return CommandResult(bool(response.get("ok")), str(response.get("message", "")))

    def stop(self, key: ProviderInstanceKey) -> CommandResult:
        response = self._request(key.node_id, {"action": "stop", "key": _key_to_dict(key)})
        return CommandResult(bool(response.get("ok")), str(response.get("message", "")))

    def status(self, node_id: str) -> NodeStatus:
        response = self._request(node_id, {"action": "status"})
        if not response.get("ok"):
            raise RuntimeError(str(response.get("message", "node status failed")))
        return _status_from_dict(response["status"])

    def _request(self, node_id: str, request: dict) -> dict:
        try:
            host, port = self.endpoints[node_id]
        except KeyError as exc:
            raise RuntimeError(f"no TCP endpoint configured for node {node_id!r}") from exc
        payload = (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")
        with socket.create_connection((host, int(port)), timeout=self.timeout_s) as sock:
            sock.sendall(payload)
            with sock.makefile("rb") as reader:
                line = reader.readline()
        if not line:
            raise RuntimeError(f"node {node_id!r} closed connection without an acknowledgement")
        return json.loads(line.decode("utf-8"))


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = self.server
        agent: NodeAgent = server.agent  # type: ignore[attr-defined]
        try:
            raw = json.loads(self.rfile.readline().decode("utf-8"))
            action = raw.get("action")
            if action == "start":
                result = agent.start(_spec_from_dict(raw["spec"]))
                response = {"ok": result.ok, "message": result.message}
            elif action == "stop":
                result = agent.stop(_key_from_dict(raw["key"]))
                response = {"ok": result.ok, "message": result.message}
            elif action == "status":
                response = {"ok": True, "status": _status_to_dict(agent.status())}
            else:
                response = {"ok": False, "message": f"unknown action {action!r}"}
        except Exception as exc:
            response = {"ok": False, "message": str(exc)}
        self.wfile.write((json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8"))


class NodeAgentTCPServer:
    """Small TCP server wrapper suitable for ``scripts/run_node_agent.py``."""

    def __init__(self, agent: NodeAgent, host: str = "0.0.0.0", port: int = 8765) -> None:
        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server((host, port), _Handler)
        self.server.agent = agent  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.server.server_address[:2]
        return str(host), int(port)

    def serve_forever(self) -> None:
        self.server.serve_forever()

    def start_background(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


def _key_to_dict(key: ProviderInstanceKey) -> dict:
    return {"provider_id": key.provider_id, "node_id": key.node_id, "source_ids": list(key.source_ids)}


def _key_from_dict(raw: Mapping) -> ProviderInstanceKey:
    return ProviderInstanceKey(str(raw["provider_id"]), str(raw["node_id"]), tuple(raw.get("source_ids", ())))


def _spec_to_dict(spec: ProviderInstanceSpec) -> dict:
    return {"key": _key_to_dict(spec.key), "output_type": spec.output_type}


def _spec_from_dict(raw: Mapping) -> ProviderInstanceSpec:
    return ProviderInstanceSpec(_key_from_dict(raw["key"]), str(raw.get("output_type", "")))


def _status_to_dict(status: NodeStatus) -> dict:
    return {
        "node_id": status.node_id,
        "node_type": status.node_type,
        "available": status.available,
        "cpu_free": status.cpu_free,
        "memory_mb_free": status.memory_mb_free,
        "gpu_memory_mb_free": status.gpu_memory_mb_free,
        "running": [
            {"provider_id": row.provider_id, "node_id": row.node_id, "source_ids": list(row.source_ids)}
            for row in status.running
        ],
    }


def _status_from_dict(raw: Mapping) -> NodeStatus:
    from fable.planning.runtime_state import RunningProvider
    return NodeStatus(
        str(raw["node_id"]), str(raw["node_type"]), bool(raw["available"]),
        float(raw["cpu_free"]), float(raw["memory_mb_free"]), float(raw["gpu_memory_mb_free"]),
        tuple(
            RunningProvider(str(row["provider_id"]), str(row["node_id"]), tuple(row.get("source_ids", ())))
            for row in raw.get("running", ())
        ),
    )
