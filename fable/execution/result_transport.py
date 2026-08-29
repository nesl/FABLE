"""Provider-result transport from node workers back to the FABLE controller.

Lifecycle commands and provider results intentionally use separate transports:
``command_transport.py`` controls START/STOP/STATUS, while this module carries
small semantic terminal results asynchronously back to the controller.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import socket
import socketserver
import threading
from typing import Callable, Protocol

from fable.providers.identity import IdentityAssociation
from fable.providers.predicate_result import PredicateMatch


class ResultTransport(Protocol):
    def send_predicate_match(self, match: PredicateMatch) -> None: ...
    def send_identity_association(self, association: IdentityAssociation) -> None: ...


class DirectResultTransport:
    """Direct callbacks used by tests and single-process deployments."""

    def __init__(
        self,
        on_predicate_match: Callable[[PredicateMatch], object],
        on_identity_association: Callable[[IdentityAssociation], object] | None = None,
    ) -> None:
        self.on_predicate_match = on_predicate_match
        self.on_identity_association = on_identity_association

    def send_predicate_match(self, match: PredicateMatch) -> None:
        self.on_predicate_match(match)

    def send_identity_association(self, association: IdentityAssociation) -> None:
        if self.on_identity_association is not None:
            self.on_identity_association(association)


class TcpResultTransport:
    """Small newline-delimited JSON result sender."""

    def __init__(self, host: str, port: int, *, timeout_s: float = 5.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    def send_predicate_match(self, match: PredicateMatch) -> None:
        self._send({"type": "predicate_match", "value": match.to_dict()})

    def send_identity_association(self, association: IdentityAssociation) -> None:
        self._send({"type": "identity_association", "value": asdict(association)})

    def _send(self, payload: dict) -> None:
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as sock:
            sock.sendall(raw)
            response = b""
            while not response.endswith(b"\n"):
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
        if response:
            decoded = json.loads(response.decode("utf-8"))
            if not decoded.get("ok", False):
                raise RuntimeError(str(decoded.get("message", "result receiver rejected message")))


class _ResultRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = self.server
        try:
            raw = json.loads(self.rfile.readline().decode("utf-8"))
            kind = raw.get("type")
            value = raw.get("value", {})
            if kind == "predicate_match":
                server.on_predicate_match(PredicateMatch.from_dict(value))  # type: ignore[attr-defined]
            elif kind == "identity_association":
                server.on_identity_association(  # type: ignore[attr-defined]
                    IdentityAssociation(
                        str(value["left_object_id"]),
                        str(value["right_object_id"]),
                        float(value.get("cosine_similarity", 1.0)),
                    )
                )
            else:
                raise ValueError(f"unknown result message type {kind!r}")
            response = {"ok": True}
        except Exception as exc:
            response = {"ok": False, "message": str(exc)}
        self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))


class _ThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class ResultTCPServer:
    """Controller-side asynchronous terminal-result receiver."""

    def __init__(
        self,
        on_predicate_match: Callable[[PredicateMatch], object],
        on_identity_association: Callable[[IdentityAssociation], object],
        host: str = "0.0.0.0",
        port: int = 0,
    ) -> None:
        self.server = _ThreadingTCPServer((host, int(port)), _ResultRequestHandler)
        self.server.on_predicate_match = on_predicate_match  # type: ignore[attr-defined]
        self.server.on_identity_association = on_identity_association  # type: ignore[attr-defined]
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
