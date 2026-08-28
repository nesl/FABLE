"""Dedicated cloud-side hosted-VLM proxy with per-run invocation bounds."""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import socket
import socketserver
from threading import Lock
from typing import Any, Callable

from .vlm_reid import OpenAIVisionIdentityComparator


ALLOWED_KEYS = {
    "schema_version",
    "run_id",
    "invocation_id",
    "entity_kind",
    "left_image_url",
    "right_image_url",
    "left_local_entity_id",
    "right_local_entity_id",
}


class HostedVlmProxy:
    def __init__(
        self,
        comparator: Any,
        *,
        maximum_calls_per_run: int = 10,
        debug_directory: Path | None = None,
        readiness_check: Callable[[], tuple[bool, str]] | None = None,
    ) -> None:
        if not 1 <= maximum_calls_per_run <= 10:
            raise ValueError("hosted VLM proxy limit must be between one and ten")
        self.comparator = comparator
        self.maximum_calls_per_run = maximum_calls_per_run
        self.debug_directory = debug_directory
        self._readiness_check = readiness_check or (lambda: (True, "ready"))
        self._counts: dict[str, int] = {}
        self._seen: set[tuple[str, str]] = set()
        self._lock = Lock()

    def readiness(self) -> tuple[bool, str]:
        """Report whether the external inference dependency is reachable."""

        try:
            return self._readiness_check()
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def compare(self, document: dict[str, Any]) -> dict[str, Any]:
        unknown = set(document) - ALLOWED_KEYS
        if unknown:
            raise ValueError(f"unknown hosted VLM request fields: {sorted(unknown)}")
        if document.get("schema_version") != "fable.hosted_vlm_request.v1":
            raise ValueError("unsupported hosted VLM request schema")
        run_id = str(document.get("run_id") or "")
        invocation_id = str(document.get("invocation_id") or "")
        entity_kind = str(document.get("entity_kind") or "")
        if not run_id or not invocation_id or entity_kind not in {"person", "vehicle"}:
            raise ValueError("run, invocation, and entity kind are required")
        left = str(document.get("left_image_url") or "")
        right = str(document.get("right_image_url") or "")
        if not left.startswith("data:image/") or not right.startswith("data:image/"):
            raise ValueError("hosted VLM proxy accepts only inline image data URLs")
        with self._lock:
            key = (run_id, invocation_id)
            if key in self._seen:
                raise ValueError("duplicate hosted VLM invocation")
            count = self._counts.get(run_id, 0)
            if count >= self.maximum_calls_per_run:
                raise ValueError("hosted VLM invocation budget exhausted")
            self._seen.add(key)
            self._counts[run_id] = count + 1
        decision = self.comparator.compare(
            entity_kind=entity_kind,
            left_image_url=left,
            right_image_url=right,
        )
        response = {
            "schema_version": "fable.hosted_vlm_response.v1",
            "run_id": run_id,
            "invocation_id": invocation_id,
            "same_identity": decision.same_identity,
            "confidence": decision.confidence,
            "reason": decision.reason,
        }
        self._persist_debug_evidence(
            run_id=run_id,
            invocation_id=invocation_id,
            entity_kind=entity_kind,
            left=left,
            right=right,
            response=response,
            left_local_entity_id=str(document.get("left_local_entity_id") or ""),
            right_local_entity_id=str(document.get("right_local_entity_id") or ""),
        )
        return response

    def _persist_debug_evidence(
        self,
        *,
        run_id: str,
        invocation_id: str,
        entity_kind: str,
        left: str,
        right: str,
        response: dict[str, Any],
        left_local_entity_id: str,
        right_local_entity_id: str,
    ) -> None:
        if self.debug_directory is None:
            return
        safe_run = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)[:160]
        safe_call = re.sub(r"[^A-Za-z0-9_.-]", "_", invocation_id)[:200]
        target = self.debug_directory / safe_run
        target.mkdir(parents=True, exist_ok=True)
        for side, value in (("left", left), ("right", right)):
            header, encoded = value.split(",", 1)
            extension = "png" if "image/png" in header else "jpg"
            (target / f"{safe_call}-{side}.{extension}").write_bytes(
                base64.b64decode(encoded, validate=True)
            )
        metadata = {
            "schema_version": "fable.hosted_vlm_debug.v1",
            "run_id": run_id,
            "invocation_id": invocation_id,
            "entity_kind": entity_kind,
            "same_identity": response["same_identity"],
            "confidence": response["confidence"],
            "reason": response["reason"],
            "left_local_entity_id": left_local_entity_id,
            "right_local_entity_id": right_local_entity_id,
        }
        (target / f"{safe_call}-decision.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )


def handler_for(proxy: HostedVlmProxy):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/healthz":
                self.send_error(404)
                return
            ready, reason = proxy.readiness()
            self._send(200 if ready else 503, {"ready": ready, "reason": reason})

        def do_POST(self):
            if self.path != "/v1/compare":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 20 * 1024 * 1024:
                    raise ValueError("invalid hosted VLM request size")
                document = json.loads(self.rfile.read(length))
                if not isinstance(document, dict):
                    raise ValueError("hosted VLM request must be an object")
                self._send(200, proxy.compare(document))
            except Exception as exc:
                self._send(
                    400,
                    {
                        "schema_version": "fable.hosted_vlm_response.v1",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )

        def log_message(self, _format, *_args):
            return

        def _send(self, status: int, document: dict[str, Any]) -> None:
            payload = json.dumps(document, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


class ThreadingUnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--unix-socket")
    args = parser.parse_args()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        class UnavailableComparator:
            def compare(self, **_kwargs):
                raise RuntimeError(
                    "OPENAI_API_KEY is unavailable in hosted VLM proxy"
                )

        comparator: Any = UnavailableComparator()
    else:
        comparator = OpenAIVisionIdentityComparator(
            api_key=api_key,
            model=os.getenv(
                "FABLE_VLM_REID_MODEL",
                "gpt-4o-mini-2024-07-18",
            ),
        )
    def upstream_readiness() -> tuple[bool, str]:
        if not api_key:
            return False, "OPENAI_API_KEY is unavailable"
        try:
            socket.getaddrinfo("api.openai.com", 443, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            return False, f"OpenAI endpoint DNS resolution failed: {exc}"
        return True, "OpenAI endpoint DNS resolved"

    proxy = HostedVlmProxy(
        comparator,
        maximum_calls_per_run=int(os.getenv("FABLE_VLM_REID_MAX_CALLS", "10")),
        debug_directory=(
            Path(os.environ["FABLE_VLM_DEBUG_DIR"])
            if os.getenv("FABLE_VLM_DEBUG_DIR")
            else None
        ),
        readiness_check=upstream_readiness,
    )
    if args.unix_socket:
        socket_path = Path(args.unix_socket)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        socket_path.unlink(missing_ok=True)
        server = ThreadingUnixHTTPServer(str(socket_path), handler_for(proxy))
        socket_path.chmod(0o660)
    else:
        server = ThreadingHTTPServer((args.host, args.port), handler_for(proxy))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if args.unix_socket:
            Path(args.unix_socket).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
