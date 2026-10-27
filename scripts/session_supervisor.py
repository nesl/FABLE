#!/usr/bin/env python3
"""Run one replaceable experiment process inside a long-lived container."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def _send(path: Path, request: dict) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(path))
        client.sendall((json.dumps(request) + "\n").encode())
        payload = bytearray()
        while b"\n" not in payload:
            chunk = client.recv(65536)
            if not chunk:
                break
            payload.extend(chunk)
    return json.loads(payload.split(b"\n", 1)[0])


def serve(path: Path) -> int:
    path.unlink(missing_ok=True)
    process: subprocess.Popen | None = None
    log_handle = None

    def stop() -> None:
        nonlocal process, log_handle
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        process = None
        if log_handle is not None:
            log_handle.close()
            log_handle = None

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        server.listen(4)
        try:
            while True:
                connection, _ = server.accept()
                with connection:
                    request = json.loads(connection.makefile().readline())
                    action = request.get("action")
                    if action == "start":
                        stop()
                        try:
                            log_path = Path(request["log"])
                            log_path.parent.mkdir(parents=True, exist_ok=True)
                            log_handle = log_path.open("w", encoding="utf-8")
                            environment = os.environ.copy()
                            environment.update({str(k): str(v) for k, v in request.get("env", {}).items()})
                            process = subprocess.Popen(
                                [str(value) for value in request["argv"]],
                                cwd=request.get("cwd", "/workspace/FABLE"),
                                env=environment,
                                stdout=log_handle,
                                stderr=subprocess.STDOUT,
                                text=True,
                            )
                            response = {"ok": True, "pid": process.pid}
                        except Exception as exc:
                            stop()
                            response = {"ok": False, "error": str(exc)}
                    elif action == "stop":
                        stop()
                        response = {"ok": True}
                    elif action == "status":
                        response = {
                            "ok": True,
                            "running": process is not None and process.poll() is None,
                            "pid": None if process is None else process.pid,
                            "returncode": None if process is None else process.poll(),
                        }
                    else:
                        response = {"ok": False, "error": f"unknown action: {action}"}
                    connection.sendall((json.dumps(response) + "\n").encode())
        except KeyboardInterrupt:
            pass
        finally:
            stop()
            path.unlink(missing_ok=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("serve", "start", "stop", "status"))
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if args.action == "serve":
        return serve(args.socket)
    request = {"action": args.action}
    if args.request:
        request.update(json.loads(args.request.read_text(encoding="utf-8")))
    response = _send(args.socket, request)
    print(json.dumps(response, sort_keys=True))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
