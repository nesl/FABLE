"""Serve the FABLE CE-progress dashboard.

Adapted from mobile_data_collector/event-detector/services/visualization.py.
Unlike that service, state is read directly from FABLE on each GET and is never
posted, queued, or persisted here.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Callable, Mapping
from urllib.parse import parse_qs, unquote, urlparse


HTML = Path(__file__).with_name("dashboard.html").read_bytes()
DEVICES_HTML = Path(__file__).with_name("devices.html").read_bytes()


class VisualizationServer:
    def __init__(
        self,
        state_provider: Callable[[], Mapping[str, object]],
        host: str = "127.0.0.1",
        port: int = 5000,
        device_provider=None,
    ) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(handler) -> None:
                parsed = urlparse(handler.path)
                path = parsed.path
                if path == "/":
                    handler._send(HTML, "text/html; charset=utf-8")
                elif path == "/devices":
                    handler._send(DEVICES_HTML, "text/html; charset=utf-8")
                elif path == "/api/state":
                    handler._send(
                        json.dumps(state_provider(), sort_keys=True).encode("utf-8"),
                        "application/json",
                    )
                elif path == "/api/dashboard-snapshot.jpg":
                    frame = _dashboard_snapshot(state_provider())
                    if frame is None:
                        handler.send_error(503)
                    else:
                        handler._send(frame, "image/jpeg")
                elif path == "/api/devices" and device_provider is not None:
                    handler._send(json.dumps(device_provider.devices()).encode("utf-8"), "application/json")
                elif path == "/api/devices-snapshot.jpg" and device_provider is not None:
                    frame = device_provider.snapshot()
                    if frame is None:
                        handler.send_error(404)
                    else:
                        handler._send(frame, "image/jpeg")
                elif path.startswith("/api/devices/") and path.endswith("/frame.jpg") and device_provider is not None:
                    source_id = unquote(path[len("/api/devices/"):-len("/frame.jpg")]).strip("/")
                    requested_width = parse_qs(parsed.query).get("width", ["640"])[0]
                    try:
                        max_width = min(1920, max(160, int(requested_width)))
                    except ValueError:
                        max_width = 640
                    frame = device_provider.frame(source_id, max_width)
                    if frame is None:
                        handler.send_error(404)
                    else:
                        handler._send(frame, "image/jpeg")
                else:
                    handler.send_error(404)

            def _send(handler, body: bytes, content_type: str) -> None:
                handler.send_response(200)
                handler.send_header("Content-Type", content_type)
                handler.send_header("Cache-Control", "no-store")
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)

            def log_message(handler, format, *args) -> None:
                if args and str(args[1]) != "200":
                    super().log_message(format, *args)

        self._server = ThreadingHTTPServer((host, int(port)), Handler)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start_background(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


def _dashboard_snapshot(state: Mapping[str, object]) -> bytes | None:
    """Render the CE progress view for headless experiment recording."""
    try:
        import cv2
        import numpy as np

        image = np.full((720, 1280, 3), (248, 249, 251), dtype=np.uint8)
        event = dict(state.get("event", {}))
        counts = dict(state.get("counts", {}))
        session = dict(state.get("session", {}))
        instances = list(state.get("instances", ()))
        lead_id = state.get("lead_instance_id")
        lead = next((dict(row) for row in instances if row.get("instance_id") == lead_id), None)
        labels = {
            "waiting_for_replay": "Waiting for replay", "idle": "Waiting for replay",
            "replay_started": "Replay started", "running": "Replay started",
            "replay_completed": "Replay completed", "completed": "Replay completed",
            "replay_failed": "Replay failed", "failed": "Replay failed",
        }
        status = labels.get(str(session.get("status", "waiting_for_replay")), "Waiting for replay")

        def put(text, xy, scale=.65, color=(35, 35, 35), thickness=1):
            cv2.putText(image, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, thickness, cv2.LINE_AA)

        name = str(event.get("name", "FABLE CE Progress")).replace("_", " ").title()
        put(name, (45, 55), 1.05, thickness=2)
        put(status, (45, 102), .72, (128, 65, 20), 2)
        put(
            f'{counts.get("total", len(instances))} total   {counts.get("active", 0)} active   '
            f'{counts.get("completed", 0)} completed   {counts.get("expired", 0)} expired',
            (700, 58), .62,
        )
        cv2.line(image, (40, 128), (1240, 128), (205, 208, 215), 2)
        put("Complex-event stages", (45, 175), .78, thickness=2)

        stages = list(event.get("stages", ()))
        satisfied = set((lead or {}).get("satisfied_paths", ()))
        active = set((lead or {}).get("active_paths", ()))
        failed = dict((lead or {}).get("failed_paths", {}))
        if lead:
            put(f'Selected: {lead.get("instance_id")} ({lead.get("status")})', (45, 215), .57, (80, 80, 80))
        else:
            put("No instance yet; showing the authored event structure", (45, 215), .57, (95, 95, 95))
        count = max(1, len(stages))
        usable = 1170
        gap = 34
        width = min(205, max(120, int((usable - gap * (count - 1)) / count)))
        total_width = width * count + gap * (count - 1)
        start_x = max(45, (1280 - total_width) // 2)
        y, height = 285, 150
        for index, stage in enumerate(stages):
            stage = dict(stage)
            leaves = list(stage.get("leaf_paths", ()))
            path = str(stage.get("path", ""))
            if path in failed or any(value in failed for value in leaves):
                color, fill, state_label = (60, 60, 180), (220, 220, 255), "FAILED"
            elif path in satisfied or (leaves and all(value in satisfied for value in leaves)):
                color, fill, state_label = (55, 145, 55), (220, 248, 220), "COMPLETED"
            elif path in active or any(value in active or value in satisfied for value in leaves):
                color, fill, state_label = (45, 90, 205), (220, 232, 255), "ACTIVE"
            else:
                color, fill, state_label = (115, 115, 115), (245, 245, 245), "WAITING"
            x = start_x + index * (width + gap)
            cv2.rectangle(image, (x, y), (x + width, y + height), fill, -1)
            cv2.rectangle(image, (x, y), (x + width, y + height), color, 3)
            words = str(stage.get("label", "Stage")).split()
            lines, line = [], ""
            for word in words:
                trial = (line + " " + word).strip()
                if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, .55, 1)[0][0] > width - 18 and line:
                    lines.append(line); line = word
                else: line = trial
            if line: lines.append(line)
            for line_index, value in enumerate(lines[:3]):
                put(value, (x + 10, y + 48 + line_index * 25), .55, (35, 35, 35))
            put(state_label, (x + 10, y + height - 18), .46, color, 2)
            if index + 1 < count:
                ax = x + width
                cv2.arrowedLine(image, (ax + 5, y + height // 2),
                                (ax + gap - 5, y + height // 2), (105, 105, 105), 2)
        put("Latest instance activity", (45, 510), .78, thickness=2)
        for index, row in enumerate(instances[:5]):
            row = dict(row)
            progress = dict(row.get("progress", {}))
            put(
                f'{row.get("instance_id")}   {str(row.get("status", "")).upper()}   '
                f'{progress.get("satisfied", 0)}/{progress.get("total", 0)} conditions',
                (60, 552 + index * 31), .56,
                (35, 120, 45) if row.get("status") == "completed" else (55, 55, 55),
            )
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return encoded.tobytes() if ok else None
    except Exception:
        return None
