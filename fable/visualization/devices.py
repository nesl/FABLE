"""Thread-safe live camera state assembled from MQTT replay messages."""
from __future__ import annotations

import base64
import json
import threading
import time
from typing import Mapping
from math import ceil, sqrt


class LiveDeviceMonitor:
    def __init__(self) -> None:
        self._devices: dict[str, dict] = {}
        self._lock = threading.Lock()

    def subscribe(self, client) -> None:
        client.subscribe("fable/v1/source/video/+", self.on_frame, qos=0)
        client.subscribe("fable/v1/monitor/vision/+", self.on_metadata, qos=0)

    def on_frame(self, topic: str, payload: bytes) -> None:
        row = json.loads(payload)
        source_id = str(row["source_id"])
        jpeg = base64.b64decode(row["jpeg"])
        event_time = str(row.get("event_time", ""))
        with self._lock:
            device = self._devices.setdefault(source_id, {})
            # The visualization service is intentionally long-lived while
            # replay sessions rewind to older event timestamps.  Annotations
            # from the preceding session must never be painted on that new
            # session's frames.
            if event_time != str(device.get("event_time", "")):
                device["metadata_by_kind"] = {}
            device.update(source_id=source_id, jpeg=jpeg,
                          event_time=event_time,
                          frame_id=str(row.get("frame_id", "")),
                          frame_origin=str(row.get("origin_node", "")),
                          received_at=time.time())
            device["version"] = int(device.get("version", 0)) + 1

    def on_metadata(self, topic: str, payload: bytes) -> None:
        row = json.loads(payload)
        source_id = str(row["source_id"])
        with self._lock:
            device = self._devices.setdefault(source_id, {"source_id": source_id})
            # Detector output is asynchronous.  Retain it only when it belongs
            # to the frame currently displayed; otherwise it is either late or
            # stale state from an earlier replay of the same source.
            if str(row.get("event_time", "")) != str(device.get("event_time", "")):
                return
            by_kind = device.setdefault("metadata_by_kind", {})
            current = by_kind.get(str(row.get("kind", "")), {})
            if str(row.get("event_time", "")) >= str(current.get("event_time", "")):
                by_kind[str(row.get("kind", ""))] = row
                device["version"] = int(device.get("version", 0)) + 1

    def devices(self) -> list[Mapping[str, object]]:
        now = time.time()
        with self._lock:
            rows = []
            for source_id, value in sorted(self._devices.items()):
                by_kind = value.get("metadata_by_kind", {})
                tracks = by_kind.get("tracks", {})
                detections = by_kind.get("detections", {})
                rows.append({
                    "source_id": source_id,
                    "event_time": value.get("event_time", ""),
                    "frame_id": value.get("frame_id", ""),
                    "origin_node": tracks.get("origin_node", detections.get("origin_node", value.get("frame_origin", ""))),
                    "annotation_kind": "detections + tracks" if detections and tracks else ("tracks" if tracks else ("detections" if detections else "raw")),
                    "annotation_count": len(tracks.get("items", ())) or len(detections.get("items", ())),
                    "detection_count": len(detections.get("items", ())),
                    "track_count": len(tracks.get("items", ())),
                    "age_seconds": round(max(0.0, now - value.get("received_at", now)), 1),
                    "has_frame": "jpeg" in value,
                })
            return rows

    def frame(self, source_id: str, max_width: int | None = None) -> bytes | None:
        with self._lock:
            value = self._devices.get(source_id)
            if not value or "jpeg" not in value:
                return None
            jpeg = value["jpeg"]
            metadata_rows = [dict(row) for row in value.get("metadata_by_kind", {}).values()]
        try:
            import cv2
            import numpy as np
            image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return jpeg
            # Draw YOLO detections first in orange, then stable tracks in green.
            for metadata in sorted(metadata_rows, key=lambda row: row.get("kind") == "tracks"):
                color = (40, 220, 120) if metadata.get("kind") == "tracks" else (50, 180, 255)
                for item in metadata.get("items", ()):
                    box = item["bbox"]
                    p1 = (int(box["x1"]), int(box["y1"]))
                    p2 = (int(box["x2"]), int(box["y2"]))
                    cv2.rectangle(image, p1, p2, color, 2)
                    identity = item.get("id")
                    label = ("YOLO " if metadata.get("kind") == "detections" else "Track ") + str(item.get("label", "object"))
                    if identity:
                        label += f" {identity}"
                    confidence = item.get("confidence")
                    if confidence is not None:
                        label += f" {float(confidence):.2f}"
                    label_y = max(18, p1[1] - (24 if metadata.get("kind") == "tracks" else 6))
                    cv2.putText(image, label, (p1[0], label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
            if max_width and image.shape[1] > max_width:
                scale = max_width / image.shape[1]
                image = cv2.resize(image, (max_width, max(1, int(image.shape[0] * scale))))
            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            return encoded.tobytes() if ok else jpeg
        except Exception:
            return jpeg

    def snapshot(self, tile_width: int = 480) -> bytes | None:
        """Return an equal-tile contact sheet of the latest device frames."""
        try:
            import cv2
            import numpy as np
            rows = self.devices()
            images = []
            tile_height = int(tile_width * 9 / 16) + 36
            for row in rows:
                encoded = self.frame(str(row["source_id"]), tile_width)
                if encoded is None: continue
                image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None: continue
                view_height = tile_height - 36
                canvas = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
                scale = min(tile_width / image.shape[1], view_height / image.shape[0])
                resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
                x = (tile_width - resized.shape[1]) // 2
                canvas[36:36 + resized.shape[0], x:x + resized.shape[1]] = resized
                label = f'{row["source_id"]}  {row["annotation_kind"]}: {row["annotation_count"]}'
                cv2.putText(canvas, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            .58, (235, 240, 250), 1, cv2.LINE_AA)
                images.append(canvas)
            if not images: return None
            columns = max(1, ceil(sqrt(len(images))))
            row_count = ceil(len(images) / columns)
            sheet = np.zeros((row_count * tile_height, columns * tile_width, 3), dtype=np.uint8)
            for index, image in enumerate(images):
                y, x = divmod(index, columns)
                sheet[y * tile_height:(y + 1) * tile_height,
                      x * tile_width:(x + 1) * tile_width] = image
            ok, encoded = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            return encoded.tobytes() if ok else None
        except Exception:
            return None
