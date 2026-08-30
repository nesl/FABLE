#!/usr/bin/env python3
"""Python-3.8-compatible CUDA YOLO worker for the typed deployment bridge."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import signal
import sys
import time

import numpy as np

import cv2
import torch
import torchvision


def _torch_nms(boxes, scores, threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        index = order[0]
        keep.append(index)
        if order.numel() == 1:
            break
        remaining = order[1:]
        left_top = torch.maximum(boxes[index, :2], boxes[remaining, :2])
        right_bottom = torch.minimum(boxes[index, 2:], boxes[remaining, 2:])
        width_height = (right_bottom - left_top).clamp(min=0)
        intersection = width_height[:, 0] * width_height[:, 1]
        area_index = (boxes[index, 2] - boxes[index, 0]) * (boxes[index, 3] - boxes[index, 1])
        area_remaining = (boxes[remaining, 2] - boxes[remaining, 0]) * (boxes[remaining, 3] - boxes[remaining, 1])
        iou = intersection / (area_index + area_remaining - intersection + 1e-7)
        order = remaining[iou <= threshold]
    return torch.stack(keep)


torchvision.ops.nms = _torch_nms
from ultralytics import YOLO  # noqa: E402


def _emit(document):
    sys.stdout.write(json.dumps(document, sort_keys=True) + "\n")
    sys.stdout.flush()


class _VideoHysteresisGate:
    def __init__(self, on_threshold, off_threshold):
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.open = False
        self.previous = None

    def accept(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        gray = gray[::max(1, height // 64), ::max(1, width // 64)].astype(np.float32)
        previous, self.previous = self.previous, gray
        score = 0.0 if previous is None or previous.shape != gray.shape else float(np.mean(np.abs(gray - previous)) / 255.0)
        if self.open:
            if score <= self.off_threshold:
                self.open = False
        elif score >= self.on_threshold:
            self.open = True
        return self.open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--maximum-rate-hz", type=float, default=5.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    provider_id = os.environ.get("FABLE_PROVIDER_ID", "")
    source_ids = [row for row in os.environ.get("FABLE_SOURCE_IDS", "").split(",") if row]
    if provider_id not in {"yolo_vehicle_fast_640", "yolo_vehicle_balanced_960", "yolo_full_context_960"}:
        raise RuntimeError("worker received a non-allowlisted provider ID")
    if len(source_ids) != 1:
        raise RuntimeError("external YOLO requires exactly one source")
    gate = None
    gate_specs = json.loads(os.environ.get("FABLE_INPUT_GATES", "{}"))
    gate_spec = gate_specs.get(source_ids[0])
    if gate_spec is not None:
        if gate_spec.get("type") != "video_frame_difference":
            raise RuntimeError("external YOLO only supports the video frame-difference gate")
        gate = _VideoHysteresisGate(gate_spec["on_threshold"], gate_spec["off_threshold"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    image_size = 960 if provider_id != "yolo_vehicle_fast_640" else 640
    vehicle_only = provider_id != "yolo_full_context_960"
    model = YOLO(args.model)
    capture = cv2.VideoCapture(args.stream_url)
    if not capture.isOpened():
        raise RuntimeError("could not open configured stream")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    period = max(1.0, fps / max(args.maximum_rate_hz, 0.01))
    start = datetime.now(timezone.utc)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    _emit({"type": "ready", "provider_id": provider_id, "source_id": source_ids[0]})
    frame_number = 0
    next_sample = 1.0
    try:
        while not stop[0]:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if frame_number + 1e-9 < next_sample:
                continue
            while next_sample <= frame_number + 1e-9:
                next_sample += period
            if gate is not None and not gate.accept(frame):
                continue
            event_time = start + timedelta(seconds=(frame_number - 1) / fps)
            rows = []
            for result in model(frame, device=0, imgsz=image_size, conf=args.confidence, verbose=False):
                for index, box in enumerate(result.boxes):
                    label = str(result.names[int(box.cls.item())])
                    if vehicle_only and label not in {"car", "truck", "bus", "motorcycle"}:
                        continue
                    coords = [float(value) for value in box.xyxy[0].cpu().tolist()]
                    rows.append({
                        "class_name": label,
                        "confidence": float(box.conf.item()),
                        "bbox": coords,
                        "detection_id": "%s:%s:%s" % (source_ids[0], frame_number, index),
                    })
            _emit({
                "type": "detection_frame",
                "source_id": source_ids[0],
                "event_time": event_time.isoformat(),
                "frame_id": str(frame_number),
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),
                "detections": rows,
            })
    finally:
        capture.release()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _emit({"type": "error", "message": "%s: %s" % (type(exc).__name__, exc)})
        raise
