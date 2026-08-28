#!/usr/bin/env python3
"""Offline diagnostic for YOLO -> ByteTrack -> uncalibrated lifecycle evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import enum
import json
from pathlib import Path
import typing

import cv2
from ultralytics import YOLO

# The deployed runtime is Python 3.11. The existing YOLO diagnostic image is
# Python 3.10, so provide the standard-library 3.11 enum during this read-only
# offline diagnostic only.
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        pass

    enum.StrEnum = StrEnum  # type: ignore[attr-defined]
if not hasattr(typing, "Self"):
    from typing_extensions import Self

    typing.Self = Self  # type: ignore[attr-defined]

from providers.vehicle.detector import LegacyReplayYoloAdapter
from providers.vehicle.geometry import TrackLifecycleExitEvaluator
from providers.vehicle.tracker import RoboflowTrackerAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    step = max(1, round(source_fps / args.rate_hz))
    model = YOLO(str(args.model))
    adapter = LegacyReplayYoloAdapter()
    tracker = RoboflowTrackerAdapter(
        algorithm="bytetrack", frame_rate=args.rate_hz, session_id="diagnostic"
    )
    lifecycle = TrackLifecycleExitEvaluator()
    outputs = []
    spans: dict[str, dict] = {}
    epoch = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for frame_number in range(0, frame_count, step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok:
            continue
        result = model(frame, device="cpu", imgsz=640, verbose=False)[0]
        event_time = epoch + timedelta(seconds=frame_number / source_fps)
        rows = [
            {
                "node": "diagnostic",
                "source_host": "camera",
                "model": "yolov8n",
                "class": result.names[int(box.cls.item())],
                "conf": float(box.conf.item()),
                "box": [int(value) for value in box.xywh[0].cpu().tolist()],
                "depth": -1.0,
                "world": [],
                "world_valid": False,
                "t": event_time.isoformat(),
                "frame_number": frame_number,
            }
            for box in result.boxes
        ]
        if not rows:
            continue
        track_set = tracker.update(
            adapter.parse(
                rows,
                source_id="camera",
                frame_id=str(frame_number),
                source_sequence=frame_number,
            )
        )
        for track in track_set.tracks:
            if not track.attributes.get("matched_detection_id"):
                continue
            row = spans.setdefault(
                track.scoped_track_id,
                {
                    "first": track.bbox.center,
                    "first_width": track.bbox.width,
                    "last": track.bbox.center,
                    "last_width": track.bbox.width,
                    "samples": 0,
                },
            )
            row.update(last=track.bbox.center, last_width=track.bbox.width)
            row["samples"] += 1
        outputs.extend(lifecycle.update(track_set))
    for row in spans.values():
        dx = row["last"][0] - row["first"][0]
        dy = row["last"][1] - row["first"][1]
        displacement = (dx * dx + dy * dy) ** 0.5
        width = max(1.0, (row["first_width"] + row["last_width"]) / 2.0)
        row["displacement_px"] = displacement
        row["displacement_box_widths"] = displacement / width
    print(
        json.dumps(
            {
                "source_fps": source_fps,
                "sample_rate_hz": args.rate_hz,
                "track_spans": spans,
                "outputs": [item.model_dump(mode="json") for item in outputs],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
