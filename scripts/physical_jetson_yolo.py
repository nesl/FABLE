#!/usr/bin/env python3
"""Run real CUDA YOLO on a Pi-hosted video stream and publish detections."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

import cv2
import torch
import torchvision


def torch_nms(boxes, scores, iou_threshold):
    """Torch-only NMS for Jetson builds without ABI-compatible torchvision ops."""
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
        area_index = (boxes[index, 2] - boxes[index, 0]) * (
            boxes[index, 3] - boxes[index, 1]
        )
        area_remaining = (boxes[remaining, 2] - boxes[remaining, 0]) * (
            boxes[remaining, 3] - boxes[remaining, 1]
        )
        iou = intersection / (area_index + area_remaining - intersection + 1e-7)
        order = remaining[iou <= iou_threshold]
    return torch.stack(keep)


torchvision.ops.nms = torch_nms
from ultralytics import YOLO  # noqa: E402
import paho.mqtt.client as mqtt  # noqa: E402
from physical_sampling import attach_replay_provenance, deterministic_sample_due  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--broker", required=True)
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--source-id", default="physical_rpi")
    parser.add_argument("--topic")
    parser.add_argument("--maximum-rate-hz", type=float, default=5.0)
    parser.add_argument("--maximum-frames", type=int, default=0)
    parser.add_argument("--controlled-replay", action="store_true")
    parser.add_argument("--publish-raw-frames", action="store_true")
    parser.add_argument(
        "--relay-only-flag",
        type=Path,
        default=Path("/home/nesl/FABLE/state/physical-yolo-relay-only"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on the Jetson")
    topic = args.topic or f"/{args.source_id}/analytics/yolo/bbox"
    replay_config = {}
    configured = threading.Event()
    synchronized = threading.Event()
    client = mqtt.Client(client_id=f"fable-physical-yolo-{int(time.time())}")

    def on_message(_client, _userdata, message):
        nonlocal replay_config
        if not message.payload:
            return
        document = json.loads(message.payload.decode("utf-8"))
        targets = set(document.get("target_nodes") or ())
        logical_node = args.source_id.replace("dvpg_gq_orin_", "orin")
        if targets and args.source_id not in targets and logical_node not in targets:
            return
        if message.topic == "/replay/config":
            replay_config = document
            configured.set()
            readiness = {
                "ready": True,
                "source_id": args.source_id,
                "node_id": args.source_id,
                "scenario": document.get("scenario"),
                "replay_id": document.get("replay_id"),
                "physical_replay": True,
                "t": time.time(),
            }
            for service in ("zed", "yolo"):
                readiness["service"] = service
                client.publish(
                    f"/readiness/{args.source_id}/{service}",
                    json.dumps(readiness), qos=0, retain=True,
                )
        elif message.topic == "/replay/sync":
            if replay_config and document.get("replay_id") == replay_config.get("replay_id"):
                replay_config.update(document)
                synchronized.set()

    client.on_message = on_message
    client.connect(args.broker, args.port, keepalive=60)
    if args.controlled_replay:
        client.subscribe("/replay/config", qos=1)
        client.subscribe("/replay/sync", qos=1)
    client.loop_start()
    if args.controlled_replay:
        if not configured.wait(60):
            raise RuntimeError("timed out waiting for /replay/config")
        if not synchronized.wait(60):
            raise RuntimeError("timed out waiting for /replay/sync")
        delay = float(replay_config.get("start_at") or 0) - time.time()
        if delay > 0:
            time.sleep(delay)
    capture = cv2.VideoCapture(args.stream_url)
    if not capture.isOpened():
        raise RuntimeError(f"could not open Pi stream {args.stream_url}")
    model_load_started = time.monotonic()
    model = YOLO(args.model)
    model_load_seconds = time.monotonic() - model_load_started
    frame_number = published = raw_frames_published = 0
    inference_count = 0
    inference_wall_seconds = 0.0
    started = time.monotonic()
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 30.0
    sample_period_frames = (
        source_fps / args.maximum_rate_hz
        if args.maximum_rate_hz > 0
        else 1.0
    )
    next_sample_frame = 1.0
    sampled_frame_numbers = []
    event_start_at = float(replay_config.get("event_start_at") or time.time())
    status_topic = f"/debug/{args.source_id}/analytics/yolo/status"

    def publish_status(*, complete: bool = False) -> None:
        client.publish(
            status_topic,
            json.dumps(
                {
                    "hostname": args.source_id,
                    "node": args.source_id,
                    "source_host": args.source_id,
                    "provider": "yolov8n-jetson-cuda",
                    "physical_replay": True,
                    "scenario": replay_config.get("scenario"),
                    "replay_id": replay_config.get("replay_id"),
                    "input_frames_total": frame_number,
                    "detections_total": published,
                    "dropped_superseded_frames": 0,
                    "inference_count": inference_count,
                    "inference_wall_seconds": inference_wall_seconds,
                    "gpu_inference_seconds": inference_wall_seconds,
                    "source_fps": source_fps,
                    "sampling_policy": "deterministic_media_frame_v1",
                    "sample_period_frames": sample_period_frames,
                    "last_sampled_frame_number": (
                        sampled_frame_numbers[-1]
                        if sampled_frame_numbers
                        else None
                    ),
                    "sampled_frame_count": len(sampled_frame_numbers),
                    "sampled_frame_numbers_tail": sampled_frame_numbers[-32:],
                    "relay_only": args.relay_only_flag.exists(),
                    "raw_frames_published": raw_frames_published,
                    "complete": complete,
                    "t": time.time(),
                }
            ),
            qos=0,
            retain=False,
        )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if args.maximum_frames and frame_number > args.maximum_frames:
                break
            if not deterministic_sample_due(
                frame_number,
                next_sample_frame=next_sample_frame,
            ):
                continue
            while next_sample_frame <= frame_number + 1e-9:
                next_sample_frame += sample_period_frames
            sampled_frame_numbers.append(frame_number)
            frame_event_time = event_start_at + (frame_number - 1) / source_fps
            if args.publish_raw_frames:
                encoded, jpeg = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                )
                if encoded:
                    envelope = json.dumps(
                        {
                            "schema_version": "fable.remote_camera_frame.v1",
                            "data": base64.b64encode(jpeg.tobytes()).decode("ascii"),
                            "t": frame_event_time,
                            "replay_id": replay_config.get("replay_id"),
                            "frame_number": frame_number,
                        }
                    )
                    client.publish(
                        f"/{args.source_id}/zed/rgb_left/compressed",
                        envelope,
                        qos=0,
                    )
                    # The legacy remote-camera adapter joins RGB and depth.
                    # Publish a valid zero-valued single-channel depth image;
                    # this preserves unknown-depth semantics without making
                    # the detector interpret RGB channel vectors as depth.
                    depth_ok, depth_png = cv2.imencode(
                        ".png", frame[:, :, 0] * 0
                    )
                    depth_envelope = json.dumps(
                        {
                            "schema_version": "fable.remote_camera_frame.v1",
                            "data": base64.b64encode(
                                depth_png.tobytes() if depth_ok else jpeg.tobytes()
                            ).decode("ascii"),
                            "t": frame_event_time,
                            "replay_id": replay_config.get("replay_id"),
                            "frame_number": frame_number,
                        }
                    )
                    client.publish(
                        f"/{args.source_id}/zed/depth/compressed",
                        depth_envelope,
                        qos=0,
                    )
                    raw_frames_published += 1
            if args.relay_only_flag.exists():
                publish_status()
                continue
            inference_started = time.monotonic()
            results = model(frame, device=0, imgsz=640, verbose=False)
            inference_wall_seconds += time.monotonic() - inference_started
            inference_count += 1
            # Keep evidence from recorded media in recording time. ``start_at``
            # coordinates wall playback; ``event_start_at`` timestamps frame 1.
            timestamp = datetime.fromtimestamp(
                frame_event_time, tz=timezone.utc
            ).isoformat()
            detections = []
            identity_crops = 0
            for result in results:
                for box in result.boxes:
                    detection = {
                            "node": "jetson",
                            "source_host": args.source_id,
                            "model": "yolov8n-jetson-cuda",
                            "class": result.names[int(box.cls.item())],
                            "conf": float(box.conf.item()),
                            "box": [int(value) for value in box.xywh[0].cpu().tolist()],
                            "depth": -1.0,
                            "world": [],
                            "world_valid": False,
                            "t": timestamp,
                            "frame_number": frame_number,
                        }
                    # Preserve the detector-aligned pixels needed by bounded
                    # identity replay.  The desktop detector has always
                    # carried this field; omitting it on the physical Jetson
                    # made PASSES work while its exact track IDs had no crop
                    # evidence to answer a later SAME_ENTITY demand.
                    cx, cy, width, height = detection["box"]
                    x1 = max(0, cx - width // 2)
                    y1 = max(0, cy - height // 2)
                    x2 = min(frame.shape[1], cx + width // 2)
                    y2 = min(frame.shape[0], cy + height // 2)
                    if (
                        identity_crops < 2
                        and detection["class"] in {"car", "truck", "bus", "motorcycle"}
                        and x2 > x1
                        and y2 > y1
                    ):
                        crop = frame[y1:y2, x1:x2]
                        scale = min(1.0, 256.0 / max(crop.shape[:2]))
                        if scale < 1.0:
                            crop = cv2.resize(
                                crop,
                                None,
                                fx=scale,
                                fy=scale,
                                interpolation=cv2.INTER_AREA,
                            )
                        crop_ok, crop_jpeg = cv2.imencode(
                            ".jpg",
                            crop,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 75],
                        )
                        if crop_ok:
                            detection["crop_data_url"] = (
                                "data:image/jpeg;base64,"
                                + base64.b64encode(crop_jpeg.tobytes()).decode("ascii")
                            )
                            identity_crops += 1
                    detections.append(detection)
            if detections:
                detections = attach_replay_provenance(
                    detections,
                    replay_id=replay_config.get("replay_id"),
                    scenario=replay_config.get("scenario"),
                )
                client.publish(topic, json.dumps(detections), qos=0)
                published += len(detections)
            publish_status()
    finally:
        publish_status(complete=True)
        time.sleep(0.05)
        capture.release()
        client.loop_stop()
        client.disconnect()
    if args.controlled_replay:
        client = mqtt.Client(client_id=f"fable-physical-eof-{int(time.time())}")
        client.connect(args.broker, args.port, keepalive=30)
        client.publish(
            f"/replay/status/zed/{args.source_id}",
            json.dumps(
                {
                    "event": "complete",
                    "service": "zed",
                    "node": args.source_id,
                    "scenario": replay_config.get("scenario"),
                    "replay_id": replay_config.get("replay_id"),
                    "physical_replay": True,
                    "t": time.time(),
                }
            ), qos=1,
        ).wait_for_publish(timeout=5)
        client.disconnect()
    print(
        f"PHYSICAL_YOLO_COMPLETE frames={frame_number} detections={published} "
        f"inferences={inference_count} "
        f"model_load_seconds={model_load_seconds:.6f} "
        f"inference_wall_seconds={inference_wall_seconds:.6f} "
        f"source_fps={source_fps:.3f} "
        f"sampling_policy=deterministic_media_frame_v1 "
        f"sample_period_frames={sample_period_frames:.6f} "
        f"last_sampled_frame={sampled_frame_numbers[-1] if sampled_frame_numbers else 0} "
        f"raw_frames_published={raw_frames_published} "
        f"wall_seconds={time.monotonic() - started:.3f} topic={topic}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
