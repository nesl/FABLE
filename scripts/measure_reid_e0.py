#!/usr/bin/env python3
"""Measure the production FastReID path over a bounded two-crop input."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import torch

from fable.common.time import EventTimeInterval
from providers.vehicle.descriptors import FastReidEntityDescriptor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--tier", choices=("sensor", "server"), required=True)
    parser.add_argument("--warm", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="/models/reid/vehicle.pth")
    parser.add_argument("--config", default="/app/reid/fastreid_veri_sbs_r50_ibn.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot decode fixture: {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crops = (("vehicle-a", image), ("vehicle-b", image.copy()))
    now = datetime.now(UTC)
    interval = EventTimeInterval(start=now, end=now)
    descriptor = FastReidEntityDescriptor(
        entity_kind="vehicle",
        config_path=args.config,
        model_path=args.model,
        model_id="fastreid:sbs_R50_ibn:vehicle",
        model_version="veri-wild-sbs-r50-ibn-v1",
        preprocessing_id="fastreid-veri-256x256-rgb",
        device=args.device,
    )

    def synchronize() -> None:
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    process_started_ns = time.perf_counter_ns()
    cpu_started_ns = time.process_time_ns()
    result = descriptor.encode(crops, source_id="e0-camera", event_time_interval=interval)
    synchronize()
    first_ms = (time.perf_counter_ns() - process_started_ns) / 1_000_000
    first_cpu_ms = (time.process_time_ns() - cpu_started_ns) / 1_000_000

    warm_ms = []
    warm_cpu_ms = []
    for _ in range(args.warm):
        started_ns = time.perf_counter_ns()
        cpu_ns = time.process_time_ns()
        descriptor.encode(crops, source_id="e0-camera", event_time_interval=interval)
        synchronize()
        warm_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000)
        warm_cpu_ms.append((time.process_time_ns() - cpu_ns) / 1_000_000)

    payload = {
        "schema_version": "fable.reid_e0_measurement.v1",
        "provider_id": (
            "vehicle_reid_descriptor_site"
            if args.tier == "server"
            else "vehicle_reid_descriptor"
        ),
        "tier": args.tier,
        "input_class": "bounded_reid_crop_set.v1" if args.tier == "server" else "image_crop_set.v1",
        "batch_size": len(crops),
        "fixture": args.image,
        "device": args.device,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "first_execution_ms": first_ms,
        "first_cpu_ms": first_cpu_ms,
        "warm_execution_samples_ms": warm_ms,
        "warm_cpu_samples_ms": warm_cpu_ms,
        "gpu_peak_allocated_mb": (
            torch.cuda.max_memory_allocated() / 1024 / 1024
            if torch.cuda.is_available() else 0
        ),
        "output_dimension": result.dimension,
        "successful": len(result.records) == len(crops),
    }
    serialized = json.dumps(payload, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
