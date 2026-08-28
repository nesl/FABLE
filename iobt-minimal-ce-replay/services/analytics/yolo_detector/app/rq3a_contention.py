#!/usr/bin/env python3
"""Fixed local YOLO workload used only by the E1 GPU-contention profile."""

from __future__ import annotations

import os
import signal
import time
import json
from statistics import median

import numpy as np
import torch
from ultralytics import YOLO


running = True


def _stop(_signum, _frame):
    global running
    running = False


def main() -> int:
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    model_path = os.environ.get("YOLO_MODEL", "/app/yolov8s.pt")
    device = os.environ.get("YOLO_DEVICE", "0")
    batch_size = max(1, min(32, int(os.environ.get("FABLE_CONTENTION_BATCH_SIZE", "8"))))
    streams = max(1, min(4, int(os.environ.get("FABLE_CONTENTION_STREAMS", "2"))))
    duty = max(0.1, min(1.0, float(os.environ.get("RQ3A_GPU_DUTY", "0.80"))))
    if not torch.cuda.is_available():
        raise RuntimeError("E1 GPU contention requires CUDA")
    cuda_device = torch.device(f"cuda:{device}")
    model = YOLO(model_path)
    detector = model.model.to(cuda_device).eval()
    # Allocate and retain a fixed decoded input batch. The workload performs no
    # disk reads, downloads, MQTT publication, or network transfer after load.
    images = torch.zeros(
        (batch_size, 3, 640, 640), dtype=torch.float32, device=cuda_device
    )
    with torch.inference_mode():
        detector(images)
    torch.cuda.synchronize(cuda_device)
    if os.environ.get("FABLE_CONTENTION_MODE") == "benchmark":
        samples = []
        iterations = max(5, int(os.environ.get("FABLE_BENCHMARK_ITERATIONS", "12")))
        for _ in range(iterations):
            started = time.perf_counter()
            with torch.inference_mode():
                detector(images)
            torch.cuda.synchronize(cuda_device)
            samples.append((time.perf_counter() - started) * 1000.0)
        ordered = sorted(samples)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        print(json.dumps({"event": "BENCHMARK", "p50_ms": median(samples), "p95_ms": p95}), flush=True)
        return 0
    print(
        json.dumps(
            {
                "event": "READY",
                "gpu_uuid": os.environ.get("FABLE_CONTENTION_GPU_UUID", ""),
                "batch_size": batch_size,
                "streams": streams,
                "duty_cycle": duty,
                "model": model_path,
            }
        ),
        flush=True,
    )
    cuda_streams = [torch.cuda.Stream(device=cuda_device) for _ in range(streams)]
    while running:
        cycle = time.monotonic()
        # Enqueue concurrent work in independent CUDA streams. Sequential
        # forwards can report high average utilization while barely affecting
        # the p95 latency of a second process because the CUDA scheduler slips
        # it between kernels. Concurrent streams produce the intended shared
        # site-accelerator queueing pressure while retaining one container and
        # one fixed in-memory input.
        for cuda_stream in cuda_streams:
            with torch.cuda.stream(cuda_stream), torch.inference_mode():
                detector(images)
        torch.cuda.synchronize(cuda_device)
        busy = time.monotonic() - cycle
        if duty < 1.0:
            time.sleep(max(0.0, busy * (1.0 - duty) / duty))
    print(json.dumps({"event": "STOPPED"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
