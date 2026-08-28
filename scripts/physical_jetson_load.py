#!/usr/bin/env python3
"""Bounded, deterministic CUDA contention workload for Jetson preflights/E4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--matrix-size", type=int, default=3072)
    parser.add_argument("--duty-cycle", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=34001)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args()
    if not 0 < args.duration_seconds <= 300:
        parser.error("duration must be in (0, 300]")
    if not 256 <= args.matrix_size <= 4096:
        parser.error("matrix size must be in [256, 4096]")
    if not 0 < args.duty_cycle <= 1:
        parser.error("duty cycle must be in (0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    left = torch.randn((args.matrix_size, args.matrix_size), device=device)
    right = torch.randn((args.matrix_size, args.matrix_size), device=device)
    torch.cuda.synchronize()
    started = time.monotonic()
    deadline = started + args.duration_seconds
    iterations = 0
    active_seconds = 0.0
    stopping = [False]

    def request_stop(_signum, _frame):
        stopping[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    ready = {
        "schema_version": "fable.physical_contention_ready.v1",
        "pid": os.getpid(),
        "ready": True,
        "matrix_size": args.matrix_size,
        "duty_cycle": args.duty_cycle,
        "started_wall_time": time.time(),
    }
    if args.ready_file is not None:
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.ready_file.with_suffix(args.ready_file.suffix + ".tmp")
        temporary.write_text(json.dumps(ready, sort_keys=True) + "\n")
        temporary.replace(args.ready_file)
    try:
        while time.monotonic() < deadline and not stopping[0]:
            cycle_started = time.monotonic()
            active_until = cycle_started + 0.1 * args.duty_cycle
            while (
                time.monotonic() < active_until
                and time.monotonic() < deadline
                and not stopping[0]
            ):
                left = torch.mm(left, right)
                left.mul_(1e-4)
                # CUDA launches are asynchronous. Synchronizing each bounded
                # unit prevents the active phase from queueing work that keeps
                # the GPU saturated throughout the nominal sleep phase.
                torch.cuda.synchronize()
                iterations += 1
            active_seconds += time.monotonic() - cycle_started
            remaining = cycle_started + 0.1 - time.monotonic()
            if remaining > 0 and not stopping[0]:
                time.sleep(remaining)
    finally:
        result = {
            "schema_version": "fable.physical_contention_result.v1",
            "duration_seconds": time.monotonic() - started,
            "active_seconds": active_seconds,
            "iterations": iterations,
            "matrix_size": args.matrix_size,
            "duty_cycle": args.duty_cycle,
            "seed": args.seed,
            "terminated": stopping[0],
        }
        encoded = json.dumps(result, sort_keys=True)
        if args.result_file is not None:
            args.result_file.parent.mkdir(parents=True, exist_ok=True)
            args.result_file.write_text(encoded + "\n")
        print(encoded, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
