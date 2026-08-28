#!/usr/bin/env python3
"""Standalone Python 3.8-compatible physical-node E0 benchmark."""

from __future__ import print_function

import argparse
import json
import os
import time

import cv2
import numpy as np


def crop_benchmark(image, repetitions):
    height, width = image.shape[:2]
    boxes = (
        (width // 8, height // 4, width // 2, 3 * height // 4),
        (width // 2, height // 4, 7 * width // 8, 3 * height // 4),
    )
    samples = []
    encoded_bytes = []
    for _ in range(repetitions):
        started = time.perf_counter()
        outputs = []
        for x1, y1, x2, y2 in boxes:
            crop = image[y1:y2, x1:x2]
            ok, encoded = cv2.imencode(
                ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )
            if not ok:
                raise RuntimeError("JPEG crop encoding failed")
            outputs.append(encoded)
        samples.append((time.perf_counter() - started) * 1000.0)
        encoded_bytes.append(sum(item.nbytes for item in outputs))
    return {
        "provider_id": "bounded_track_crop_extractor",
        "warm_execution_samples_ms": samples,
        "encoded_bytes_samples": encoded_bytes,
    }


def reid_benchmark(image, config_path, model_path, repetitions, device):
    import torch
    from fastreid.config import get_cfg
    from fastreid.modeling.meta_arch import build_model
    from fastreid.utils.checkpoint import Checkpointer

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.BACKBONE.PRETRAIN = False
    started = time.perf_counter()
    model = build_model(cfg)
    model.eval()
    Checkpointer(model).load(model_path)
    width, height = [int(value) for value in cfg.INPUT.SIZE_TEST[::-1]]
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))
    batch = torch.stack((tensor, tensor.clone())).to(device)
    with torch.no_grad():
        output = model({"images": batch})
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - started) * 1000.0
    samples = []
    for _ in range(repetitions):
        invoked = time.perf_counter()
        with torch.no_grad():
            output = model({"images": batch})
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - invoked) * 1000.0)
    return {
        "provider_id": "vehicle_reid_descriptor",
        "first_execution_ms": first_ms,
        "warm_execution_samples_ms": samples,
        "output_dimension": int(output.shape[-1]),
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("crop", "reid"), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--config")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit("cannot decode fixture: " + args.image)
    if args.mode == "crop":
        result = crop_benchmark(image, args.repetitions)
    else:
        if not args.config or not args.model:
            parser.error("reid mode requires --config and --model")
        result = reid_benchmark(
            image, args.config, args.model, args.repetitions, args.device
        )
    result.update({
        "schema_version": "fable.physical_provider_e0.v1",
        "hostname": os.uname()[1],
        "architecture": os.uname()[4],
        "mode": args.mode,
        "fixture": args.image,
        "successful": True,
    })
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
