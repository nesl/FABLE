#!/usr/bin/env python3
"""Summarize JSON outputs from physical_reid_benchmark.py (Python 3.8+)."""

from __future__ import print_function

import argparse
import glob
import json
import math
import os
import statistics


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_glob")
    args = parser.parse_args()
    paths = sorted(glob.glob(args.results_glob))
    records = []
    skipped = []
    for path in paths:
        try:
            with open(path, "r") as handle:
                record = json.load(handle)
        except (ValueError, OSError):
            skipped.append(os.path.basename(path))
            continue
        if record.get("schema_version") != "fable.physical_provider_e0.v1":
            skipped.append(os.path.basename(path))
            continue
        records.append(record)
    cold = [record["first_execution_ms"] for record in records if "first_execution_ms" in record]
    warm = [sample for record in records for sample in record["warm_execution_samples_ms"]]
    summary = {
        "schema_version": "fable.physical_provider_e0_summary.v1",
        "provider_id": records[0]["provider_id"] if records else None,
        "hostname": records[0]["hostname"] if records else None,
        "result_files": [os.path.basename(path) for path in paths],
        "skipped_files": skipped,
        "process_count": len(records),
        "cold_sample_count": len(cold),
        "warm_sample_count": len(warm),
        "cold_ms": {
            "p50": statistics.median(cold) if cold else None,
            "p95": percentile(cold, 0.95),
            "min": min(cold) if cold else None,
            "max": max(cold) if cold else None,
        },
        "warm_ms": {
            "p50": statistics.median(warm) if warm else None,
            "p95": percentile(warm, 0.95),
            "min": min(warm) if warm else None,
            "max": max(warm) if warm else None,
        },
        "gpu_peak_allocated_mb": None,
    }
    gpu_samples = [record["gpu_peak_allocated_mb"] for record in records if "gpu_peak_allocated_mb" in record]
    if gpu_samples:
        summary["gpu_peak_allocated_mb"] = {
            "p50": statistics.median(gpu_samples),
            "max": max(gpu_samples),
        }
    byte_samples = [sample for record in records for sample in record.get("encoded_bytes_samples", [])]
    if byte_samples:
        summary["encoded_bytes"] = {
            "p50": statistics.median(byte_samples),
            "p95": percentile(byte_samples, 0.95),
            "min": min(byte_samples),
            "max": max(byte_samples),
        }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
