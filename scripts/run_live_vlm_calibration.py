#!/usr/bin/env python3
"""Run the bounded E4 LIVE_VLM identity calibration (three API calls)."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from providers.vehicle.vlm_proxy import HostedVlmProxy  # noqa: E402
from providers.vehicle.vlm_reid import OpenAIVisionIdentityComparator  # noqa: E402


CASES = (
    {
        "case_id": "vehicle_positive_same_observation",
        "entity_kind": "vehicle",
        "left": "debug/robbery32_vehicle_gallery_20260728/images/orin13-candidate0-full.jpg",
        "right": "debug/robbery32_vehicle_gallery_20260728/images/orin13-candidate0-full.jpg",
        "expected_same_identity": True,
        "label_basis": "identical boxed source observation (transport/model sanity check)",
    },
    {
        "case_id": "vehicle_negative_van_vs_white_car",
        "entity_kind": "vehicle",
        "left": "debug/robbery32_vehicle_gallery_20260728/images/orin14-candidate0-full.jpg",
        "right": "debug/robbery32_vehicle_gallery_20260728/images/orin16-candidate0-full.jpg",
        "expected_same_identity": False,
        "label_basis": "manually distinct dark van and white passenger car",
    },
    {
        "case_id": "person_negative_dark_vs_green_clothing",
        "entity_kind": "person",
        "left": "debug/robbery32_cross_camera_reid_20260728/recovered_orin16_track3_full.jpg",
        "right": "debug/robbery32_cross_camera_reid_20260728/departure_orin13_track13_full.jpg",
        "expected_same_identity": False,
        "label_basis": "manual debug review found dark- and bright-green-clothed subjects distinct",
    },
)


def _data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not available to the live calibration process")
    comparator = OpenAIVisionIdentityComparator(
        api_key=key,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    proxy = HostedVlmProxy(comparator, maximum_calls_per_run=10)
    args.output.mkdir(parents=True, exist_ok=True)
    run_id = "e4-live-vlm-calibration-20260801"
    rows = []
    for index, item in enumerate(CASES, start=1):
        left = ROOT / item["left"]
        right = ROOT / item["right"]
        if not left.is_file() or not right.is_file():
            raise FileNotFoundError(f"missing calibration image pair: {left}, {right}")
        started = time.monotonic()
        try:
            response = proxy.compare(
                {
                    "schema_version": "fable.hosted_vlm_request.v1",
                    "run_id": run_id,
                    "invocation_id": f"{run_id}:{index}",
                    "entity_kind": item["entity_kind"],
                    "left_image_url": _data_url(left),
                    "right_image_url": _data_url(right),
                }
            )
            error = None
        except Exception as exc:  # preserve partial calibration evidence
            response = {}
            error = f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.monotonic() - started) * 1000
        predicted = response.get("same_identity")
        rows.append(
            {
                "case_id": item["case_id"],
                "entity_kind": item["entity_kind"],
                "expected_same_identity": item["expected_same_identity"],
                "predicted_same_identity": predicted,
                "correct": predicted == item["expected_same_identity"] if error is None else False,
                "confidence": response.get("confidence"),
                "reason": response.get("reason"),
                "latency_ms": round(elapsed_ms, 3),
                "label_basis": item["label_basis"],
                "left_path": item["left"],
                "right_path": item["right"],
                "left_sha256": _digest(left),
                "right_sha256": _digest(right),
                "error": error,
            }
        )
    successful = [row for row in rows if row["error"] is None]
    latencies = [row["latency_ms"] for row in successful]
    summary = {
        "schema_version": "fable.live_vlm_calibration.v1",
        "run_id": run_id,
        "execution_mode": "LIVE_VLM",
        "model_id": args.model,
        "maximum_calls_per_run": 4,
        "calls_attempted": len(rows),
        "calls_succeeded": len(successful),
        "correct": sum(row["correct"] for row in successful),
        "accuracy": (
            sum(row["correct"] for row in successful) / len(successful)
            if successful else None
        ),
        "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        "maximum_latency_ms": max(latencies) if latencies else None,
        "secret_persisted": False,
        "cases": rows,
    }
    (args.output / "live_vlm_calibration.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, sort_keys=True))
    return 0 if len(successful) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
