#!/usr/bin/env python3
"""Regenerate small replay-message fixtures used for manual Phase-6 tests."""

from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests/phase6_fixtures"
OUT.mkdir(parents=True, exist_ok=True)

FIXTURES = {
    "replay_status_zed.json": {
        "start_time": "2026-04-14T18:19:51+00:00",
        "end_time": "2026-04-14T18:20:51+00:00",
        "current": 12.5,
        "event": "progress",
        "node": "dvpg_gq_orin_11",
    },
    "audio_detection.json": {
        "kind": "detection",
        "t": 1776190804.0,
        "node": "dvpg_gq_orin_11",
        "event": "loud_audio",
        "db": -10.0,
        "threshold_db": -30.0,
    },
    "yolo_detection.json": [
        {
            "node": "orin11",
            "source_host": "dvpg_gq_orin_11",
            "class": "car",
            "conf": 0.91,
            "box": [100, 80, 220, 180],
            "t": "2026/04/14 18:20:04.000000",
        }
    ],
    "heartbeat_policy.json": {
        "interval_seconds": 1,
        "suspect_after_misses": 3,
        "unavailable_after_misses": 5,
        "recovery_heartbeats": 2,
    },
}

for name, payload in FIXTURES.items():
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"wrote {len(FIXTURES)} fixtures to {OUT}")
