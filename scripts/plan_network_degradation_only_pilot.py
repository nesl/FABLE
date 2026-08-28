#!/usr/bin/env python3
"""Derive the five-policy 50%-to-95% network degradation pilot."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation/manifests/adaptation/rq3a_updated/rq3a_network_updated_pass_follow_pilot_10.jsonl"
TRACE = ROOT / "evaluation/manifests/adaptation/rq3a_updated/n1_pass_follow_orin13.json"
OUTPUT = ROOT / "evaluation/manifests/adaptation/rq3a_updated/network_degradation_only_pilot_5.jsonl"


def main() -> int:
    rows = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("disturbance_profile_id") != "N1":
            continue
        row["condition_trace_id"] = "rq3a-updated-n1-pass-follow-orin13"
        row["condition_trace_path"] = str(TRACE)
        row["run_id"] = f"{row['run_id']}-network-degradation-only-pilot"
        rows.append(row)
    if len(rows) != 5:
        raise RuntimeError(f"expected five degraded policy rows, found {len(rows)}")
    OUTPUT.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
