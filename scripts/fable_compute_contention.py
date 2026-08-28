#!/usr/bin/env python3
"""Typed CLI for the fixed site-local GPU-contention workload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.compute_contention import apply_e1, calibrate_e1, restore_n0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("E1", "N0"), required=True)
    parser.add_argument("--action", choices=("APPLY", "RESTORE", "CALIBRATE"), required=True)
    args = parser.parse_args()
    if (args.condition, args.action) not in {
        ("E1", "APPLY"), ("E1", "CALIBRATE"), ("N0", "RESTORE")
    }:
        parser.error("only E1/APPLY, E1/CALIBRATE, and N0/RESTORE are valid")
    try:
        measurements = (
            calibrate_e1() if args.action == "CALIBRATE"
            else apply_e1() if args.action == "APPLY"
            else restore_n0()
        )
        validated = (
            measurements.get("calibrated") is True
            if args.action in {"APPLY", "CALIBRATE"}
            else measurements.get("running") is False
            and measurements.get("recovered") is True
        )
        result = {"validated": validated, "measurements": measurements}
    except Exception as exc:
        if args.action == "CALIBRATE":
            try:
                restore_n0()
            except Exception:
                pass
        result = {"validated": False, "measurements": {}, "reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["validated"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
