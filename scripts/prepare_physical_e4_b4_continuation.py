#!/usr/bin/env python3
"""Derive the B4-only physical E4 continuation from the canonical matrix."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation/manifests/adaptation/physical_e4_multitrace.json"
OUTPUT = ROOT / "evaluation/manifests/adaptation/physical_e4_b4_continuation.json"


def main() -> int:
    document = json.loads(SOURCE.read_text(encoding="utf-8"))
    cases = []
    for source_case in document["cases"]:
        case = dict(source_case)
        case["baselines"] = ["B4_GREEDY_FRONTIER"]
        # B4 is grounded-frontier greedy and does not consume a B1-authored
        # calibration. Retain every condition and trace from canonical E4.
        case["b1_calibration_result"] = None
        cases.append(case)
    output = {
        **document,
        "schema_version": "fable.physical_e4_multitrace.v1",
        "campaign_id": "physical-e4-b4-continuation-20260826",
        "description": (
            "B4-only continuation over the canonical E4 traces and nominal, "
            "compute-contention, network-degradation, and disconnect conditions."
        ),
        "cases": cases,
        "expected_cells": sum(
            len(case["baselines"]) * len(case["conditions"])
            for case in cases
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(f"cases={len(cases)} cells={output['expected_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
