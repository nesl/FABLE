#!/usr/bin/env python3
"""Derive the FABLE-only physical E4 continuation from the canonical matrix."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evaluation/manifests/adaptation/physical_e4_multitrace.json"
OUTPUT = ROOT / "evaluation/manifests/adaptation/physical_e4_fable_continuation.json"


def main() -> int:
    document = json.loads(SOURCE.read_text(encoding="utf-8"))
    cases = []
    for source_case in document["cases"]:
        case = dict(source_case)
        case["baselines"] = ["FABLE"]
        # FABLE plans directly from the grounded frontier and does not consume
        # the trace-authored B1 calibration contract.
        case["b1_calibration_result"] = None
        cases.append(case)
    output = {
        **document,
        "schema_version": "fable.physical_e4_multitrace.v1",
        "campaign_id": "physical-e4-fable-continuation-20260826",
        "description": (
            "FABLE-only continuation over the same canonical E4 traces and "
            "conditions used by the B4 continuation."
        ),
        "cases": cases,
        "expected_cells": sum(
            len(case["baselines"]) * len(case["conditions"])
            for case in cases
        ),
    }
    OUTPUT.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(OUTPUT)
    print(f"cases={len(cases)} cells={output['expected_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
