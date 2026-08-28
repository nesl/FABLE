#!/usr/bin/env python3
"""Prepare same-revision B1- and B3-only physical E4 continuations."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "evaluation/manifests/adaptation"
SOURCE = MANIFEST_DIR / "physical_e4_multitrace.json"
CURRENT_FABLE = Path(
    "/media/brianw/Extreme SSD2/fable_results/physical_e4_fable_continuation_20260826"
)


def current_nominal_fable(experiment_id: str) -> Path | None:
    result = (
        CURRENT_FABLE
        / experiment_id
        / "matrix/nominal/FABLE/repetition-01"
        / f"{experiment_id}.json"
    )
    if not result.is_file():
        return None
    document = json.loads(result.read_text(encoding="utf-8"))
    return result if document.get("classification") == "TRUE_POSITIVE" else None


def write_policy(document: dict, baseline: str, filename: str) -> Path:
    cases = []
    for source_case in document["cases"]:
        case = dict(source_case)
        case["baselines"] = [baseline]
        if baseline == "B1_STATIC_WHOLE_EVENT":
            # Prefer a calibration from the current-revision FABLE campaign.
            # Preserve a prior audited TP only when current FABLE did not
            # complete the trace; never fabricate a calibration.
            current = current_nominal_fable(case["experiment_id"])
            if current is not None:
                case["b1_calibration_result"] = str(current)
        else:
            case["b1_calibration_result"] = None
        cases.append(case)
    output = {
        **document,
        "campaign_id": f"physical-e4-{baseline.lower()}-continuation-20260826",
        "description": (
            f"Same-revision {baseline}-only continuation over the canonical "
            "22 physical E4 traces and four conditions."
        ),
        "cases": cases,
        "expected_cells": sum(len(case["conditions"]) for case in cases),
    }
    path = MANIFEST_DIR / filename
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    document = json.loads(SOURCE.read_text(encoding="utf-8"))
    outputs = (
        write_policy(document, "B1_STATIC_WHOLE_EVENT", "physical_e4_b1_continuation.json"),
        write_policy(document, "B3_TASK_RESOURCE_ADAPTIVE", "physical_e4_b3_continuation.json"),
    )
    for path in outputs:
        prepared = json.loads(path.read_text(encoding="utf-8"))
        print(f"{path}: cases={len(prepared['cases'])} cells={prepared['expected_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
