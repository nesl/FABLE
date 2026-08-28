#!/usr/bin/env python3
"""Consolidate the four same-revision physical E4 policy continuations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from evaluation.metrics.event_matching import GroundTruthEvent, evaluate_event_results
from evaluation.schemas import ComplexEventResult


ROOTS = {
    "B1_STATIC_WHOLE_EVENT": Path("/media/brianw/Extreme SSD2/fable_results/physical_e4_b1_continuation_20260826"),
    "B3_TASK_RESOURCE_ADAPTIVE": Path("/media/brianw/Extreme SSD2/fable_results/physical_e4_b3_continuation_20260826"),
    "B4_GREEDY_FRONTIER": Path("/media/brianw/Extreme SSD2/fable_results/physical_e4_b4_continuation_20260826"),
    "FABLE": Path("/media/brianw/Extreme SSD2/fable_results/physical_e4_fable_continuation_20260826"),
}
CONDITIONS = ("nominal", "compute_contention", "network_degradation", "network_disconnect")


def load_rows() -> list[dict]:
    rows = []
    for baseline, root in ROOTS.items():
        for condition in CONDITIONS:
            pattern = f"*/matrix/{condition}/{baseline}/repetition-01/*.json"
            for path in root.glob(pattern):
                if path.name in {"plan.json", "report.json", "physical_proxy_validation.json"}:
                    continue
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if "classification" not in result:
                    continue
                metrics = result.get("metrics") or {}
                rescored = evaluate_event_results(
                    tuple(GroundTruthEvent.model_validate(item) for item in result.get("ground_truth") or ()),
                    tuple(ComplexEventResult.model_validate(item) for item in result.get("predictions") or ()),
                    minimum_temporal_iou=float(
                        (result.get("event_matching_policy") or {}).get("minimum_temporal_iou", 0.1)
                    ),
                    temporal_boundary_tolerance_seconds=float(
                        (result.get("event_matching_policy") or {}).get(
                            "temporal_boundary_tolerance_seconds", 0.0
                        )
                    ),
                )
                presence = rescored.event_presence
                assert presence is not None
                resources = (result.get("resource_instrumentation") or {}).get("totals") or {}
                workload = result.get("processed_workload") or {}
                rows.append({
                    "baseline": baseline,
                    "experiment_id": result.get("experiment_id"),
                    "family_id": result.get("family_id"),
                    "condition": condition,
                    "classification": result.get("classification"),
                    "true_positives": presence.true_positives,
                    "false_positives": presence.false_positives,
                    "false_negatives": presence.false_negatives,
                    "raw_identity_true_positives": rescored.true_positives,
                    "raw_identity_false_positives": rescored.false_positives,
                    "raw_identity_false_negatives": rescored.false_negatives,
                    "identity_hypothesis_count": rescored.identity_hypothesis_count,
                    "alternative_hypothesis_count": rescored.alternative_hypothesis_count,
                    "cpu_seconds": resources.get("evaluated_system_cpu_seconds"),
                    "gpu_seconds": resources.get("host_gpu_seconds"),
                    "gpu_energy_joules": resources.get("host_gpu_energy_joules"),
                    "network_bytes": (
                        resources.get("evaluated_path_network_rx_bytes", 0)
                        + resources.get("evaluated_path_network_tx_bytes", 0)
                    ),
                    "frames": sum(
                        (item.get("frames", 0) or 0)
                        for item in workload.values() if isinstance(item, dict)
                    ),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "result_path": str(path),
                })
    return rows


def average(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return mean(values) if values else None


def aggregate(rows: list[dict]) -> dict:
    tp = sum(row["true_positives"] for row in rows)
    fp = sum(row["false_positives"] for row in rows)
    fn = sum(row["false_negatives"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    raw_tp = sum(row["raw_identity_true_positives"] for row in rows)
    raw_fp = sum(row["raw_identity_false_positives"] for row in rows)
    raw_fn = sum(row["raw_identity_false_negatives"] for row in rows)
    raw_precision = raw_tp / (raw_tp + raw_fp) if raw_tp + raw_fp else 0.0
    raw_recall = raw_tp / (raw_tp + raw_fn) if raw_tp + raw_fn else 0.0
    return {
        "cells": len(rows), "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "metric_resolution": "EVENT_PRESENCE",
        "raw_identity_hypotheses": {
            "true_positives": raw_tp,
            "false_positives": raw_fp,
            "false_negatives": raw_fn,
            "precision": raw_precision,
            "recall": raw_recall,
            "f1": (
                2 * raw_precision * raw_recall / (raw_precision + raw_recall)
                if raw_precision + raw_recall else 0.0
            ),
            "hypothesis_count": sum(row["identity_hypothesis_count"] for row in rows),
            "alternative_hypothesis_count": sum(
                row["alternative_hypothesis_count"] for row in rows
            ),
        },
        "classifications": dict(Counter(row["classification"] for row in rows)),
        **{f"mean_{key}": average(rows, key) for key in (
            "cpu_seconds", "gpu_seconds", "gpu_energy_joules", "network_bytes",
            "frames", "elapsed_seconds",
        )},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    keys_by_baseline = defaultdict(set)
    for row in rows:
        keys_by_baseline[row["baseline"]].add((row["experiment_id"], row["condition"]))
    common = set.intersection(*(keys_by_baseline[name] for name in ROOTS)) if all(keys_by_baseline[name] for name in ROOTS) else set()
    matched = [row for row in rows if (row["experiment_id"], row["condition"]) in common]
    report = {
        "schema_version": "fable.physical_e4_consolidated.v2",
        "accuracy_contract": {
            "primary": "EVENT_PRESENCE",
            "secondary": "RAW_IDENTITY_HYPOTHESES",
            "note": (
                "Event-presence labels do not identify participants. Temporally "
                "equivalent binding hypotheses are one occurrence for primary accuracy."
            ),
        },
        "source_roots": {key: str(value) for key, value in ROOTS.items()},
        "available_cells": {name: len(keys_by_baseline[name]) for name in ROOTS},
        "matched_cells_per_baseline": len(common),
        "all_available": {name: aggregate([r for r in rows if r["baseline"] == name]) for name in ROOTS},
        "matched": {name: aggregate([r for r in matched if r["baseline"] == name]) for name in ROOTS},
        "by_condition": {
            name: {condition: aggregate([r for r in matched if r["baseline"] == name and r["condition"] == condition]) for condition in CONDITIONS}
            for name in ROOTS
        },
    }
    (args.output / "consolidated.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output / "cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["baseline"])
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
