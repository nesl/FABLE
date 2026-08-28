#!/usr/bin/env python3
"""Summarize one common replay trace executed across controlled baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CANONICAL = (
    "B0_PRODUCE_ALL",
    "B1_STATIC_WHOLE_EVENT",
    "B2_FRONTIER_FIXED_REALIZATION",
    "B3_TASK_RESOURCE_ADAPTIVE",
    "B4_GREEDY_FRONTIER",
    "FABLE",
)

RECORD_FAMILIES = (
    "predicate_demand",
    "plan_decision",
    "provider_command",
    "provider_lease",
    "provider_lifecycle",
    "resource_sample",
    "predicate_observation",
    "artifact_event",
)


def _record_summary(result_path: Path, request_id: str) -> dict[str, object]:
    record_dir = result_path.parent / f"{result_path.stem}.records"
    summary: dict[str, object] = {"record_dir": str(record_dir)}
    observed_request_ids: set[str] = set()
    for family in RECORD_FAMILIES:
        path = record_dir / f"{family}.jsonl"
        rows = (
            [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if path.is_file()
            else []
        )
        summary[f"{family}_records"] = len(rows)
        if family in {"provider_lifecycle", "resource_sample", "artifact_event"}:
            summary[f"{family}_native_request_records"] = sum(
                row.get("metadata", {}).get("attribution") == "native_request"
                for row in rows
            )
            summary[f"{family}_measurement_window_records"] = sum(
                row.get("metadata", {}).get("attribution")
                == "measurement_window"
                for row in rows
            )
        observed_request_ids.update(
            str(row["request_id"]) for row in rows if row.get("request_id")
        )
    summary["record_request_isolation"] = (
        bool(observed_request_ids)
        and observed_request_ids == {request_id}
    )
    summary["normalization_errors"] = sum(
        1
        for line in (record_dir / "normalization_errors.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ) if (record_dir / "normalization_errors.jsonl").is_file() else 0
    summary["required_common_records_complete"] = (
        summary["record_request_isolation"] is True
        and summary["normalization_errors"] == 0
        and all(
            int(summary[f"{family}_records"]) > 0
            for family in (
                "predicate_demand",
                "plan_decision",
                "provider_command",
                "provider_lease",
                "provider_lifecycle",
                "resource_sample",
                "predicate_observation",
            )
        )
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for baseline in CANONICAL:
        candidates = (
            args.result_dir / f"{baseline}.rerun.json",
            args.result_dir / f"{baseline}.json",
        )
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            rows.append({"baseline_id": baseline, "classification": "MISSING"})
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        row = {
                "baseline_id": baseline,
                "classification": result.get("classification", ""),
                "detected": result.get("detected", False),
                "elapsed_seconds": result.get("elapsed_seconds", 0),
                "progress_messages": result.get("progress_messages", 0),
                "passes_observed": (result.get("vehicle_predicates_by_id") or {}).get(
                    "PASSES", 0
                ),
                "follows_observed": (result.get("vehicle_predicates_by_id") or {}).get(
                    "FOLLOWS", 0
                ),
                "yolo_vehicle_detections": sum(
                    int(count)
                    for label, count in (result.get("yolo_classes") or {}).items()
                    if label in {"car", "truck", "bus", "motorcycle"}
                ),
                "source_file": path.name,
            }
        row.update(_record_summary(path, str(result.get("request_id") or "")))
        rows.append(row)
    output = args.output or args.result_dir / "common_baseline_summary.csv"
    fields = (
        "baseline_id",
        "classification",
        "detected",
        "elapsed_seconds",
        "progress_messages",
        "passes_observed",
        "follows_observed",
        "yolo_vehicle_detections",
        "source_file",
        "record_dir",
        *(f"{family}_records" for family in RECORD_FAMILIES),
        *(
            f"{family}_{attribution}_records"
            for family in (
                "provider_lifecycle",
                "resource_sample",
                "artifact_event",
            )
            for attribution in ("native_request", "measurement_window")
        ),
        "record_request_isolation",
        "normalization_errors",
        "required_common_records_complete",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "fable.common_baseline_trace_report.v1",
        "result_dir": str(args.result_dir.resolve()),
        "summary_csv": str(output.resolve()),
        "baselines": len(rows),
        "true_positives": sum(
            item.get("classification") == "TRUE_POSITIVE" for item in rows
        ),
        "missing": sum(item.get("classification") == "MISSING" for item in rows),
        "request_isolation_passed": sum(
            item.get("record_request_isolation") is True for item in rows
        ),
        "normalization_errors": sum(
            int(item.get("normalization_errors", 0)) for item in rows
        ),
        "common_records_complete": sum(
            item.get("required_common_records_complete") is True
            for item in rows
        ),
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
