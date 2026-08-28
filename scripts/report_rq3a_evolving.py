#!/usr/bin/env python3
"""Produce machine-readable RQ3a run and grouped summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results_root.rglob("*.json")):
        if path.name.endswith(("summary.json", "provenance.json")):
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if document.get("schema_version") != "fable.replay_accuracy_run.v2":
            continue
        timing = document.get("timing") or {}
        disturbances = document.get("disturbance_results") or []
        validated = [
            item for item in disturbances
            if item.get("validated") is True
            or (item.get("response") or {}).get("validated") is True
        ]
        helper_seconds = [
            float(item["helper_wall_seconds"])
            for item in validated if item.get("helper_wall_seconds") is not None
        ]
        exposure_classes = [
            str(item.get("classification"))
            for item in (document.get("disturbance_exposure") or [])
        ]
        active_overlap = any(
            item in {
                "ACTIVE_DEMAND_CROSSES_DISTURBANCE_ONSET",
                "ACTIVE_DEMAND_CROSSES_RECOVERY",
                "DEMAND_BEGINS_UNDER_DISTURBANCE",
            }
            for item in exposure_classes
        )
        conformance = document.get("execution_conformance") or {}
        resources = (document.get("resource_instrumentation") or {}).get("totals") or {}
        rows.append({
            "result_path": str(path),
            "experiment_id": document.get("experiment_id") or document.get("trace_id"),
            "year": str(document.get("experiment_id") or document.get("trace_id") or "")[:4],
            "baseline_id": document.get("baseline") or document.get("baseline_id"),
            "condition_trace_id": (document.get("condition_trace") or {}).get("trace_id"),
            "ce_start_offset_seconds": document.get("requested_ce_start_offset_seconds"),
            "exposure_classification": ";".join(exposure_classes),
            "active_demand_overlap": active_overlap,
            "execution_conformance_valid": conformance.get("valid"),
            "selected_yolo_nodes": ";".join(conformance.get("selected_yolo_nodes") or []),
            "observed_yolo_nodes": ";".join(conformance.get("observed_yolo_nodes") or []),
            "classification": document.get("classification"),
            "elapsed_seconds": document.get("elapsed_seconds"),
            "evaluated_system_cpu_seconds": resources.get("evaluated_system_cpu_seconds"),
            "evaluated_path_network_tx_bytes": resources.get("evaluated_path_network_tx_bytes"),
            "evaluated_path_network_rx_bytes": resources.get("evaluated_path_network_rx_bytes"),
            "host_gpu_seconds": resources.get("host_gpu_seconds"),
            "host_gpu_energy_joules": resources.get("host_gpu_energy_joules"),
            "device_tier_host_gpu_seconds": resources.get("device_tier_host_gpu_seconds"),
            "site_local_host_gpu_seconds": resources.get("site_local_host_gpu_seconds"),
            "device_tier_host_gpu_energy_joules": resources.get("device_tier_host_gpu_energy_joules"),
            "site_local_host_gpu_energy_joules": resources.get("site_local_host_gpu_energy_joules"),
            "disturbance_workload_cpu_seconds": resources.get("disturbance_workload_cpu_seconds"),
            "disturbance_gpu_memory_byte_seconds": resources.get("disturbance_gpu_memory_byte_seconds"),
            "provider_gpu_memory_byte_seconds": resources.get("provider_gpu_memory_byte_seconds"),
            "transition_count": len(disturbances),
            "validated_transition_count": len(validated),
            "median_helper_seconds": median(helper_seconds) if helper_seconds else None,
            "error": document.get("error") or "",
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "rq3a_runs.csv", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["baseline_id"], row["condition_trace_id"])].append(row)
    summary = []
    for (baseline, condition), members in sorted(grouped.items(), key=lambda item: str(item[0])):
        elapsed = [float(item["elapsed_seconds"]) for item in members if item["elapsed_seconds"] is not None]
        summary.append({
            "baseline_id": baseline,
            "condition_trace_id": condition,
            "runs": len(members),
            "true_positives": sum(item["classification"] == "TRUE_POSITIVE" for item in members),
            "claim_eligible_runs": sum(
                item["classification"] == "TRUE_POSITIVE"
                and item["active_demand_overlap"]
                and item["execution_conformance_valid"] is True
                for item in members
            ),
            "active_demand_overlap_runs": sum(item["active_demand_overlap"] for item in members),
            "execution_conformant_runs": sum(
                item["execution_conformance_valid"] is True for item in members
            ),
            "failed_transitions": sum(item["transition_count"] - item["validated_transition_count"] for item in members),
            "median_elapsed_seconds": median(elapsed) if elapsed else None,
            "runs_with_error": sum(bool(item["error"]) for item in members),
        })
    _write_csv(args.output_dir / "rq3a_grouped.csv", summary)
    print(json.dumps({"runs": len(rows), "groups": len(summary), "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
