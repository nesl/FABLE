#!/usr/bin/env python3
"""Build a supplemental timing/resource report from durable RQ1 records."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median


TERMINAL_LIFECYCLE_EVENTS = {"IDLE", "STOPPED", "FAILED", "DRAINED"}
ACTIVE_LIFECYCLE_EVENTS = {"READY", "ACTIVE", "STARTED"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def parse_time(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]


def lifecycle_metrics(rows: list[dict]) -> dict[str, object]:
    starts: dict[str, datetime] = {}
    completed_seconds = 0.0
    provider_starts = 0
    duplicate_active_events = 0
    for row in sorted(rows, key=lambda item: str(item.get("wall_timestamp") or item.get("event_time") or "")):
        instance = str(row.get("provider_instance_id") or "")
        event = str(row.get("lifecycle_event") or "").upper()
        timestamp = parse_time(row.get("wall_timestamp") or row.get("event_time"))
        if not instance or timestamp is None:
            continue
        if event in ACTIVE_LIFECYCLE_EVENTS:
            if instance in starts:
                duplicate_active_events += 1
            else:
                starts[instance] = timestamp
                provider_starts += 1
        elif event in TERMINAL_LIFECYCLE_EVENTS:
            started = starts.pop(instance, None)
            if started is not None:
                completed_seconds += max(0.0, (timestamp - started).total_seconds())
    return {
        "provider_starts": provider_starts,
        "provider_active_seconds": round(completed_seconds, 6) if rows and not starts else None,
        "provider_active_complete": bool(rows) and not starts,
        "unclosed_provider_instances": len(starts),
        "duplicate_active_events": duplicate_active_events,
    }


def record_dir(result_path: Path, result: dict) -> Path:
    configured = result.get("common_record_dir")
    if configured:
        path = Path(str(configured))
        if path.is_dir():
            return path
    return result_path.with_suffix(".records")


def optional_number(mapping: dict, key: str) -> float | None:
    value = mapping.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consolidated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    consolidated = args.consolidated_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (consolidated / "outcomes.csv").open(encoding="utf-8", newline="") as handle:
        outcomes = list(csv.DictReader(handle))

    per_run: list[dict] = []
    instrumentation = defaultdict(int)
    for outcome in outcomes:
        source = Path(outcome["result_source"]) if outcome.get("result_source") else None
        if source is None or not source.is_file():
            per_run.append({**outcome, "result_available": False})
            continue
        result = load_json(source)
        metrics = result.get("metrics") or {}
        timing = result.get("timing") or {}
        records = record_dir(source, result)
        lifecycle = load_jsonl(records / "provider_lifecycle.jsonl")
        resources = load_jsonl(records / "resource_sample.jsonl")
        artifacts = load_jsonl(records / "artifact_event.jsonl")
        plans = load_jsonl(records / "plan_decision.jsonl")
        lifecycle_summary = lifecycle_metrics(lifecycle)

        evaluated_container_rows = [
            row for row in resources
            if (row.get("metadata") or {}).get("measurement_kind") == "docker_cgroup_interval_delta"
            and (row.get("metadata") or {}).get("container_category") == "evaluated_system"
        ]
        gpu_rows = [
            row for row in resources
            if (row.get("metadata") or {}).get("measurement_kind") == "nvidia_smi_host_gpu_interval"
        ]
        provider_gpu_residency_rows = [
            row for row in resources
            if (row.get("metadata") or {}).get("measurement_kind")
            == "nvidia_smi_process_gpu_residency"
            and (row.get("metadata") or {}).get("container_category")
            == "evaluated_system"
        ]
        network_rows = [
            row for row in resources
            if (row.get("metadata") or {}).get("measurement_kind") == "network_namespace_interval_delta"
        ]
        measured_network = bool(network_rows)
        measured_cpu = bool(evaluated_container_rows)
        measured_gpu = any(
            row.get("gpu_energy_joules") is not None or row.get("gpu_utilization") is not None
            for row in gpu_rows
        )
        instrumentation["resource_rows"] += len(resources)
        instrumentation["runs_with_measured_cpu"] += int(measured_cpu)
        instrumentation["runs_with_measured_gpu"] += int(measured_gpu)
        instrumentation["runs_with_measured_network"] += int(measured_network)
        instrumentation["runs_with_complete_provider_active"] += int(
            lifecycle_summary["provider_active_complete"]
        )

        artifact_bytes = sum(
            int(row.get("bytes") or 0)
            for row in artifacts
            if str(row.get("action") or "").upper() in {"CREATE", "WRITE", "RETAIN"}
        )
        transfer_event_bytes = sum(
            int(row.get("bytes") or 0)
            for row in artifacts
            if str(row.get("action") or "").upper() == "TRANSFER"
        )
        predicted_transfer_bytes = sum(int(row.get("predicted_transfer_bytes") or 0) for row in plans)
        processed_workload = result.get("processed_workload") or {}
        yolo_workload = [
            counters for key, counters in processed_workload.items()
            if str(key).startswith("yolo:") and isinstance(counters, dict)
        ]
        per_run.append(
            {
                **outcome,
                "result_available": True,
                "timely_recall": optional_number(metrics, "timely_recall"),
                "median_detection_delay_seconds": optional_number(metrics, "median_detection_delay_seconds"),
                "p95_detection_delay_seconds": optional_number(metrics, "p95_detection_delay_seconds"),
                "total_wall_seconds": optional_number(timing, "total_wall_seconds"),
                "sync_to_terminal_or_cutoff_seconds": optional_number(timing, "sync_to_terminal_or_cutoff_seconds"),
                "observed_event_time_span_seconds": optional_number(timing, "observed_event_time_span_seconds"),
                "labeled_experiment_duration_seconds": optional_number(timing, "labeled_experiment_duration_seconds"),
                **lifecycle_summary,
                "artifact_bytes_written": artifact_bytes,
                "transfer_event_bytes": transfer_event_bytes,
                "predicted_transfer_bytes": predicted_transfer_bytes,
                "resource_sample_count": len(resources),
                "evaluated_system_cpu_seconds": sum(float(row.get("cpu_time_seconds") or 0) for row in evaluated_container_rows),
                "unique_network_namespace_tx_bytes": sum(int(row.get("network_tx_bytes") or 0) for row in network_rows),
                "unique_network_namespace_rx_bytes": sum(int(row.get("network_rx_bytes") or 0) for row in network_rows),
                "host_gpu_seconds": sum(float(row.get("gpu_time_seconds") or 0) for row in gpu_rows),
                "host_gpu_energy_joules": sum(float(row.get("gpu_energy_joules") or 0) for row in gpu_rows),
                "provider_gpu_memory_byte_seconds": sum(
                    float((row.get("metadata") or {}).get("memory_byte_seconds") or 0)
                    for row in provider_gpu_residency_rows
                ),
                "peak_provider_gpu_memory_bytes": max(
                    (
                        int(
                            (row.get("metadata") or {}).get(
                                "concurrent_provider_gpu_memory_bytes"
                            )
                            or row.get("gpu_memory_bytes")
                            or 0
                        )
                        for row in provider_gpu_residency_rows
                    ),
                    default=0,
                ),
                "yolo_gpu_inference_seconds": sum(
                    float(row.get("gpu_inference_seconds") or 0) for row in yolo_workload
                ),
                "yolo_inference_wall_seconds": sum(
                    float(row.get("inference_wall_seconds") or 0) for row in yolo_workload
                ),
                "yolo_model_resident_seconds": sum(
                    float(row.get("model_resident_seconds") or 0) for row in yolo_workload
                ),
                "yolo_inference_count": sum(
                    int(row.get("inference_count") or 0) for row in yolo_workload
                ),
                "reid_gpu_inference_seconds": sum(
                    float(row.get("reid_gpu_inference_seconds") or 0) for row in yolo_workload
                ),
                "reid_inference_wall_seconds": sum(
                    float(row.get("reid_inference_wall_seconds") or 0) for row in yolo_workload
                ),
                "reid_inference_count": sum(
                    int(row.get("reid_inference_count") or 0) for row in yolo_workload
                ),
                "measured_cpu_available": measured_cpu,
                "measured_gpu_available": measured_gpu,
                "measured_network_bytes_available": measured_network,
                "record_dir": str(records),
            }
        )

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in per_run:
        if row.get("result_available"):
            groups[str(row["baseline_id"])].append(row)
    summary = []
    for baseline, rows in sorted(groups.items()):
        def values(key: str) -> list[float]:
            return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]

        timely = values("timely_recall")
        wall = values("total_wall_seconds")
        active = values("provider_active_seconds")
        summary.append(
            {
                "baseline_id": baseline,
                "executed_runs": len(rows),
                "runs_with_timely_recall": len(timely),
                "mean_timely_recall": round(mean(timely), 6) if timely else None,
                "mean_total_wall_seconds": round(mean(wall), 6) if wall else None,
                "median_total_wall_seconds": round(median(wall), 6) if wall else None,
                "p95_total_wall_seconds": round(p95(wall), 6) if wall else None,
                "runs_with_complete_provider_active": len(active),
                "mean_provider_active_seconds": round(mean(active), 6) if active else None,
                "mean_provider_starts": round(mean(values("provider_starts")), 6),
                "mean_artifact_bytes_written": round(mean(values("artifact_bytes_written")), 3),
                "mean_transfer_event_bytes": round(mean(values("transfer_event_bytes")), 3),
                "mean_predicted_transfer_bytes": round(mean(values("predicted_transfer_bytes")), 3),
                "mean_evaluated_system_cpu_seconds": round(mean(values("evaluated_system_cpu_seconds")), 6),
                "mean_unique_network_namespace_tx_bytes": round(mean(values("unique_network_namespace_tx_bytes")), 3),
                "mean_unique_network_namespace_rx_bytes": round(mean(values("unique_network_namespace_rx_bytes")), 3),
                "mean_host_gpu_seconds": round(mean(values("host_gpu_seconds")), 6),
                "mean_host_gpu_energy_joules": round(mean(values("host_gpu_energy_joules")), 6),
                "mean_provider_gpu_memory_byte_seconds": round(mean(values("provider_gpu_memory_byte_seconds")), 3),
                "mean_peak_provider_gpu_memory_bytes": round(mean(values("peak_provider_gpu_memory_bytes")), 3),
                "mean_yolo_gpu_inference_seconds": round(mean(values("yolo_gpu_inference_seconds")), 6),
                "mean_yolo_inference_wall_seconds": round(mean(values("yolo_inference_wall_seconds")), 6),
                "mean_yolo_model_resident_seconds": round(mean(values("yolo_model_resident_seconds")), 6),
                "mean_yolo_inference_count": round(mean(values("yolo_inference_count")), 3),
                "mean_reid_gpu_inference_seconds": round(mean(values("reid_gpu_inference_seconds")), 6),
                "mean_reid_inference_wall_seconds": round(mean(values("reid_inference_wall_seconds")), 6),
                "mean_reid_inference_count": round(mean(values("reid_inference_count")), 3),
            }
        )

    per_run_fields = tuple(dict.fromkeys(key for row in per_run for key in row))
    summary_fields = tuple(summary[0])
    write_csv(output / "per_run_timing_resources.csv", per_run, per_run_fields)
    write_csv(output / "baseline_timing_resources.csv", summary, summary_fields)
    availability = {
        "schema_version": "fable.rq1_supplemental_timing.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "consolidated_dir": str(consolidated),
        "planned_rows": len(per_run),
        "executed_rows": sum(bool(row.get("result_available")) for row in per_run),
        "instrumentation": dict(instrumentation),
        "metric_validity": {
            "timely_recall": "measured per live result",
            "wall_and_event_timing": "measured per live result",
            "provider_active_seconds": "READY/ACTIVE/STARTED to IDLE/STOPPED/FAILED/DRAINED; null when an interval is unclosed",
            "artifact_bytes_written": "measured artifact CREATE/WRITE/RETAIN bytes; not wire traffic",
            "transfer_event_bytes": "measured only when explicit artifact TRANSFER records exist",
            "predicted_transfer_bytes": "planner estimate summed across plan decisions; not measured traffic",
            "cpu_seconds": "Docker cgroup interval deltas for future/instrumented RQ1 runs; unavailable in the original campaign",
            "gpu_seconds": "host-GPU utilization integral for future/instrumented RQ1 runs; unavailable in the original campaign",
            "provider_gpu_residency": "Per-CUDA-process VRAM mapped to its Docker cgroup; memory-byte-seconds measure residency, not computation",
            "yolo_gpu_inference_seconds": "Run-delta of CUDA-event timing around YOLO inference; excludes idle model residency",
            "measured_network_bytes": "container network-namespace interval deltas for future/instrumented RQ1 runs; unavailable in the original campaign",
        },
        "summary": summary,
        "per_run_csv": str((output / "per_run_timing_resources.csv").resolve()),
        "baseline_summary_csv": str((output / "baseline_timing_resources.csv").resolve()),
    }
    (output / "report.json").write_text(
        json.dumps(availability, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(availability, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
