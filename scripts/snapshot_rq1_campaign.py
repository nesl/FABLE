#!/usr/bin/env python3
"""Snapshot a paused RQ1 campaign and emit an explicit resumable manifest."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_FIELDS = (
    "evaluated_system_cpu_seconds",
    "evaluated_path_network_tx_bytes",
    "evaluated_path_network_rx_bytes",
    "host_gpu_seconds",
    "host_gpu_energy_joules",
    "device_tier_host_gpu_seconds",
    "site_local_host_gpu_seconds",
    "device_tier_host_gpu_energy_joules",
    "site_local_host_gpu_energy_joules",
    "disturbance_workload_cpu_seconds",
    "disturbance_gpu_memory_byte_seconds",
    "provider_gpu_memory_byte_seconds",
    "peak_provider_gpu_memory_bytes",
    "yolo_gpu_inference_seconds",
    "reid_gpu_inference_seconds",
    "processed_frames",
    "yolo_model_resident_seconds",
)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resource_row(result: dict, baseline: str, experiment_id: str) -> dict:
    totals = (result.get("resource_instrumentation") or {}).get("totals") or {}
    yolo = [
        counters
        for key, counters in (result.get("processed_workload") or {}).items()
        if str(key).startswith("yolo:") and isinstance(counters, dict)
    ]
    return {
        "baseline_id": baseline,
        "experiment_id": experiment_id,
        "classification": result.get("classification"),
        "evaluated_system_cpu_seconds": totals.get("evaluated_system_cpu_seconds"),
        "evaluated_path_network_tx_bytes": totals.get("evaluated_path_network_tx_bytes"),
        "evaluated_path_network_rx_bytes": totals.get("evaluated_path_network_rx_bytes"),
        "host_gpu_seconds": totals.get("host_gpu_seconds"),
        "host_gpu_energy_joules": totals.get("host_gpu_energy_joules"),
        "device_tier_host_gpu_seconds": totals.get("device_tier_host_gpu_seconds"),
        "site_local_host_gpu_seconds": totals.get("site_local_host_gpu_seconds"),
        "device_tier_host_gpu_energy_joules": totals.get("device_tier_host_gpu_energy_joules"),
        "site_local_host_gpu_energy_joules": totals.get("site_local_host_gpu_energy_joules"),
        "disturbance_workload_cpu_seconds": totals.get("disturbance_workload_cpu_seconds"),
        "disturbance_gpu_memory_byte_seconds": totals.get("disturbance_gpu_memory_byte_seconds"),
        "provider_gpu_memory_byte_seconds": totals.get("provider_gpu_memory_byte_seconds"),
        "peak_provider_gpu_memory_bytes": totals.get("peak_provider_gpu_memory_bytes"),
        "yolo_gpu_inference_seconds": sum(
            float(item.get("gpu_inference_seconds") or 0) for item in yolo
        ),
        "reid_gpu_inference_seconds": sum(
            float(item.get("reid_gpu_inference_seconds") or 0) for item in yolo
        ),
        "processed_frames": sum(int(item.get("frames") or 0) for item in yolo),
        "yolo_model_resident_seconds": sum(
            float(item.get("model_resident_seconds") or 0) for item in yolo
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = args.manifest.resolve(strict=True)
    campaign = args.campaign_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    planned = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baselines = sorted({str(row["baseline_id"]) for row in planned})

    event_failures: set[tuple[str, str, int]] = set()
    event_path = campaign / "campaign-events.jsonl"
    if event_path.is_file():
        for line in event_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if int(event.get("returncode") or 0) != 0:
                for experiment_id in event.get("experiment_ids") or []:
                    event_failures.add(
                        (
                            str(event["baseline_id"]),
                            str(experiment_id),
                            int(event["repetition"]),
                        )
                    )

    results: dict[tuple[str, str, int], dict] = {}
    outcome_rows: list[dict] = []
    remaining: list[dict] = []
    counts: dict[str, Counter] = defaultdict(Counter)
    for run in planned:
        baseline = str(run["baseline_id"])
        experiment_id = str(run["experiment_id"])
        repetition = int(run["repetition"])
        key = (baseline, experiment_id, repetition)
        result_path = (
            campaign
            / baseline
            / f"repetition-{repetition:02d}"
            / f"{experiment_id}.json"
        )
        result = read_json(result_path)
        complete = bool(result and result.get("suite")) and result.get(
            "classification"
        ) != "RUNTIME_FAILURE"
        if complete:
            classification = str(result.get("classification") or "UNKNOWN")
            results[key] = result
        elif key in event_failures:
            classification = "RUNTIME_FAILURE"
            remaining.append(run)
        else:
            classification = "NOT_COMPLETED"
            remaining.append(run)
        counts[baseline][classification] += 1
        outcome_rows.append(
            {
                "baseline_id": baseline,
                "experiment_id": experiment_id,
                "repetition": repetition,
                "classification": classification,
                "result_path": str(result_path) if result_path.exists() else "",
            }
        )

    # A fair resource comparison uses only paired traces for which every policy
    # produced a correct detection. Failed/short runs must not appear efficient.
    experiment_ids = sorted({str(row["experiment_id"]) for row in planned})
    paired_correct = []
    for experiment_id in experiment_ids:
        matching = [
            results.get((baseline, experiment_id, 1)) for baseline in baselines
        ]
        if all(
            result is not None and result.get("classification") == "TRUE_POSITIVE"
            for result in matching
        ):
            paired_correct.append(experiment_id)

    resource_rows = []
    for experiment_id in paired_correct:
        for baseline in baselines:
            resource_rows.append(
                resource_row(results[(baseline, experiment_id, 1)], baseline, experiment_id)
            )
    resource_summary = []
    for baseline in baselines:
        rows = [row for row in resource_rows if row["baseline_id"] == baseline]
        summary = {
            "baseline_id": baseline,
            "paired_correct_traces": len(rows),
            "comparison_cohort": "all_baselines_true_positive_on_same_trace",
        }
        for field in RESOURCE_FIELDS:
            values = [
                float(row[field])
                for row in rows
                if isinstance(row.get(field), (int, float))
            ]
            summary[f"mean_{field}"] = round(mean(values), 6) if values else None
        resource_summary.append(summary)

    outcome_summary = []
    for baseline in baselines:
        row = {"baseline_id": baseline, "planned": sum(counts[baseline].values())}
        for classification in (
            "TRUE_POSITIVE",
            "FALSE_NEGATIVE",
            "FALSE_POSITIVE",
            "TRUE_NEGATIVE",
            "RUNTIME_FAILURE",
            "NOT_COMPLETED",
        ):
            row[classification.lower()] = counts[baseline][classification]
        classified = row["true_positive"] + row["false_negative"]
        row["recall_completed_positive_runs"] = (
            round(row["true_positive"] / classified, 6) if classified else None
        )
        outcome_summary.append(row)

    remaining_path = output / "remaining_runs.jsonl"
    remaining_path.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in remaining),
        encoding="utf-8",
    )
    write_csv(
        output / "outcomes_all_planned.csv",
        outcome_rows,
        ["baseline_id", "experiment_id", "repetition", "classification", "result_path"],
    )
    write_csv(
        output / "baseline_outcome_summary.csv",
        outcome_summary,
        list(outcome_summary[0]),
    )
    write_csv(
        output / "resource_comparison_paired_correct_runs.csv",
        resource_summary,
        list(resource_summary[0]),
    )
    write_csv(
        output / "resource_per_run_paired_correct.csv",
        resource_rows,
        ["baseline_id", "experiment_id", "classification", *RESOURCE_FIELDS],
    )
    (output / "paired_correct_trace_ids.txt").write_text(
        "\n".join(paired_correct) + ("\n" if paired_correct else ""), encoding="utf-8"
    )

    resume_command = (
        f"cd {shlex.quote(str(ROOT))} && "
        "tmux new-session -d -s rq1-full-20260806-v2-resume \""
        ".venv/bin/python scripts/run_planned_ce_campaign.py "
        f"--manifest '{remaining_path}' --output-dir '{campaign}' "
        "--execution-order ce-round-robin --max-seconds 300 --ready-seconds 30 "
        "--mobile-root '/media/brianw/Extreme SSD3' "
        f"2>&1 | tee -a '{campaign / 'campaign-resume.log'}'\""
    )
    state = {
        "schema_version": "fable.rq1_pause_snapshot.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "original_manifest": str(manifest),
        "campaign_dir": str(campaign),
        "planned_cells": len(planned),
        "complete_cells": len(planned) - len(remaining),
        "remaining_cells": len(remaining),
        "paired_correct_trace_count": len(paired_correct),
        "resource_comparison_rule": (
            "Include a trace only when every baseline has a durable TRUE_POSITIVE; "
            "then compare the same trace set across all baselines."
        ),
        "resume_command": resume_command,
        "outcome_summary": outcome_summary,
    }
    (output / "resume_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "RESUME_COMMAND.txt").write_text(resume_command + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "RQ1 paused campaign snapshot\n"
        "============================\n\n"
        f"Original matrix: {manifest}\n"
        f"Durable results: {campaign}\n"
        f"Complete cells: {state['complete_cells']} / {state['planned_cells']}\n"
        f"Remaining/retry cells: {state['remaining_cells']}\n"
        f"Paired correct resource cohort: {len(paired_correct)} traces\n\n"
        "remaining_runs.jsonl is the explicit resumable workload. It includes both "
        "never-completed cells and runtime failures that must be retried. The runner "
        "also checks the durable campaign directory and skips any already-complete cell.\n\n"
        "Resource comparison validity\n"
        "----------------------------\n"
        "resource_comparison_paired_correct_runs.csv compares CPU, network, GPU, "
        "energy, model residency, inference time, and frames only on traces where "
        "all six policies produced TRUE_POSITIVE. This prevents incorrect or short "
        "runs from appearing artificially inexpensive. Host GPU values include all "
        "activity on the host during each sequential run; CUDA YOLO/ReID counters "
        "are provider-level inference measurements.\n\n"
        "Use the exact command in RESUME_COMMAND.txt to resume unattended.\n",
        encoding="utf-8",
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
