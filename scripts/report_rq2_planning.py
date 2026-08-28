#!/usr/bin/env python3
"""Generate an RQ2-specific planning report from a bounded live campaign."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean


def load_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def avg(values) -> float:
    values = [float(item) for item in values if item is not None]
    return round(mean(values), 6) if values else 0.0


def percentile(values, fraction: float) -> float:
    values = sorted(float(item) for item in values if item is not None)
    if not values:
        return 0.0
    return round(values[int(round((len(values) - 1) * fraction))], 6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    run_rows = []
    artifact_rows = []
    signatures = {}
    all_plans = []
    for result_path in sorted(campaign.glob("*/*/repetition-01/*.json")):
        if result_path.name in {"plan.json", "report.json"}:
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not result.get("experiment_id"):
            continue
        profile = result_path.parts[-4]
        policy = result_path.parts[-3]
        record_dir = Path(result.get("common_record_dir") or "")
        plans = load_lines(record_dir / "plan_decision.jsonl")
        artifacts = load_lines(record_dir / "artifact_event.jsonl")
        resources = load_lines(record_dir / "resource_sample.jsonl")
        all_plans.extend(plans)
        key = (profile, policy, str(result["experiment_id"]))
        signatures[key] = tuple(
            (
                tuple(item.get("selected_chain_ids", ())),
                tuple(item.get("selected_node_ids", ())),
                tuple(item.get("activated_provider_keys", ())),
                tuple(item.get("continuation_types", ())),
            )
            for item in plans
        )
        latency_values = [float(item.get("planning_latency_ms") or 0) for item in plans]
        predicted_slacks = [item.get("predicted_slack_ms") for item in plans]
        run_rows.append(
            {
                "network_profile_id": profile,
                "baseline_id": policy,
                "experiment_id": result["experiment_id"],
                "campaign_year": result.get("campaign_year"),
                "variant": result.get("variant"),
                "classification": result.get("classification"),
                "end_to_end_success": int(result.get("classification") == "TRUE_POSITIVE"),
                "feasible_plan": int(bool(plans)),
                "plan_decisions": len(plans),
                "mean_predicted_completion_ms": avg(item.get("predicted_completion_ms") for item in plans),
                "total_predicted_transfer_bytes": sum(int(item.get("predicted_transfer_bytes") or 0) for item in plans),
                "mean_labels_generated": avg(item.get("labels_generated") for item in plans),
                "mean_labels_retained": avg(item.get("labels_retained") for item in plans),
                "unique_selected_nodes": len({node for item in plans for node in item.get("selected_node_ids", ())}),
                "continuation_selection_count": sum(len(item.get("continuation_types", ())) for item in plans),
                "artifact_bytes_created": sum(int(item.get("bytes") or 0) for item in artifacts if item.get("action") == "CREATE"),
                "cpu_seconds": round(sum(float(item.get("cpu_time_seconds") or 0) for item in resources), 6),
                "planning_latency_instrumented": int(bool(latency_values) and any(value > 0 for value in latency_values)),
                "predicted_slack_instrumented": int(any(value is not None for value in predicted_slacks)),
            }
        )
        by_type = defaultdict(lambda: {"count": 0, "bytes": 0})
        for item in artifacts:
            if item.get("action") != "CREATE":
                continue
            bucket = by_type[str(item.get("artifact_type") or "unknown")]
            bucket["count"] += 1
            bucket["bytes"] += int(item.get("bytes") or 0)
        for artifact_type, values in sorted(by_type.items()):
            artifact_rows.append(
                {
                    "network_profile_id": profile,
                    "baseline_id": policy,
                    "experiment_id": result["experiment_id"],
                    "artifact_type": artifact_type,
                    **values,
                }
            )

    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["network_profile_id"], row["baseline_id"])].append(row)
    summary_rows = []
    for (profile, policy), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "network_profile_id": profile,
                "baseline_id": policy,
                "runs": len(rows),
                "feasible_plan_rate": avg(row["feasible_plan"] for row in rows),
                "end_to_end_success_rate": avg(row["end_to_end_success"] for row in rows),
                "end_to_end_deadline_miss_rate": avg(1 - row["end_to_end_success"] for row in rows),
                "mean_plan_decisions": avg(row["plan_decisions"] for row in rows),
                "mean_predicted_completion_ms": avg(row["mean_predicted_completion_ms"] for row in rows),
                "mean_predicted_transfer_bytes": avg(row["total_predicted_transfer_bytes"] for row in rows),
                "mean_labels_generated": avg(row["mean_labels_generated"] for row in rows),
                "mean_labels_retained": avg(row["mean_labels_retained"] for row in rows),
                "mean_artifact_bytes_created": avg(row["artifact_bytes_created"] for row in rows),
                "mean_cpu_seconds": avg(row["cpu_seconds"] for row in rows),
            }
        )

    paired_rows = []
    metrics = (
        "end_to_end_success",
        "plan_decisions",
        "mean_predicted_completion_ms",
        "total_predicted_transfer_bytes",
        "mean_labels_generated",
        "mean_labels_retained",
        "artifact_bytes_created",
        "cpu_seconds",
    )
    indexed = {(r["network_profile_id"], r["baseline_id"], r["experiment_id"]): r for r in run_rows}
    for profile in sorted({r["network_profile_id"] for r in run_rows}):
        experiments = sorted({r["experiment_id"] for r in run_rows if r["network_profile_id"] == profile})
        for control in ("B2_FRONTIER_FIXED_REALIZATION", "B4_GREEDY_FRONTIER"):
            for metric in metrics:
                differences = []
                for experiment in experiments:
                    treatment = indexed[(profile, "FABLE", experiment)][metric]
                    baseline = indexed[(profile, control, experiment)][metric]
                    differences.append(float(treatment) - float(baseline))
                paired_rows.append(
                    {
                        "network_profile_id": profile,
                        "treatment": "FABLE",
                        "control": control,
                        "metric": metric,
                        "pair_count": len(differences),
                        "mean_treatment_minus_control": avg(differences),
                    }
                )

    realization_rows = []
    for profile in sorted({r["network_profile_id"] for r in run_rows}):
        experiments = sorted({r["experiment_id"] for r in run_rows if r["network_profile_id"] == profile})
        for control in ("B2_FRONTIER_FIXED_REALIZATION", "B4_GREEDY_FRONTIER"):
            changed = sum(
                signatures[(profile, "FABLE", experiment)] != signatures[(profile, control, experiment)]
                for experiment in experiments
            )
            realization_rows.append(
                {
                    "network_profile_id": profile,
                    "treatment": "FABLE",
                    "control": control,
                    "trace_pairs": len(experiments),
                    "different_realization_traces": changed,
                    "different_realization_fraction": round(changed / len(experiments), 6),
                }
            )

    run_fields = tuple(run_rows[0])
    summary_fields = tuple(summary_rows[0])
    write_csv(output / "per_run_planning.csv", run_rows, run_fields)
    write_csv(output / "policy_profile_summary.csv", summary_rows, summary_fields)
    write_csv(output / "artifact_representation.csv", artifact_rows, tuple(artifact_rows[0]))
    write_csv(output / "paired_fable_comparisons.csv", paired_rows, tuple(paired_rows[0]))
    write_csv(output / "realization_differences.csv", realization_rows, tuple(realization_rows[0]))

    latency_values = [float(item.get("planning_latency_ms") or 0) for item in all_plans]
    report = {
        "schema_version": "fable.rq2_planning_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_dir": str(campaign),
        "run_count": len(run_rows),
        "plan_decision_count": len(all_plans),
        "classification_counts": dict(Counter(row["classification"] for row in run_rows)),
        "planning_latency": {
            "available": bool(latency_values) and any(value > 0 for value in latency_values),
            "record_count": len(latency_values),
            "nonzero_record_count": sum(value > 0 for value in latency_values),
            "reason_unavailable": "live plan records emitted zero for planning_latency_ms" if not any(value > 0 for value in latency_values) else "",
        },
        "deadline_slack": {
            "available": any(item.get("predicted_slack_ms") is not None for item in all_plans),
            "reason_unavailable": "live plan records did not populate predicted_slack_ms",
        },
        "oracle_cost_gap": {
            "available": False,
            "reason_unavailable": "oracle execution was intentionally deferred for this bounded campaign",
        },
        "metric_coverage": {
            "search_label_counts": {
                "fully_comparable": False,
                "reason": "B2 and B4 live records emit zero label counters; FABLE emits bounded-search counters",
            },
            "cpu_seconds": {
                "available": any(float(row["cpu_seconds"]) > 0 for row in run_rows),
                "reason": "resource records did not expose nonzero cpu_time_seconds for these live runs",
            },
            "transfer_by_representation": {
                "available": True,
                "scope": "artifact CREATE bytes by artifact_type; predicted plan transfer remains aggregate",
            },
        },
        "interpretation": {
            "feasible_plan_rate": "fraction of runs with at least one emitted plan decision",
            "end_to_end_deadline_miss_rate": "fraction of positive traces ending without a timely accepted CE; not a planner-slack measurement",
            "realization_difference": "trace-level comparison of ordered selected chains, nodes, provider keys, and continuation types",
        },
        "files": {
            "per_run_planning": str(output / "per_run_planning.csv"),
            "policy_profile_summary": str(output / "policy_profile_summary.csv"),
            "artifact_representation": str(output / "artifact_representation.csv"),
            "paired_fable_comparisons": str(output / "paired_fable_comparisons.csv"),
            "realization_differences": str(output / "realization_differences.csv"),
        },
        "summary": summary_rows,
        "realization_differences": realization_rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
