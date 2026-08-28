"""Deterministic JSON and CSV reports over common evaluation records."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from evaluation.metrics.statistics import (
    LoadSample,
    confidence_interval,
    maximum_sustainable_load,
    paired_comparison,
)


RECORD_FILES = (
    "predicate_observation",
    "complex_event_result",
    "hypothesis_transition",
    "predicate_demand",
    "plan_decision",
    "provider_command",
    "provider_lifecycle",
    "provider_lease",
    "artifact_event",
    "resource_sample",
    "network_condition",
    "disturbance_event",
    "coordination_episode",
    "retrospective_attempt",
)


def generate_evaluation_report(
    run_roots: Iterable[str | Path],
    output_dir: str | Path,
) -> dict[str, object]:
    roots = tuple(sorted((Path(item).resolve() for item in run_roots), key=str))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, object]]] = {
        record_type: [] for record_type in RECORD_FILES
    }
    exclusions: list[dict[str, str]] = []
    included_runs: set[str] = set()
    for root in roots:
        if not root.is_dir():
            exclusions.append(
                {"path": str(root), "reason": "run directory does not exist"}
            )
            continue
        for record_type in RECORD_FILES:
            path = root / f"{record_type}.jsonl"
            if not path.exists():
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    exclusions.append(
                        {
                            "path": str(path),
                            "reason": f"invalid JSON line {line_number}: {exc}",
                        }
                    )
                    continue
                records[record_type].append(item)
                run_id = item.get("run_id")
                if run_id:
                    included_runs.add(str(run_id))

    event_counts: Counter[tuple[str, str]] = Counter()
    for item in records["complex_event_result"]:
        event_counts[
            (str(item.get("baseline_id", "")), str(item.get("event_family", "")))
        ] += 1
    planning: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in records["plan_decision"]:
        planning[
            (str(item.get("baseline_id", "")), str(item.get("trace_id", "")))
        ].append(item)

    summary_rows: list[dict[str, object]] = []
    keys = sorted(set(event_counts) | {(baseline, "") for baseline, _ in planning})
    for baseline, family in keys:
        plan_items = [
            item
            for (item_baseline, _), values in planning.items()
            if item_baseline == baseline
            for item in values
        ]
        summary_rows.append(
            {
                "baseline_id": baseline,
                "event_family": family,
                "accepted_event_count": event_counts[(baseline, family)],
                "plan_count": len(plan_items),
                "mean_planning_latency_ms": _mean(
                    item.get("planning_latency_ms") for item in plan_items
                ),
                "total_predicted_transfer_bytes": sum(
                    int(item.get("predicted_transfer_bytes") or 0)
                    for item in plan_items
                ),
            }
        )

    _write_csv(
        output / "summary.csv",
        summary_rows,
        (
            "baseline_id",
            "event_family",
            "accepted_event_count",
            "plan_count",
            "mean_planning_latency_ms",
            "total_predicted_transfer_bytes",
        ),
    )
    run_keys = sorted(
        {
            (
                str(item.get("run_id", "")),
                str(item.get("baseline_id", "")),
                str(item.get("trace_id", "")),
            )
            for values in records.values()
            for item in values
            if item.get("run_id")
        }
    )
    run_metric_rows = []
    for run_id, baseline_id, trace_id in run_keys:
        def selected(record_type: str):
            return [
                item
                for item in records[record_type]
                if str(item.get("run_id", "")) == run_id
                and str(item.get("baseline_id", "")) == baseline_id
                and str(item.get("trace_id", "")) == trace_id
            ]

        plans = selected("plan_decision")
        resources = selected("resource_sample")
        artifacts = selected("artifact_event")
        latencies = sorted(
            float(item.get("planning_latency_ms") or 0) for item in plans
        )
        run_metric_rows.append(
            {
                "run_id": run_id,
                "baseline_id": baseline_id,
                "trace_id": trace_id,
                "predicate_observations": len(selected("predicate_observation")),
                "accepted_events": sum(
                    bool(item.get("accepted", True))
                    for item in selected("complex_event_result")
                ),
                "predicate_demands": len(selected("predicate_demand")),
                "plan_decisions": len(plans),
                "p95_planning_latency_ms": _percentile(latencies, 0.95),
                "provider_starts": sum(
                    str(item.get("lifecycle_event", "")).upper()
                    in {"STARTED", "READY", "ACTIVE"}
                    for item in selected("provider_lifecycle")
                ),
                "cpu_seconds": round(
                    sum(float(item.get("cpu_time_seconds") or 0) for item in resources),
                    6,
                ),
                "gpu_energy_joules": round(
                    sum(float(item.get("gpu_energy_joules") or 0) for item in resources),
                    6,
                ),
                "gpu_seconds": round(
                    sum(float(item.get("gpu_time_seconds") or 0) for item in resources),
                    6,
                ),
                "network_bytes": sum(
                    int(item.get("network_tx_bytes") or 0)
                    + int(item.get("network_rx_bytes") or 0)
                    for item in resources
                ),
                "artifact_bytes": sum(
                    int(item.get("bytes") or 0) for item in artifacts
                ),
                "retrospective_attempts": len(selected("retrospective_attempt")),
            }
        )
    metric_fields = (
        "run_id",
        "baseline_id",
        "trace_id",
        "predicate_observations",
        "accepted_events",
        "predicate_demands",
        "plan_decisions",
        "p95_planning_latency_ms",
        "provider_starts",
        "cpu_seconds",
        "gpu_energy_joules",
        "gpu_seconds",
        "network_bytes",
        "artifact_bytes",
        "retrospective_attempts",
    )
    _write_csv(output / "run_metrics.csv", run_metric_rows, metric_fields)
    confidence_rows = []
    by_baseline: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in run_metric_rows:
        by_baseline[str(row["baseline_id"])].append(row)
    for baseline, rows in sorted(by_baseline.items()):
        for metric in (
            "accepted_events",
            "p95_planning_latency_ms",
            "cpu_seconds",
            "network_bytes",
        ):
            interval = confidence_interval(
                tuple(float(row[metric]) for row in rows)
            )
            confidence_rows.append(
                {
                    "baseline_id": baseline,
                    "metric": metric,
                    **interval.model_dump(),
                }
            )
    _write_csv(
        output / "confidence_intervals.csv",
        confidence_rows,
        (
            "baseline_id",
            "metric",
            "count",
            "mean",
            "lower_95",
            "upper_95",
        ),
    )
    paired_rows = []
    treatment_rows = by_baseline.get("FABLE", [])
    for control, rows in sorted(by_baseline.items()):
        if control == "FABLE":
            continue
        for metric, lower_is_better in (
            ("accepted_events", False),
            ("p95_planning_latency_ms", True),
            ("cpu_seconds", True),
            ("network_bytes", True),
        ):
            treatment = _mean_by_trace(treatment_rows, metric)
            baseline = _mean_by_trace(rows, metric)
            comparison = paired_comparison(
                treatment,
                baseline,
                lower_is_better=lower_is_better,
            )
            paired_rows.append(
                {
                    "treatment": "FABLE",
                    "control": control,
                    "metric": metric,
                    "lower_is_better": lower_is_better,
                    "pair_count": comparison.pair_count,
                    "treatment_mean": comparison.treatment_mean,
                    "control_mean": comparison.control_mean,
                    "mean_difference": comparison.difference.mean,
                    "difference_lower_95": comparison.difference.lower_95,
                    "difference_upper_95": comparison.difference.upper_95,
                    "treatment_better_fraction": comparison.treatment_better_fraction,
                }
            )
    _write_csv(
        output / "paired_comparisons.csv",
        paired_rows,
        (
            "treatment",
            "control",
            "metric",
            "lower_is_better",
            "pair_count",
            "treatment_mean",
            "control_mean",
            "mean_difference",
            "difference_lower_95",
            "difference_upper_95",
            "treatment_better_fraction",
        ),
    )
    _write_csv(output / "run_exclusions.csv", exclusions, ("path", "reason"))
    report = {
        "schema_version": "fable.evaluation_report.v1",
        "included_run_ids": sorted(included_runs),
        "input_roots": [str(item) for item in roots],
        "record_counts": {
            key: len(value) for key, value in sorted(records.items())
        },
        "summary_csv": str(output / "summary.csv"),
        "run_metrics_csv": str(output / "run_metrics.csv"),
        "run_exclusions_csv": str(output / "run_exclusions.csv"),
        "confidence_intervals_csv": str(output / "confidence_intervals.csv"),
        "paired_comparisons_csv": str(output / "paired_comparisons.csv"),
        "excluded_input_count": len(exclusions),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _mean_by_trace(
    rows: list[dict[str, object]],
    metric: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trace_id"])].append(float(row[metric]))
    return {
        trace_id: sum(values) / len(values)
        for trace_id, values in grouped.items()
    }


def _mean(values: Iterable[object]) -> float:
    numbers = [float(value) for value in values if value is not None]
    return round(sum(numbers) / len(numbers), 6) if numbers else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return round(values[int(round((len(values) - 1) * fraction))], 6)


def _write_csv(
    path: Path,
    rows: Iterable[dict[str, object]],
    fields: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_scaling_report(
    samples: Iterable[LoadSample],
    output_dir: str | Path,
    *,
    target_timely_recall: float,
    maximum_p95_latency_ms: float,
) -> dict[str, object]:
    """Write the predeclared sustainable-load decision and all input points."""

    rows = tuple(samples)
    result = maximum_sustainable_load(
        rows,
        target_timely_recall=target_timely_recall,
        maximum_p95_latency_ms=maximum_p95_latency_ms,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "scaling_samples.csv",
        (item.model_dump() for item in rows),
        ("workload", "timely_recall", "p95_latency_ms", "completed"),
    )
    document = {
        "schema_version": "fable.scaling_report.v1",
        **result.model_dump(),
        "scaling_samples_csv": str(output / "scaling_samples.csv"),
    }
    (output / "scaling_summary.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document
