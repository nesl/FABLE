#!/usr/bin/env python3
"""Run the bounded RQ2 planner matrix without replaying sensor recordings."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.baselines.factory import build_baseline_policy
from evaluation.deployment_artifacts import load_deployment_artifacts
from evaluation.networking import (
    apply_netwaggle_profile,
    load_netwaggle_profile,
    network_condition_records,
)
from evaluation.planning_cases import compile_evaluation_planning_case
from evaluation.runner import EvaluationRunner
from evaluation.schemas import BaselineId, EvaluationMode
from fable.distributed.config import load_deployment_graph
from fable.planning import BoundedLabelPlanner
from fable.planning.beam_search import BeamSearchConfig
from fable.planning.alternative_graph import (
    AlternativeBuildConfig,
    PhysicalAlternativeGraphBuilder,
)
from fable.planning.provider_registry import ProviderRegistry

DEFAULT_TRACES = (
    "20241008-route-convoy-10-r022",
    "20250812-talking-rendezvous-rendezvous-brianjulian-1-r029",
    "20260414-cross-sensor-robbery-robbery-32-r032",
)
DEFAULT_POLICIES = (
    BaselineId.B2_FRONTIER_FIXED_REALIZATION,
    BaselineId.B4_GREEDY_FRONTIER,
    BaselineId.FABLE,
    BaselineId.O1_EXHAUSTIVE_ORACLE,
)
DEFAULT_PROFILES = (
    ROOT / "netwaggle/configs/profiles/good_network.json",
    ROOT / "netwaggle/configs/profiles/constrained_bandwidth.json",
    ROOT / "netwaggle/configs/profiles/high_latency_cloud.json",
)


def _catalog() -> dict[str, dict[str, object]]:
    result = {}
    path = ROOT / "evaluation/manifests/workloads/ground_truth.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[row["experiment_id"]] = row
    return result


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.5)))
    return ordered[index]


def _write_report(rows: list[dict[str, object]], output: Path) -> None:
    fields = list(rows[0])
    with (output / "planning_replay_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["profile_id"]), str(row["baseline_id"]))].append(row)
    summaries = []
    for (profile, baseline), items in sorted(grouped.items()):
        latencies = [float(item["planning_latency_ms"]) for item in items]
        summaries.append(
            {
                "profile_id": profile,
                "baseline_id": baseline,
                "runs": len(items),
                "feasible_runs": sum(bool(item["feasible"]) for item in items),
                "predicted_deadline_misses": sum(
                    item["predicted_slack_ms"] != ""
                    and int(item["predicted_slack_ms"]) < 0
                    for item in items
                ),
                "planning_latency_mean_ms": round(statistics.fmean(latencies), 6),
                "planning_latency_p95_ms": round(_percentile(latencies, 0.95), 6),
                "predicted_completion_mean_ms": round(
                    statistics.fmean(
                        float(item["predicted_completion_ms"]) for item in items
                    ),
                    3,
                ),
                "predicted_compute_mean_ms": round(
                    statistics.fmean(
                        float(item["predicted_compute_ms"]) for item in items
                    ),
                    3,
                ),
                "predicted_transfer_mean_bytes": round(
                    statistics.fmean(
                        float(item["predicted_transfer_bytes"]) for item in items
                    ),
                    3,
                ),
                "labels_generated_mean": round(
                    statistics.fmean(float(item["labels_generated"]) for item in items),
                    3,
                ),
                "labels_pruned_mean": round(
                    statistics.fmean(float(item["labels_pruned"]) for item in items), 3
                ),
                "labels_retained_mean": round(
                    statistics.fmean(float(item["labels_retained"]) for item in items),
                    3,
                ),
            }
        )
    with (output / "planning_replay_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "fable.rq2_planning_replay.v1",
                "evaluation_mode": EvaluationMode.PLANNING_REPLAY.value,
                "sensor_recordings_replayed": False,
                "rows": len(rows),
                "traces": len({row["trace_id"] for row in rows}),
                "profiles": len({row["profile_id"] for row in rows}),
                "policies": len({row["baseline_id"] for row in rows}),
                "repetitions": len({row["repetition"] for row in rows}),
                "feasible_rows": sum(bool(row["feasible"]) for row in rows),
                "summary_csv": "planning_replay_summary.csv",
                "rows_csv": "planning_replay_rows.csv",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument(
        "--capture-e2-snapshots",
        action="store_true",
        help="persist complete typed checkpoint inputs for redesigned E2 replay",
    )
    parser.add_argument("--trace", action="append")
    parser.add_argument("--profile", type=Path, action="append")
    parser.add_argument(
        "--policy", choices=[item.value for item in DEFAULT_POLICIES], action="append"
    )
    parser.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="write results even when no tested policy changes physical realization",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    catalog = _catalog()
    traces = tuple(args.trace or DEFAULT_TRACES)
    profiles = tuple(args.profile or DEFAULT_PROFILES)
    policies = tuple(
        BaselineId(item)
        for item in (args.policy or [p.value for p in DEFAULT_POLICIES])
    )
    missing = sorted(set(traces) - set(catalog))
    if missing:
        raise SystemExit("unknown traces: " + ", ".join(missing))

    base_deployment = load_deployment_graph(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    providers = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
        profiles_path=ROOT
        / "evaluation/manifests/providers/calibrated_desktop_profiles.json",
    )
    artifacts = load_deployment_artifacts(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment_artifacts.yaml",
        repository_root=ROOT,
    )
    trace_cases = {}
    for trace_id in traces:
        source = catalog[trace_id]
        now = datetime.fromisoformat(str(source["recording_start"]))
        trace_cases[trace_id] = compile_evaluation_planning_case(
            variant=str(source["ce_variant"]),
            run_id="rq2-planning-canonical",
            trace_id=trace_id,
            request_id=f"rq2-planning-{trace_id}-request",
            now=now,
            provider_registry=providers,
            artifact_catalog=artifacts,
            deployment=base_deployment,
        )
    rows: list[dict[str, object]] = []
    design_cells: list[dict[str, object]] = []
    for profile_index, profile_path in enumerate(profiles):
        profile = load_netwaggle_profile(profile_path)
        applied = apply_netwaggle_profile(
            base_deployment, profile, resource_epoch=profile_index
        )
        for trace_id in traces:
            source = catalog[trace_id]
            now = datetime.fromisoformat(str(source["recording_start"]))
            canonical_request_id = f"rq2-planning-{trace_id}-request"
            semantic_case = trace_cases[trace_id]
            graph_builder = PhysicalAlternativeGraphBuilder(
                provider_registry=providers,
                artifact_catalog=artifacts,
                deployment=applied.deployment,
                config=AlternativeBuildConfig(
                    max_external_assignments_per_chain=256,
                    max_placement_variants_per_assignment=256,
                    max_total_alternatives=2048,
                    max_alternatives_per_chain=1024,
                    # RQ2's three tiers must all enter the choice set.
                    max_candidate_nodes_per_step=3,
                ),
            )
            canonical_case = replace(
                semantic_case,
                frontier_graph=graph_builder.build(
                    semantic_case.frontier_demands, now=now
                ),
                whole_event_graph=graph_builder.build(
                    semantic_case.all_task_demands, now=now
                ),
                resource_epoch=profile_index,
            )
            alternatives = (
                *canonical_case.frontier_graph.alternatives,
                *canonical_case.whole_event_graph.alternatives,
            )
            design_cells.append(
                {
                    "trace_id": trace_id,
                    "profile_id": profile.profile_id,
                    "alternative_count": len(alternatives),
                    "has_nonzero_transfer": any(
                        item.estimated_transfer_bytes > 0 for item in alternatives
                    ),
                    "has_remote_placement": any(
                        any(
                            placement.node_id != "dvpg_gq_orin_11"
                            for placement in item.step_placements
                        )
                        for item in alternatives
                    ),
                    "completion_range_ms": (
                        max(item.estimated_completion_ms for item in alternatives)
                        - min(item.estimated_completion_ms for item in alternatives)
                        if alternatives
                        else 0
                    ),
                }
            )
            for baseline_id in policies:
                for repetition in range(1, args.repetitions + 1):
                    run_id = f"rq2-planning-{profile.profile_id}-{baseline_id.value}-{trace_id}-r{repetition}"
                    request_id = canonical_request_id
                    # Repetitions replay one immutable planning input. Recompiling
                    # would mint fresh UUIDv7 demand IDs and turn tie-breaking into
                    # an accidental source of experimental nondeterminism.
                    case = replace(canonical_case, run_id=run_id)
                    planner = BoundedLabelPlanner(
                        provider_registry=providers,
                        artifact_catalog=artifacts,
                        deployment=applied.deployment,
                        config=BeamSearchConfig(beam_width=args.beam_width),
                    )
                    policy = build_baseline_policy(
                        baseline_id,
                        planner=planner,
                        static_registry_path=ROOT
                        / "evaluation/manifests/baselines/static_pipelines.yaml",
                    )
                    run_dir = output / "records" / run_id
                    runner = EvaluationRunner(
                        run_dir,
                        mode=EvaluationMode.PLANNING_REPLAY,
                        capture_e2_snapshots=args.capture_e2_snapshots,
                    )
                    runner.record_predicate_demands(case, baseline_id=baseline_id)
                    runner.record_network_conditions(
                        network_condition_records(
                            applied,
                            run_id=run_id,
                            baseline_id=baseline_id,
                            trace_id=trace_id,
                            request_id=request_id,
                            event_time=now,
                        )
                    )
                    tracemalloc.start()
                    cpu_started_ns = time.process_time_ns()
                    decision = runner.run_planning_case(policy, case)
                    planner_cpu_ms = (time.process_time_ns() - cpu_started_ns) / 1_000_000
                    _current_memory, peak_memory = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    rows.append(
                        {
                            "run_id": run_id,
                            "trace_id": trace_id,
                            "campaign_year": source["campaign_year"],
                            "ce_variant": source["ce_variant"],
                            "profile_id": profile.profile_id,
                            "baseline_id": baseline_id.value,
                            "repetition": repetition,
                            "feasible": bool(decision.selected_alternative_ids),
                            "planning_latency_ms": round(
                                decision.planning_latency_ms, 6
                            ),
                            "planner_cpu_ms": round(planner_cpu_ms, 6),
                            "planner_peak_memory_mb": round(
                                peak_memory / (1024 * 1024), 6
                            ),
                            "beam_width": args.beam_width,
                            "predicted_completion_ms": decision.predicted_completion_ms
                            or 0,
                            "predicted_compute_ms": decision.predicted_compute_ms or 0,
                            "predicted_transfer_bytes": decision.predicted_transfer_bytes
                            or 0,
                            "predicted_slack_ms": decision.predicted_slack_ms
                            if decision.predicted_slack_ms is not None
                            else "",
                            "labels_generated": decision.labels_generated,
                            "labels_pruned": decision.labels_pruned,
                            "labels_retained": decision.labels_retained,
                            "selected_alternatives": len(
                                decision.selected_alternative_ids
                            ),
                            "selected_alternative_ids": ";".join(
                                decision.selected_alternative_ids
                            ),
                            "selected_nodes": ";".join(decision.selected_node_ids),
                            "selected_chains": ";".join(decision.selected_chain_ids),
                        }
                    )
    signatures: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
    for row in rows:
        signatures[(str(row["trace_id"]), str(row["baseline_id"]))].add(
            (
                str(row["selected_alternative_ids"]),
                str(row["selected_nodes"]),
                str(row["selected_chains"]),
            )
        )
    trace_validity = []
    for trace_id in traces:
        cells = [item for item in design_cells if item["trace_id"] == trace_id]
        changed = sorted(
            baseline
            for (candidate_trace, baseline), values in signatures.items()
            if candidate_trace == trace_id and len(values) > 1
        )
        trace_validity.append(
            {
                "trace_id": trace_id,
                "all_profiles_have_remote_alternative": all(
                    bool(item["has_remote_placement"]) for item in cells
                ),
                "all_profiles_have_nonzero_transfer_alternative": all(
                    bool(item["has_nonzero_transfer"]) for item in cells
                ),
                "profile_sensitive_policies": changed,
            }
        )
    validity = {
        "schema_version": "fable.rq2_design_validity.v1",
        "valid": (
            all(
                item["all_profiles_have_remote_alternative"]
                and item["all_profiles_have_nonzero_transfer_alternative"]
                for item in trace_validity
            )
            and any(item["profile_sensitive_policies"] for item in trace_validity)
        ),
        "claim_boundary": (
            "At least one workload must change physical realization; workloads "
            "without a change are retained and reported as negative controls."
        ),
        "traces": trace_validity,
        "cells": design_cells,
    }
    (output / "design_validity.json").write_text(
        json.dumps(validity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not validity["valid"] and not args.allow_degenerate:
        raise RuntimeError(
            "RQ2 design is degenerate; see design_validity.json or use "
            "--allow-degenerate only for diagnostic runs"
        )
    _write_report(rows, output)
    print(json.dumps(json.loads((output / "summary.json").read_text()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
