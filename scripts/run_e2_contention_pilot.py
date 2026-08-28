#!/usr/bin/env python3
"""Run the fail-closed, planning-only redesigned E2 pilot."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.baselines.factory import build_baseline_policy
from evaluation.concurrent_admission import (
    fractional_reservation,
    joint_batch_case,
    sequential_committed_admission,
)
from evaluation.deployment_artifacts import load_deployment_artifacts
from evaluation.e2_discrimination import (
    evaluate_e2_discrimination,
    evaluate_e2_network_discrimination,
)
from evaluation.e2_snapshots import export_checkpoint_snapshot, load_checkpoint_snapshot
from evaluation.planning_cases import compile_evaluation_planning_case
from evaluation.schemas import BaselineId
from fable.distributed.config import load_deployment_graph
from fable.planning import BoundedLabelPlanner
from fable.planning.beam_search import BeamSearchConfig
from fable.planning.alternative_graph import AlternativeBuildConfig, PhysicalAlternativeGraphBuilder
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import ComputeCapacity, NetworkLink, PhysicalAlternativeGraph
from fable.planning.provider_registry import ProviderRegistry


TRACE_ID = "20260414-cross-sensor-robbery-robbery-32-r032"
POLICIES = (
    BaselineId.B4_GREEDY_FRONTIER,
    BaselineId.FABLE,
    BaselineId.O1_EXHAUSTIVE_ORACLE,
)
CONDITIONS = {
    "R0": None,
    "RJ50": ("fraction", "physical_jetson", 0.50),
    "RJCPU75": ("cpu", "physical_jetson", 0.75),
    "RJ80": ("fraction", "physical_jetson", 0.80),
    "RS50": ("fraction", "physical_rpi", 0.50),
    "RPC50": ("fraction", "physical_host", 0.50),
}


def _catalog_row() -> dict[str, object]:
    for line in (ROOT / "evaluation/manifests/workloads/ground_truth.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["experiment_id"] == TRACE_ID:
            return row
    raise RuntimeError(f"missing trace {TRACE_ID}")


def _bounded_diverse_graph(graph: PhysicalAlternativeGraph, limit: int = 4) -> PhysicalAlternativeGraph:
    by_demand: dict[object, list[object]] = {}
    for alternative in sorted(
        graph.alternatives,
        key=lambda item: (item.estimated_completion_ms, item.estimated_transfer_bytes, item.alternative_id),
    ):
        by_demand.setdefault(alternative.demand_id, []).append(alternative)
    retained = []
    for values in by_demand.values():
        by_signature = {}
        for item in values:
            signature = tuple((step.provider_id, step.node_id) for step in item.step_placements)
            by_signature.setdefault(signature, item)
        # Round-robin the terminal placement so the beam cannot lose all tier diversity.
        by_terminal = {}
        for item in by_signature.values():
            by_terminal.setdefault(item.step_placements[-1].node_id, []).append(item)
        candidates = []
        while by_terminal and len(candidates) < limit:
            for node_id in sorted(tuple(by_terminal)):
                values_for_node = by_terminal[node_id]
                candidates.append(values_for_node.pop(0))
                if not values_for_node:
                    del by_terminal[node_id]
                if len(candidates) == limit:
                    break
        retained.extend(candidates)
    return graph.model_copy(update={"alternatives": tuple(retained)})


def _compact_network(deployment: DeploymentGraph) -> DeploymentGraph:
    links = tuple(
        NetworkLink(
            source_node_id=item.source_node_id,
            target_node_id=item.target_node_id,
            latency_ms=max(item.latency_ms, 25),
            bandwidth_mbps=min(item.bandwidth_mbps, 20),
            policy_tags=tuple(sorted(set(item.policy_tags) | {"e2-compact-constrained"})),
            available=item.available,
            bidirectional=item.bidirectional,
        )
        for item in deployment.links
    )
    return DeploymentGraph(nodes=deployment.nodes.values(), sources=deployment.sources.values(), links=links, resource_pools=deployment.resource_pools)


def _remap_to_physical(demand):
    """Map a recorded camera demand to the measured Pi->Jetson ingress path."""
    return demand.model_copy(
        update={
            "eligible_source_ids": (
                "physical_jetson_camera_ingress",
                "physical_rpi_microphone_replay",
            ),
            "eligible_regions": ("lab",),
            "source_preferences": (),
        }
    )


def _policy(policy_id, *, providers, artifacts, deployment, beam_width: int = 8):
    return build_baseline_policy(
        policy_id,
        planner=BoundedLabelPlanner(
            provider_registry=providers,
            artifact_catalog=artifacts,
            deployment=deployment,
            config=BeamSearchConfig(beam_width=beam_width, run_oracle=False),
        ),
        static_registry_path=ROOT / "evaluation/manifests/baselines/static_pipelines.yaml",
    )


def _condition_reservations(deployment: DeploymentGraph, spec) -> dict[str, ComputeCapacity]:
    if spec is None:
        return {}
    kind, node_id, fraction = spec
    if kind == "fraction":
        return fractional_reservation(deployment, node_id=node_id, fraction=fraction)
    pool_id, capacity = deployment.resource_pool(node_id)
    if kind == "cpu":
        return {
            pool_id: ComputeCapacity(
                cpu_cores=capacity.cpu_cores * fraction,
                memory_mb=0,
                gpu_memory_mb=0,
            )
        }
    raise ValueError(f"unknown E2 reservation kind: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument(
        "--hypotheses",
        default="1,2,4",
        help="comma-separated concurrency levels",
    )
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument(
        "--oracle-max-hypotheses",
        type=int,
        default=4,
        help="run exhaustive O1 only through this concurrency level",
    )
    parser.add_argument(
        "--condition",
        action="append",
        choices=tuple(CONDITIONS),
        help="run only this reservation condition; may be repeated",
    )
    parser.add_argument(
        "--network",
        action="append",
        choices=("N0", "NC"),
        help="run only this network condition; may be repeated",
    )
    parser.add_argument(
        "--runtime-snapshot",
        type=Path,
        help="validated typed live-runtime frontier satisfying the headline gate",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime_snapshot = None
    if args.runtime_snapshot is not None:
        runtime_snapshot = args.runtime_snapshot.resolve(strict=True)
        load_checkpoint_snapshot(runtime_snapshot)

    source = _catalog_row()
    now = datetime.fromisoformat(str(source["recording_start"]))
    legacy = load_deployment_graph(ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml")
    physical = load_deployment_graph(ROOT / "config/physical_devices.yaml")
    providers = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
        profiles_path=ROOT / "evaluation/manifests/providers/calibrated_desktop_profiles.json",
    )
    legacy_artifacts = load_deployment_artifacts(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment_artifacts.yaml", repository_root=ROOT
    )
    physical_artifacts = load_deployment_artifacts(
        ROOT / "config/physical_deployment_artifacts.yaml", repository_root=ROOT
    )

    # These are two different real semantic frontiers from the same recorded
    # robbery: retrospective presence and subsequent exit. E2-B admits them as
    # concurrent hypotheses; it does not claim they were one semantic checkpoint.
    hypothesis_levels = tuple(
        sorted({int(item) for item in args.hypotheses.split(",") if item.strip()})
    )
    if not hypothesis_levels or min(hypothesis_levels) < 1:
        raise SystemExit("hypothesis levels must be positive")
    semantic_members = []
    for member_index in range(max(hypothesis_levels)):
        frontier_index = member_index % 2
        semantic = compile_evaluation_planning_case(
            variant=str(source["ce_variant"]), run_id=f"e2-source-{member_index}",
            trace_id=TRACE_ID, request_id=f"e2-source-{member_index}", now=now,
            provider_registry=providers, artifact_catalog=legacy_artifacts,
            deployment=legacy, frontier_index=frontier_index,
        )
        demands = tuple(_remap_to_physical(item) for item in semantic.frontier_demands)
        semantic_members.append((semantic, demands))

    def build_cases(deployment: DeploymentGraph):
        builder = PhysicalAlternativeGraphBuilder(
            provider_registry=providers,
            artifact_catalog=physical_artifacts,
            deployment=deployment,
            config=AlternativeBuildConfig(
                max_external_assignments_per_chain=256,
                max_placement_variants_per_assignment=256,
                max_total_alternatives=4096,
                max_alternatives_per_chain=2048,
                max_candidate_nodes_per_step=3,
            ),
        )
        result = []
        for semantic, demands in semantic_members:
            graph = _bounded_diverse_graph(builder.build(demands, now=now))
            result.append(
                replace(
                    semantic,
                    frontier_demands=demands,
                    all_task_demands=demands,
                    frontier_graph=graph,
                    whole_event_graph=graph,
                    replay_supported_sensor_ids=(
                        "physical_jetson_camera_ingress",
                        "physical_rpi_microphone_replay",
                    ),
                )
            )
        return tuple(result)

    networks = {"N0": physical, "NC": _compact_network(physical)}
    cases_by_network = {
        network_id: build_cases(network)
        for network_id, network in networks.items()
    }
    for member_index, case in enumerate(cases_by_network["N0"]):
        export_checkpoint_snapshot(
            case, output / "snapshots" / f"member-{member_index:02d}.json",
            capture_kind="compiled_real_trace_frontier_with_physical_source_remap",
        )

    cells = []
    selected_networks = tuple(args.network or networks)
    selected_conditions = tuple(args.condition or CONDITIONS)
    for network_id in selected_networks:
        network = networks[network_id]
        cases = cases_by_network[network_id]
        for condition_id in selected_conditions:
            reservation_spec = CONDITIONS[condition_id]
            reservations = _condition_reservations(network, reservation_spec)
            snapshot = network.with_resource_reservations(reservations)
            for hypotheses in hypothesis_levels:
                batch = joint_batch_case(cases[:hypotheses], run_id=f"e2-{network_id}-{condition_id}-h{hypotheses}", request_id=f"e2-{network_id}-{condition_id}-h{hypotheses}")
                decisions = []
                instrumentation = {}
                policies = tuple(
                    item
                    for item in POLICIES
                    if not (
                        item == BaselineId.O1_EXHAUSTIVE_ORACLE
                        and (args.skip_oracle or hypotheses > args.oracle_max_hypotheses)
                    )
                )
                for baseline in policies:
                    tracemalloc.start()
                    started_cpu = time.process_time_ns()
                    started = time.perf_counter_ns()
                    decision = _policy(
                        baseline, providers=providers, artifacts=physical_artifacts,
                        deployment=snapshot, beam_width=args.beam_width,
                    ).plan(batch)
                    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                    cpu_ms = (time.process_time_ns() - started_cpu) / 1_000_000
                    _current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    decisions.append(decision)
                    instrumentation[baseline.value] = {"wall_ms": elapsed_ms, "cpu_ms": cpu_ms, "peak_memory_mb": peak / 1048576}
                gate = evaluate_e2_discrimination(batch, deployment=snapshot, decisions=decisions)
                sequential = sequential_committed_admission(
                    cases[:hypotheses], deployment=network,
                    initial_reservations=reservations,
                    policy_factory=lambda d: _policy(
                        BaselineId.FABLE, providers=providers,
                        artifacts=physical_artifacts, deployment=d,
                        beam_width=args.beam_width,
                    ),
                )
                cells.append({
                    "network_id": network_id, "condition_id": condition_id,
                    "hypotheses": hypotheses, "reservations": {k: v.model_dump(mode="json") for k, v in reservations.items()},
                    "gate": gate, "instrumentation": instrumentation,
                    "joint_decisions": [item.model_dump(mode="json") for item in decisions],
                    "sequential_fable": [{"request_id": item.request_id, "admitted": item.admitted, "reason": item.rejection_reason, "decision": item.decision.model_dump(mode="json")} for item in sequential],
                })

    valid = [f"{c['network_id']}/{c['condition_id']}/H{c['hypotheses']}" for c in cells if c["gate"]["valid"]]
    network_gate = evaluate_e2_network_discrimination(cells)
    result = {
        "schema_version": "fable.e2_contention_pilot.v2",
        "trace_id": TRACE_ID,
        "provider_execution": False,
        "raw_video_transfer_allowed": False,
        "physical_deployment": "config/physical_devices.yaml",
        "beam_width": args.beam_width,
        "hypothesis_levels": hypothesis_levels,
        "oracle_enabled": not args.skip_oracle,
        "oracle_max_hypotheses": args.oracle_max_hypotheses,
        "headline_runtime_frontiers_exported": runtime_snapshot is not None,
        "runtime_snapshot": str(runtime_snapshot) if runtime_snapshot else None,
        "valid_discriminating_cells": valid,
        "network_discrimination": network_gate,
        "network_mechanism_ready": network_gate["valid"],
        "full_experiment_ready": bool(valid) and runtime_snapshot is not None,
        "mechanism_pilot_ready": bool(valid),
        "cells": cells,
    }
    (output / "pilot.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
