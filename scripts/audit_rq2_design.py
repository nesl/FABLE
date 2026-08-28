#!/usr/bin/env python3
"""Audit whether the proposed RQ2/E2 matrix contains real planning tradeoffs.

This is planning-only: it does not replay recordings or alter physical devices.
The manifest explicitly forbids synthetic raw-video transfer and redundant views.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.baselines.factory import build_baseline_policy
from evaluation.deployment_artifacts import load_deployment_artifacts
from evaluation.networking import apply_netwaggle_profile, load_netwaggle_profile
from evaluation.planning_cases import compile_evaluation_planning_case
from evaluation.schemas import BaselineId
from fable.distributed.config import load_deployment_graph
from fable.planning import BoundedLabelPlanner
from fable.planning.alternative_graph import (
    AlternativeBuildConfig,
    PhysicalAlternativeGraphBuilder,
)
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import TransferMode
from fable.planning.provider_registry import ProviderRegistry


def _catalog() -> dict[str, dict[str, object]]:
    path = ROOT / "evaluation/manifests/workloads/ground_truth.jsonl"
    return {
        row["experiment_id"]: row
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _disconnect_node(deployment: DeploymentGraph, node_id: str) -> DeploymentGraph:
    """Remove a node's links, while leaving local sensing/compute available."""

    deployment.node(node_id)
    return DeploymentGraph(
        nodes=deployment.nodes.values(),
        sources=deployment.sources.values(),
        links=(
            link
            for link in deployment.links
            if node_id not in (link.source_node_id, link.target_node_id)
        ),
        resource_pools=deployment.resource_pools,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "evaluation/manifests/planning/rq2_audit_v2.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    boundary = manifest["claim_boundary"]
    if boundary.get("raw_video_transfer_allowed") is not False:
        raise SystemExit("RQ2 audit requires raw_video_transfer_allowed: false")
    if boundary.get("redundant_sensor_views_assumed") is not False:
        raise SystemExit("RQ2 audit requires redundant_sensor_views_assumed: false")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    catalog = _catalog()
    base_deployment = load_deployment_graph(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    providers = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
        profiles_path=ROOT / "evaluation/manifests/providers/calibrated_desktop_profiles.json",
    )
    artifacts = load_deployment_artifacts(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment_artifacts.yaml",
        repository_root=ROOT,
    )

    # Compile each semantic frontier once. Recompilation mints UUIDv7 demand IDs;
    # using fresh IDs per condition would make equal-cost tie-breaking look like
    # network-sensitive adaptation.
    semantic_cases = {}
    for case_spec in manifest["cases"]:
        source = catalog[case_spec["trace_id"]]
        now = datetime.fromisoformat(str(source["recording_start"]))
        semantic_cases[case_spec["id"]] = compile_evaluation_planning_case(
            variant=str(source["ce_variant"]),
            run_id=f"rq2-audit-canonical-{case_spec['id']}",
            trace_id=case_spec["trace_id"],
            request_id=f"rq2-audit-{case_spec['id']}",
            now=now,
            provider_registry=providers,
            artifact_catalog=artifacts,
            deployment=base_deployment,
            frontier_index=int(case_spec["frontier_index"]),
        )

    cells = []
    for condition_index, condition in enumerate(manifest["conditions"]):
        profile = load_netwaggle_profile(ROOT / condition["profile"])
        applied = apply_netwaggle_profile(
            base_deployment, profile, resource_epoch=condition_index
        )
        deployment = applied.deployment
        if condition.get("disconnect_node"):
            deployment = _disconnect_node(deployment, condition["disconnect_node"])
        for case_spec in manifest["cases"]:
            source = catalog[case_spec["trace_id"]]
            now = datetime.fromisoformat(str(source["recording_start"]))
            semantic_case = semantic_cases[case_spec["id"]]
            builder = PhysicalAlternativeGraphBuilder(
                provider_registry=providers,
                artifact_catalog=artifacts,
                deployment=deployment,
                # E2 has exactly three physical tiers. Enumerate all three;
                # the generic default of two candidate nodes can otherwise
                # hide a site-local realization whenever cloud sorts first.
                config=AlternativeBuildConfig(
                    max_external_assignments_per_chain=256,
                    max_placement_variants_per_assignment=256,
                    max_total_alternatives=2048,
                    max_alternatives_per_chain=1024,
                    max_candidate_nodes_per_step=3,
                ),
            )
            case = replace(
                semantic_case,
                run_id=f"rq2-audit-{condition['id']}-{case_spec['id']}",
                frontier_graph=builder.build(semantic_case.frontier_demands, now=now),
                whole_event_graph=builder.build(semantic_case.all_task_demands, now=now),
                resource_epoch=condition_index,
            )
            alternatives = {
                item.alternative_id: item
                for item in (*case.frontier_graph.alternatives, *case.whole_event_graph.alternatives)
            }
            reference_ambiguities = sum(
                transfer.mode == TransferMode.REMOTE_REFERENCE
                and transfer.bytes == 0
                and transfer.source_node_id != transfer.target_node_id
                for item in alternatives.values()
                for transfer in item.transfers
            )
            decisions = []
            for policy_name in manifest["policies"]:
                baseline = BaselineId(policy_name)
                planner = BoundedLabelPlanner(
                    provider_registry=providers,
                    artifact_catalog=artifacts,
                    deployment=deployment,
                )
                decision = build_baseline_policy(
                    baseline,
                    planner=planner,
                    static_registry_path=ROOT / "evaluation/manifests/baselines/static_pipelines.yaml",
                ).plan(case)
                selected = [
                    alternatives[item]
                    for item in decision.selected_alternative_ids
                    if item in alternatives
                ]
                dominance_witnesses = []
                for chosen in selected:
                    for candidate in alternatives.values():
                        if candidate.demand_id != chosen.demand_id:
                            continue
                        weakly_better = (
                            candidate.estimated_completion_ms
                            <= chosen.estimated_completion_ms
                            and candidate.estimated_transfer_bytes
                            <= chosen.estimated_transfer_bytes
                            and candidate.minimum_quality_score
                            >= chosen.minimum_quality_score
                        )
                        strictly_better = (
                            candidate.estimated_completion_ms
                            < chosen.estimated_completion_ms
                            or candidate.estimated_transfer_bytes
                            < chosen.estimated_transfer_bytes
                            or candidate.minimum_quality_score
                            > chosen.minimum_quality_score
                        )
                        if weakly_better and strictly_better:
                            dominance_witnesses.append(
                                {
                                    "selected_id": chosen.alternative_id,
                                    "candidate_id": candidate.alternative_id,
                                    "candidate_completion_ms": candidate.estimated_completion_ms,
                                    "candidate_transfer_bytes": candidate.estimated_transfer_bytes,
                                    "candidate_nodes": sorted(
                                        {step.node_id for step in candidate.step_placements}
                                    ),
                                }
                            )
                            break
                decisions.append(
                    {
                        "policy_id": policy_name,
                        "feasible": bool(decision.selected_alternative_ids),
                        "selected_alternative_ids": list(
                            decision.selected_alternative_ids
                        ),
                        "selected_chains": list(decision.selected_chain_ids),
                        "selected_nodes": list(decision.selected_node_ids),
                        "completion_ms": decision.predicted_completion_ms or 0,
                        "transfer_bytes": decision.predicted_transfer_bytes or 0,
                        "dominance_witnesses": dominance_witnesses,
                    }
                )
            cells.append(
                {
                    "case_id": case_spec["id"],
                    "trace_id": case_spec["trace_id"],
                    "frontier_index": case_spec["frontier_index"],
                    "role": case_spec["role"],
                    "condition_id": condition["id"],
                    "alternative_count": len(alternatives),
                    "alternative_chains": sorted({item.chain_id for item in alternatives.values()}),
                    "alternative_nodes": sorted({step.node_id for item in alternatives.values() for step in item.step_placements}),
                    "zero_byte_remote_reference_count": reference_ambiguities,
                    "decisions": decisions,
                }
            )

    case_results = []
    for case_spec in manifest["cases"]:
        relevant = [cell for cell in cells if cell["case_id"] == case_spec["id"]]
        policy_signatures = {}
        for cell in relevant:
            for decision in cell["decisions"]:
                policy_signatures.setdefault(decision["policy_id"], set()).add(
                    (tuple(decision["selected_chains"]), tuple(decision["selected_nodes"]), decision["feasible"])
                )
        good = next(cell for cell in relevant if cell["condition_id"] == "good_network")
        good_signatures = {
            (tuple(item["selected_chains"]), tuple(item["selected_nodes"]))
            for item in good["decisions"]
        }
        good_metrics = {
            (item["completion_ms"], item["transfer_bytes"], item["feasible"])
            for item in good["decisions"]
        }
        case_results.append(
            {
                "case_id": case_spec["id"],
                "role": case_spec["role"],
                "structurally_diverse": len(good["alternative_chains"]) > 1 or len(good["alternative_nodes"]) > 1,
                "policy_diverse_on_good_network": len(good_signatures) > 1,
                "policy_metric_diverse_on_good_network": len(good_metrics) > 1,
                "condition_sensitive_policies": sorted(
                    policy for policy, values in policy_signatures.items() if len(values) > 1
                ),
            }
        )

    primary = [item for item in case_results if item["role"] == "primary"]
    limitations = []
    if any(cell["zero_byte_remote_reference_count"] for cell in cells):
        limitations.append(
            "Remote references crossing nodes are charged zero bytes and latency only; network-volume conclusions are invalid until that semantic is resolved."
        )
    shared_pools = {
        node.resource_pool_id for node in base_deployment.nodes.values()
    }
    if len(shared_pools) == 1:
        limitations.append(
            "All logical nodes share one resource pool, so the current model cannot express independent RPi, Jetson, and PC compute contention."
        )
    for case_spec in manifest["cases"]:
        relevant = [cell for cell in cells if cell["case_id"] == case_spec["id"]]
        good = next(cell for cell in relevant if cell["condition_id"] == "good_network")
        disconnected = next(
            cell for cell in relevant if cell["condition_id"] == "cloud_disconnected"
        )
        good_by_policy = {item["policy_id"]: item for item in good["decisions"]}
        disconnected_by_policy = {
            item["policy_id"]: item for item in disconnected["decisions"]
        }
        if any(
            disconnected_by_policy[policy]["completion_ms"]
            < good_by_policy[policy]["completion_ms"]
            for policy in good_by_policy
        ):
            limitations.append(
                f"{case_spec['id']}: disconnecting cloud improves predicted completion, a cost-model paradox rather than a defensible network benefit."
            )
    result = {
        "schema_version": "fable.rq2_audit_result.v1",
        "execution_ready": False,
        "planning_tradeoff_observed": any(
            item["policy_diverse_on_good_network"] or item["condition_sensitive_policies"]
            for item in primary
        ),
        "meaningful_policy_tradeoff_observed": any(
            item["policy_metric_diverse_on_good_network"] for item in primary
        ),
        "claim_boundary": boundary,
        "case_results": case_results,
        "limitations": limitations,
        "cells": cells,
    }
    result["execution_ready"] = (
        result["meaningful_policy_tradeoff_observed"] and not limitations
    )
    (output / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
