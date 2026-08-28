#!/usr/bin/env python3
"""Run a bounded provider-placement test without replaying sensor data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.baselines.factory import build_baseline_policy
from evaluation.deployment_artifacts import load_deployment_artifacts
from evaluation.networking import apply_netwaggle_profile, load_netwaggle_profile
from evaluation.plan_discrimination import (
    alternative_signature,
    load_plan_discrimination_manifest,
    load_synthetic_profiles,
    validate_expected_plans,
    validate_transition_behavior,
)
from evaluation.planning_cases import (
    compile_evaluation_planning_case,
    executable_runtime_graph,
)
from evaluation.schemas import BaselineId
from fable.distributed.config import ProviderRuntimeResolver, load_deployment_graph
from fable.planning import BoundedLabelPlanner
from fable.planning.alternative_graph import AlternativeBuildConfig, PhysicalAlternativeGraphBuilder
from fable.planning.provider_registry import ProviderRegistry
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import NetworkLink


def _catalog() -> dict[str, dict[str, object]]:
    rows = {}
    source = ROOT / "evaluation/manifests/workloads/ground_truth.jsonl"
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["experiment_id"])] = row
    return rows


def _permit_raw_transfer(demands, enabled: bool):
    if not enabled:
        return demands
    return tuple(
        demand.model_copy(
            update={
                "hard_constraints": demand.hard_constraints.model_copy(
                    update={"raw_data_must_remain_local": False}
                )
            }
        )
        for demand in demands
    )


def _two_camera_fixture(deployment, runtime_resolver):
    """Clone the smoke camera as a distinct second sensor for mechanism tests."""
    first_node = deployment.node("dvpg_gq_orin_11")
    first_source = deployment.source("orin11_camera")
    second_node_id = "dvpg_gq_orin_12"
    second_source_id = "orin12_camera"
    fixture = DeploymentGraph(
        nodes=(
            *deployment.nodes.values(),
            first_node.model_copy(update={"node_id": second_node_id}),
        ),
        sources=(
            *deployment.sources.values(),
            first_source.model_copy(
                update={"source_id": second_source_id, "node_id": second_node_id}
            ),
        ),
        links=(
            *deployment.links,
            NetworkLink(
                source_node_id=second_node_id,
                target_node_id="x86server",
                latency_ms=2,
                bandwidth_mbps=1000,
                bidirectional=True,
            ),
        ),
        resource_pools=deployment.resource_pools,
    )
    runtime_map = {
        (item.node_id, item.provider_id): item for item in runtime_resolver.runtimes
    }
    for item in runtime_resolver.runtimes:
        if item.node_id == "dvpg_gq_orin_11":
            clone = item.model_copy(update={"node_id": second_node_id})
            runtime_map[(second_node_id, clone.provider_id)] = clone
    return fixture, ProviderRuntimeResolver(runtime_map)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "evaluation/manifests/adaptation/plan_discrimination.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_plan_discrimination_manifest(args.manifest)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    catalog = _catalog()
    source = catalog[manifest.trace_id]
    now = datetime.fromisoformat(str(source["recording_start"]))
    deployment = load_deployment_graph(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    calibrated_path = (
        ROOT / "evaluation/manifests/providers/calibrated_desktop_profiles.json"
    )
    profiles = load_synthetic_profiles(calibrated_path, manifest.timing_multipliers)
    providers = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
        profiles=profiles,
    )
    artifacts = load_deployment_artifacts(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment_artifacts.yaml",
        repository_root=ROOT,
    )
    runtime_resolver = ProviderRuntimeResolver.from_yaml(
        ROOT / "iobt-minimal-ce-replay/config/fable_provider_runtimes.yaml"
    )
    if manifest.two_camera_fixture:
        deployment, runtime_resolver = _two_camera_fixture(deployment, runtime_resolver)
    base_case = compile_evaluation_planning_case(
        variant=manifest.variant_override or str(source["ce_variant"]),
        run_id="plan-discrimination",
        trace_id=manifest.trace_id,
        request_id="plan-discrimination-request",
        now=now,
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
        frontier_index=manifest.semantic_frontier_index,
    )
    if manifest.include_chain_ids:
        providers = ProviderRegistry(
            data_types=providers.data_types,
            providers=providers.providers,
            chains={
                chain_id: providers.chain(chain_id)
                for chain_id in manifest.include_chain_ids
            },
            profiles=profiles,
        )
    frontier_demands = _permit_raw_transfer(
        base_case.frontier_demands, manifest.permit_synthetic_raw_transfer
    )
    all_demands = _permit_raw_transfer(
        base_case.all_task_demands, manifest.permit_synthetic_raw_transfer
    )
    if manifest.two_camera_fixture:
        camera_sources = ("orin11_camera", "orin12_camera")
        allowed_nodes = tuple(deployment.nodes)
        frontier_demands = tuple(
            item.model_copy(
                update={
                    "eligible_source_ids": camera_sources,
                    "hard_constraints": item.hard_constraints.model_copy(
                        update={"allowed_node_ids": allowed_nodes}
                    ),
                }
            )
            for item in frontier_demands
        )
        all_demands = tuple(
            item.model_copy(
                update={
                    "eligible_source_ids": camera_sources,
                    "hard_constraints": item.hard_constraints.model_copy(
                        update={"allowed_node_ids": allowed_nodes}
                    ),
                }
            )
            for item in all_demands
        )

    rows = []
    alternatives_report = []
    alternatives_by_profile = {}
    for epoch, relative_profile in enumerate(manifest.profiles):
        network_profile = load_netwaggle_profile(ROOT / relative_profile)
        applied = apply_netwaggle_profile(deployment, network_profile, resource_epoch=epoch)
        builder = PhysicalAlternativeGraphBuilder(
            provider_registry=providers,
            artifact_catalog=artifacts,
            deployment=applied.deployment,
            config=AlternativeBuildConfig(
                max_external_assignments_per_chain=16,
                max_total_alternatives=128,
                max_alternatives_per_chain=64,
                max_placement_variants_per_assignment=32,
                max_candidate_nodes_per_step=3,
                # A live camera stream must pay serialization cost. Remote
                # references are meaningful for retained artifacts, not for
                # this synthetic live-input timing experiment.
                allow_remote_reference=False,
            ),
        )
        frontier_graph = executable_runtime_graph(
            builder.build(frontier_demands, now=now),
            runtime_resolver=runtime_resolver,
        )
        whole_graph = executable_runtime_graph(
            builder.build(all_demands, now=now),
            runtime_resolver=runtime_resolver,
        )
        case = replace(
            base_case,
            frontier_demands=frontier_demands,
            all_task_demands=all_demands,
            frontier_graph=frontier_graph,
            whole_event_graph=whole_graph,
            resource_epoch=epoch,
        )
        alternatives = {
            item.alternative_id: item
            for item in (*frontier_graph.alternatives, *whole_graph.alternatives)
        }
        alternatives_by_profile[network_profile.profile_id] = alternatives
        alternatives_report.append(
            {
                "profile_id": network_profile.profile_id,
                "alternatives": [
                    alternative_signature(item)
                    for item in sorted(
                        alternatives.values(), key=lambda item: item.estimated_completion_ms
                    )
                ],
            }
        )
        for policy_name in manifest.policies:
            baseline = BaselineId(policy_name)
            planner = BoundedLabelPlanner(
                provider_registry=providers,
                artifact_catalog=artifacts,
                deployment=applied.deployment,
            )
            policy = build_baseline_policy(
                baseline,
                planner=planner,
                static_registry_path=ROOT
                / "evaluation/manifests/baselines/static_pipelines.yaml",
            )
            for repetition in range(1, manifest.repetitions + 1):
                decision = policy.plan(case)
                selected = [
                    alternatives[item]
                    for item in decision.selected_alternative_ids
                    if item in alternatives
                ]
                detector_alt = next(
                    (
                        item
                        for item in selected
                        if (
                            manifest.target_chain_id is None
                            or item.chain_id == manifest.target_chain_id
                        )
                        and any(
                            step.provider_id.startswith("yolo_")
                            for step in item.step_placements
                        )
                    ),
                    selected[0] if selected else None,
                )
                signature = alternative_signature(detector_alt) if detector_alt else {}
                selected_providers = sorted(
                    {
                        step.provider_id
                        for item in selected
                        for step in item.step_placements
                    }
                )
                all_placements = [
                    {
                        "step_id": step.step_id,
                        "provider_id": step.provider_id,
                        "node_id": step.node_id,
                        "node_class": step.node_class,
                    }
                    for item in selected
                    for step in item.step_placements
                ]
                rows.append(
                    {
                        "profile_id": network_profile.profile_id,
                        "policy_id": baseline.value,
                        "repetition": repetition,
                        "selected_alternative_ids": ";".join(
                            decision.selected_alternative_ids
                        ),
                        "selected_provider_ids": ";".join(selected_providers),
                        "selected_total_completion_ms": sum(
                            item.estimated_completion_ms for item in selected
                        ),
                        "selected_total_transfer_bytes": sum(
                            item.estimated_transfer_bytes for item in selected
                        ),
                        "all_placements": all_placements,
                        **signature,
                    }
                )

    validity = validate_expected_plans(rows, manifest)
    # Passing the placement crossover proves that the planner can react to the
    # network profile.  It does not by itself prove that FABLE differs from the
    # task/resource-adaptive baseline, so report that stronger gate separately.
    b3_id = BaselineId.B3_TASK_RESOURCE_ADAPTIVE.value
    fable_id = BaselineId.FABLE.value
    divergent_profiles = []
    discrimination_effects = []
    for relative_profile in manifest.profiles:
        profile_id = Path(relative_profile).stem
        b3 = next(
            (row for row in rows if row["profile_id"] == profile_id and row["policy_id"] == b3_id),
            None,
        )
        fable = next(
            (row for row in rows if row["profile_id"] == profile_id and row["policy_id"] == fable_id),
            None,
        )
        if b3 and fable and (
            b3["selected_provider_ids"] != fable["selected_provider_ids"]
            or b3["selected_total_completion_ms"]
            != fable["selected_total_completion_ms"]
            or b3["selected_total_transfer_bytes"]
            != fable["selected_total_transfer_bytes"]
        ):
            divergent_profiles.append(profile_id)
        if b3 and fable:
            b3_providers = set(str(b3["selected_provider_ids"]).split(";"))
            fable_providers = set(str(fable["selected_provider_ids"]).split(";"))
            b3_completion = int(b3["selected_total_completion_ms"])
            fable_completion = int(fable["selected_total_completion_ms"])
            discrimination_effects.append(
                {
                    "profile_id": profile_id,
                    "b3_provider_count": len(b3_providers),
                    "fable_provider_count": len(fable_providers),
                    "provider_count_reduction_fraction": (
                        (len(b3_providers) - len(fable_providers)) / len(b3_providers)
                        if b3_providers
                        else 0.0
                    ),
                    "b3_completion_ms": b3_completion,
                    "fable_completion_ms": fable_completion,
                    "completion_reduction_fraction": (
                        (b3_completion - fable_completion) / b3_completion
                        if b3_completion
                        else 0.0
                    ),
                    "providers_avoided_by_fable": sorted(b3_providers - fable_providers),
                }
            )
    maximum_reduction = max(
        (
            max(
                item["provider_count_reduction_fraction"],
                item["completion_reduction_fraction"],
            )
            for item in discrimination_effects
        ),
        default=0.0,
    )
    validity["b3_fable_divergent_profiles"] = divergent_profiles
    validity["b3_fable_discrimination_effects"] = discrimination_effects
    validity["b3_fable_maximum_resource_reduction_fraction"] = maximum_reduction
    validity["b3_fable_discriminating"] = bool(divergent_profiles) and (
        maximum_reduction >= manifest.minimum_b3_resource_reduction_fraction
    )
    validity["semantic_frontier_index"] = manifest.semantic_frontier_index
    validity["two_camera_fixture"] = manifest.two_camera_fixture
    # Also model one request spanning the condition transition. B2 retains its
    # admission-time physical placement; B3 and FABLE consume the replanned
    # placement for the new resource epoch. No providers are executed here.
    transition_rows = []
    first_profile = Path(manifest.profiles[0]).stem
    for policy_name in manifest.policies:
        policy_rows = [item for item in rows if item["policy_id"] == policy_name]
        admission = next(item for item in policy_rows if item["profile_id"] == first_profile)
        admission_placement = tuple(
            (item["step_id"], item["provider_id"], item["node_id"])
            for item in admission["placements"]
        )
        for epoch, relative_profile in enumerate(manifest.profiles):
            profile_id = Path(relative_profile).stem
            if policy_name == BaselineId.B2_FRONTIER_FIXED_REALIZATION.value:
                candidates = alternatives_by_profile[profile_id].values()
                selected = next(
                    item
                    for item in candidates
                    if tuple(
                        (step.step_id, step.provider_id, step.node_id)
                        for step in item.step_placements
                    ) == admission_placement
                )
                row = {
                    "profile_id": profile_id,
                    "policy_id": policy_name,
                    **alternative_signature(selected),
                }
            else:
                row = next(item for item in policy_rows if item["profile_id"] == profile_id)
            transition_rows.append({**row, "resource_epoch": epoch})
    transition_validity = validate_transition_behavior(transition_rows)
    validity["transition_valid"] = transition_validity["valid"]
    validity["transition_failures"] = transition_validity["failures"]
    validity["valid"] = validity["valid"] and transition_validity["valid"]
    (output / "result.json").write_text(
        json.dumps(validity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "alternatives.json").write_text(
        json.dumps(alternatives_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "selected_plans.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "transition_plans.csv").open("w", newline="", encoding="utf-8") as handle:
        transition_fields = list(
            dict.fromkeys(key for row in transition_rows for key in row)
        )
        writer = csv.DictWriter(handle, fieldnames=transition_fields)
        writer.writeheader()
        writer.writerows(transition_rows)
    print(json.dumps(validity, indent=2, sort_keys=True))
    return 0 if validity["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
