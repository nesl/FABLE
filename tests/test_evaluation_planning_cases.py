import csv
from pathlib import Path

import pytest

from evaluation.baselines.factory import build_baseline_policy
from evaluation.baselines.models import TaskResourcePlanningCase
from evaluation.planning_cases import (
    VARIANT_TEMPLATES,
    compile_evaluation_planning_case,
    executable_runtime_graph,
    scope_demands_to_nodes,
)
from evaluation.schemas import BaselineId
from fable.common.examples import BASE_TIME
from fable.distributed.config import ProviderRuntimeResolver
from fable.distributed.models import ProviderRuntimeSpec, RuntimeMode
from fable.planning import BoundedLabelPlanner
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import ComputeCapacity
from fable.planning.artifact_catalog import ArtifactCatalog
from fable.planning.testing import (
    fake_artifact_catalog,
    fake_deployment,
    fake_provider_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def test_bounded_seed_admission_covers_repeated_visits_and_convergence() -> None:
    assert VARIANT_TEMPLATES["Two-visit stalking"].seed_admission_strategy == "reference_bounded"
    assert VARIANT_TEMPLATES["Three-visit stalking"].seed_admission_strategy == "reference_bounded"
    assert VARIANT_TEMPLATES["Two-visit stalking"].max_seed_hypotheses == 20
    assert VARIANT_TEMPLATES["Three-visit stalking"].max_seed_hypotheses == 20
    assert VARIANT_TEMPLATES["Vehicle convergence"].seed_admission_strategy == "reference_bounded"
    assert VARIANT_TEMPLATES["Vehicle convergence"].max_seed_hypotheses == 20
    assert all(
        template.seed_admission_strategy == "first_distinct"
        for variant, template in VARIANT_TEMPLATES.items()
        if variant
        not in {"Two-visit stalking", "Three-visit stalking", "Vehicle convergence"}
    )


def test_live_node_scope_is_applied_before_alternative_enumeration() -> None:
    case = compile_evaluation_planning_case(
        variant="Pass-follow-clear convoy",
        run_id="run",
        trace_id="trace",
        request_id="request",
        now=BASE_TIME,
        provider_registry=fake_provider_registry(),
        artifact_catalog=fake_artifact_catalog(),
        deployment=fake_deployment(),
    )
    scoped = scope_demands_to_nodes(
        case.frontier_demands,
        ("sensor_a", "server_1"),
    )

    assert scoped
    assert all(
        demand.hard_constraints.allowed_node_ids
        == ("sensor_a", "server_1")
        for demand in scoped
    )
    assert all(demand.sharing_key is None for demand in scoped)


def _variants():
    with (ROOT / "evaluation/labels/filtered_complex_event_experiments.csv").open() as handle:
        return sorted(
            {
                row["ce_variant"]
                for row in csv.DictReader(handle)
                if row["recommended_for_use"].lower() == "true"
            }
        )


@pytest.mark.parametrize("variant", _variants())
def test_every_recommended_variant_has_a_family_specific_physical_case(variant) -> None:
    case = compile_evaluation_planning_case(
        variant=variant,
        run_id="run",
        trace_id="trace",
        request_id=f"request-{variant}",
        now=BASE_TIME,
        provider_registry=fake_provider_registry(),
        artifact_catalog=fake_artifact_catalog(),
        deployment=fake_deployment(),
    )
    assert case.frontier_demands
    assert case.all_task_demands
    assert case.frontier_graph.alternatives
    assert case.whole_event_graph.alternatives
    assert {
        demand.demand_id for demand in case.all_task_demands
    } == {
        alternative.demand_id
        for alternative in case.whole_event_graph.alternatives
    }


@pytest.mark.parametrize(
    "variant,expected",
    [
        (
            "Cross-sensor robbery",
            {"VEHICLE_PRESENT_BEFORE", "EXITS", "SAME_ENTITY"},
        ),
        ("Pass-follow-clear convoy", {"PASSES"}),
        (
            "Robbery with alarm",
            {"EXITS"},
        ),
        (
            "Talking/rendezvous",
            {"CONVERSATION", "PERSON_PROXIMITY", "EXITS"},
        ),
        ("Vehicle rendezvous", {"DISTANCE_LT", "EXITS"}),
        ("Three-visit stalking", {"PASSES", "SAME_ENTITY"}),
        ("Two-vehicle chase", {"PASSES"}),
        ("Vehicle convergence", {"DISTANCE_LT", "EXITS"}),
    ],
)
def test_variants_use_their_authored_predicates(variant, expected) -> None:
    case = compile_evaluation_planning_case(
        variant=variant,
        run_id="run",
        trace_id="trace",
        request_id=f"request-{variant}",
        now=BASE_TIME,
        provider_registry=fake_provider_registry(),
        artifact_catalog=fake_artifact_catalog(),
        deployment=fake_deployment(),
    )
    assert {
        item.semantic_predicate.predicate_id for item in case.all_task_demands
    } == expected


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize(
    "baseline_id",
    [
        BaselineId.B0_ALWAYS_ON,
        BaselineId.B1_HANDWRITTEN_STATIC,
        BaselineId.B2_STATIC_WHOLE_EVENT,
        BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
        BaselineId.B4_GREEDY_FRONTIER,
        BaselineId.FABLE,
    ],
)
def test_every_live_baseline_finds_a_family_specific_plan(variant, baseline_id) -> None:
    providers = fake_provider_registry()
    artifacts = fake_artifact_catalog()
    base_deployment = fake_deployment()
    # This test checks policy completeness, not contention.  Give every fake
    # node enough capacity for the complete-task baselines; capacity rejection
    # is covered independently by the beam-search tests.
    deployment = DeploymentGraph(
        nodes=tuple(
            node.model_copy(
                update={
                    "capacity": ComputeCapacity(
                        cpu_cores=100,
                        memory_mb=100_000,
                        gpu_memory_mb=100_000,
                    )
                }
            )
            for node in base_deployment.nodes.values()
        ),
        sources=tuple(base_deployment.sources.values()),
        links=base_deployment.links,
    )
    case = compile_evaluation_planning_case(
        variant=variant,
        run_id="run",
        trace_id="trace",
        request_id=f"request-{variant}",
        now=BASE_TIME,
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
    )
    policy = build_baseline_policy(
        baseline_id,
        planner=BoundedLabelPlanner(
            provider_registry=providers,
            artifact_catalog=artifacts,
            deployment=deployment,
        ),
        static_registry_path=(
            ROOT / "evaluation/manifests/baselines/static_pipelines.yaml"
        ),
    )
    policy_case = (
        TaskResourcePlanningCase.from_case(case)
        if baseline_id == BaselineId.B3_TASK_RESOURCE_ADAPTIVE
        else case
    )
    assert policy.plan(policy_case).selected_alternative_ids


def test_live_runtime_filter_excludes_reference_placements() -> None:
    case = compile_evaluation_planning_case(
        variant="Vehicle convergence",
        run_id="run",
        trace_id="trace",
        request_id="request",
        now=BASE_TIME,
        provider_registry=fake_provider_registry(),
        artifact_catalog=fake_artifact_catalog(),
        deployment=fake_deployment(),
    )
    placement_keys = {
        (step.node_id, step.provider_id)
        for alternative in case.frontier_graph.alternatives
        for step in alternative.step_placements
    }
    resolver = ProviderRuntimeResolver(
        {
            key: ProviderRuntimeSpec(
                provider_id=key[1],
                provider_contract_version=1,
                node_id=key[0],
                mode=RuntimeMode.REFERENCE,
            )
            for key in placement_keys
        }
    )
    assert not executable_runtime_graph(
        case.frontier_graph, runtime_resolver=resolver
    ).alternatives
    assert executable_runtime_graph(
        case.frontier_graph,
        runtime_resolver=resolver,
        allow_reference_runtimes=True,
    ).alternatives
    selected_nodes = tuple(
        sorted(
            {
                step.node_id
                for step in case.frontier_graph.alternatives[0].step_placements
            }
        )
    )
    scoped = executable_runtime_graph(
        case.frontier_graph,
        runtime_resolver=resolver,
        allow_reference_runtimes=True,
        allowed_node_ids=selected_nodes,
    )
    assert scoped.alternatives
    assert all(
        step.node_id in selected_nodes
        for alternative in scoped.alternatives
        for step in alternative.step_placements
    )
    assert not executable_runtime_graph(
        case.frontier_graph,
        runtime_resolver=resolver,
        allow_reference_runtimes=True,
        allowed_node_ids=("not-a-deployed-node",),
    ).alternatives


def test_vehicle_rendezvous_has_uncalibrated_distance_and_exit_plans() -> None:
    """The 2025 mobile replay has video but no surveyed camera artifacts."""

    case = compile_evaluation_planning_case(
        variant="Vehicle rendezvous",
        run_id="run",
        trace_id="uncalibrated-mobile",
        request_id="request-uncalibrated-mobile",
        now=BASE_TIME,
        provider_registry=fake_provider_registry(),
        artifact_catalog=ArtifactCatalog(),
        deployment=fake_deployment(),
    )

    alternatives_by_predicate = {
        demand.semantic_predicate.predicate_id: tuple(
            alternative
            for alternative in case.whole_event_graph.alternatives
            if alternative.demand_id == demand.demand_id
        )
        for demand in case.all_task_demands
    }
    assert alternatives_by_predicate["DISTANCE_LT"]
    assert all(
        not any(
            item.data_type in {"camera_calibration.v1", "route_graph.v1"}
            for item in alternative.external_inputs
        )
        for alternative in alternatives_by_predicate["DISTANCE_LT"]
    )
    exit_demands = tuple(
        demand
        for demand in case.all_task_demands
        if demand.semantic_predicate.predicate_id == "EXITS"
    )
    assert len(exit_demands) == 2
    assert all(
        any(
            alternative.demand_id == demand.demand_id
            and any(
                step.provider_id == "track_lifecycle_exit_evaluator"
                for step in alternative.step_placements
            )
            for alternative in case.whole_event_graph.alternatives
        )
        for demand in exit_demands
    )
