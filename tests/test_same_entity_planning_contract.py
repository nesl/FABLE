from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from evaluation.planning_cases import executable_runtime_graph
from fable.common.enums import BindingCapability, ResultKind
from fable.common.ids import uuid7
from fable.common.schemas import (
    ContinuationRequirement,
    DataMovementConstraints,
    DemandBindingPolicy,
    PredicateDemand,
    PredicateRole,
    SemanticPredicate,
)
from fable.common.time import DeadlineSpec, EventTimeInterval
from fable.distributed.config import ProviderRuntimeResolver, load_deployment_graph
from fable.planning.alternative_graph import (
    AlternativeBuildConfig,
    PhysicalAlternativeGraphBuilder,
)
from fable.planning.artifact_catalog import ArtifactCatalog
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import NetworkLink
from fable.planning.provider_registry import ProviderRegistry


ROOT = Path(__file__).resolve().parents[1]


def test_same_entity_has_executable_reid_plan_with_required_continuation() -> None:
    registry = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
    )
    deployment = load_deployment_graph(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    runtimes = ProviderRuntimeResolver.from_yaml(
        ROOT / "iobt-minimal-ce-replay/config/fable_provider_runtimes.yaml"
    )
    # The checked-in deployment is intentionally a one-camera smoke fixture.
    # Clone its sensor contract to exercise a genuine two-camera identity plan
    # without depending on generated run artifacts.
    sensor = deployment.node("dvpg_gq_orin_11")
    camera = deployment.source("orin11_camera")
    second_node_id = "dvpg_gq_orin_13"
    deployment = DeploymentGraph(
        nodes=(*deployment.nodes.values(), sensor.model_copy(update={"node_id": second_node_id})),
        sources=(
            *deployment.sources.values(),
            camera.model_copy(
                update={"source_id": "orin13_camera", "node_id": second_node_id}
            ),
        ),
        links=(
            *deployment.links,
            NetworkLink(
                source_node_id=second_node_id,
                target_node_id="x86server",
                latency_ms=6,
                bandwidth_mbps=100,
                bidirectional=True,
            ),
        ),
    )
    runtime_map = {(item.node_id, item.provider_id): item for item in runtimes.runtimes}
    for item in runtimes.runtimes:
        if item.node_id == "dvpg_gq_orin_11":
            clone = item.model_copy(update={"node_id": second_node_id})
            runtime_map[(second_node_id, clone.provider_id)] = clone
    runtimes = ProviderRuntimeResolver(runtime_map)
    now = datetime(2026, 8, 2, tzinfo=UTC)
    demand = PredicateDemand(
        request_id="same-entity-contract-test",
        graph_hash="same-entity-contract-test",
        hypothesis_id=uuid7(),
        hypothesis_version=1,
        frontier_id=uuid7(),
        checkpoint_id=uuid7(),
        graph_node_id="same_vehicle",
        semantic_predicate=SemanticPredicate(
            predicate_id="SAME_ENTITY",
            roles=(
                PredicateRole(role_name="left", variable="vehicle", entity_type="vehicle"),
                PredicateRole(
                    role_name="right",
                    variable="departing_vehicle",
                    entity_type="vehicle",
                ),
            ),
            result_kind=ResultKind.STATE_OBSERVATION,
        ),
        bound_roles={
            "left": "dvpg_gq_orin_11:historical-track",
            "right": "dvpg_gq_orin_13:departing-track",
        },
        event_time_interval=EventTimeInterval(
            start=now - timedelta(seconds=180), end=now
        ),
        deadline=DeadlineSpec(latest_useful_completion=now + timedelta(seconds=180)),
        eligible_source_ids=tuple(deployment.sources),
        required_capabilities=("vehicle_identity", "cross_sensor_identity"),
        acceptable_output_types=("canonical_entity_map.v1",),
        hard_constraints=DataMovementConstraints(
            raw_data_must_remain_local=True,
            allowed_node_ids=tuple(deployment.nodes),
        ),
        continuation_requirements=(
            ContinuationRequirement(
                artifact_type="canonical_entity_map.v1",
                required_until=now + timedelta(seconds=180),
                required_bindings=("left", "right"),
            ),
        ),
        desired_continuation_types=("canonical_entity_map.v1",),
        binding_policy=DemandBindingPolicy(
            role_modes={
                "left": BindingCapability.CONSUME,
                "right": BindingCapability.CONSUME,
            }
        ),
    )
    graph = PhysicalAlternativeGraphBuilder(
        provider_registry=registry,
        artifact_catalog=ArtifactCatalog(),
        deployment=deployment,
        config=AlternativeBuildConfig(
            max_candidate_nodes_per_step=16,
            max_total_alternatives=512,
            max_alternatives_per_chain=256,
        ),
    ).build((demand,), now=now)
    plans = [
        alternative
        for alternative in graph.alternatives
        if alternative.chain_id == "same_entity_cross_camera_reid"
    ]
    assert plans, [(item.code, item.reason) for item in graph.pruned]
    provider_ids = {
        placement.provider_id
        for plan in plans
        for placement in plan.step_placements
    }
    assert {
        "track_crop_extractor",
        "vehicle_reid_descriptor",
        "cross_sensor_identity_association",
    }.issubset(provider_ids)
    for plan in plans:
        inputs = {item.input_name: item for item in plan.external_inputs}
        assert inputs["video_a"].source_id == "orin11_camera"
        assert inputs["video_b"].source_id == "orin13_camera"

def test_same_entity_can_repair_two_fragmented_tracks_from_one_camera() -> None:
    registry = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
    )
    deployment = load_deployment_graph(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    now = datetime(2026, 8, 4, tzinfo=UTC)
    demand = PredicateDemand(
        request_id="same-camera-fragment-contract-test",
        graph_hash="same-camera-fragment-contract-test",
        hypothesis_id=uuid7(),
        hypothesis_version=1,
        frontier_id=uuid7(),
        checkpoint_id=uuid7(),
        graph_node_id="same_vehicle",
        semantic_predicate=SemanticPredicate(
            predicate_id="SAME_ENTITY",
            roles=(
                PredicateRole(role_name="left", variable="vehicle", entity_type="vehicle"),
                PredicateRole(role_name="right", variable="returning_vehicle", entity_type="vehicle"),
            ),
            result_kind=ResultKind.STATE_OBSERVATION,
        ),
        bound_roles={
            "left": "dvpg_gq_orin_11:stable-session:0",
            "right": "dvpg_gq_orin_11:stable-session:4",
        },
        event_time_interval=EventTimeInterval(start=now - timedelta(seconds=180), end=now),
        deadline=DeadlineSpec(latest_useful_completion=now + timedelta(seconds=180)),
        eligible_source_ids=tuple(deployment.sources),
        required_capabilities=("vehicle_identity", "cross_sensor_identity"),
        acceptable_output_types=("canonical_entity_map.v1",),
        hard_constraints=DataMovementConstraints(
            raw_data_must_remain_local=True,
            allowed_node_ids=tuple(deployment.nodes),
        ),
        binding_policy=DemandBindingPolicy(
            role_modes={
                "left": BindingCapability.CONSUME,
                "right": BindingCapability.CONSUME,
            }
        ),
    )
    graph = PhysicalAlternativeGraphBuilder(
        provider_registry=registry,
        artifact_catalog=ArtifactCatalog(),
        deployment=deployment,
        config=AlternativeBuildConfig(max_candidate_nodes_per_step=16),
    ).build((demand,), now=now)

    plans = [
        alternative
        for alternative in graph.alternatives
        if alternative.chain_id == "same_entity_same_camera_reid"
    ]
    assert plans
    for plan in plans:
        inputs = {item.input_name: item for item in plan.external_inputs}
        assert set(inputs) == {"video"}
        assert inputs["video"].node_id == "dvpg_gq_orin_11"
        assert [
            item.provider_id for item in plan.step_placements
        ].count("yolo_vehicle_fast_640") == 1


def test_runtime_eligibility_is_applied_before_identity_placement_budget() -> None:
    registry = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
    )
    deployment = load_deployment_graph(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    now = datetime(2026, 8, 4, tzinfo=UTC)
    demand = PredicateDemand(
        request_id="bounded-runtime-placement-test",
        graph_hash="bounded-runtime-placement-test",
        hypothesis_id=uuid7(),
        hypothesis_version=1,
        frontier_id=uuid7(),
        checkpoint_id=uuid7(),
        graph_node_id="same_vehicle",
        semantic_predicate=SemanticPredicate(
            predicate_id="SAME_ENTITY",
            roles=(
                PredicateRole(role_name="left", variable="vehicle", entity_type="vehicle"),
                PredicateRole(role_name="right", variable="returning_vehicle", entity_type="vehicle"),
            ),
            result_kind=ResultKind.STATE_OBSERVATION,
        ),
        bound_roles={
            "left": "dvpg_gq_orin_11:session:1",
            "right": "dvpg_gq_orin_11:session:4",
        },
        event_time_interval=EventTimeInterval(start=now - timedelta(seconds=180), end=now),
        deadline=DeadlineSpec(latest_useful_completion=now + timedelta(seconds=180)),
        eligible_source_ids=("orin11_camera",),
        required_capabilities=("vehicle_identity", "cross_sensor_identity"),
        acceptable_output_types=("canonical_entity_map.v1",),
        hard_constraints=DataMovementConstraints(
            raw_data_must_remain_local=True,
            allowed_node_ids=tuple(deployment.nodes),
        ),
        binding_policy=DemandBindingPolicy(
            role_modes={"left": BindingCapability.CONSUME, "right": BindingCapability.CONSUME}
        ),
    )
    sensor_providers = {
        "yolo_vehicle_fast_640",
        "multi_object_tracker",
        "track_crop_extractor",
    }
    site_providers = {
        "vehicle_reid_descriptor",
        "cross_sensor_identity_association",
    }

    def executable(node_id: str, provider_id: str) -> bool:
        return (
            node_id == "dvpg_gq_orin_11" and provider_id in sensor_providers
        ) or (node_id == "x86server" and provider_id in site_providers)

    graph = PhysicalAlternativeGraphBuilder(
        provider_registry=registry,
        artifact_catalog=ArtifactCatalog(),
        deployment=deployment,
        config=AlternativeBuildConfig(max_placement_variants_per_assignment=1),
        placement_eligible=executable,
    ).build((demand,), now=now)
    assert {item.chain_id for item in graph.alternatives} == {
        "same_entity_cross_camera_reid",
        "same_entity_same_camera_reid",
    }
    assert all(
        executable(step.node_id, step.provider_id)
        for alternative in graph.alternatives
        for step in alternative.step_placements
    )


def test_cross_camera_suffix_steps_stay_with_their_sensor_pipeline() -> None:
    registry = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
    )
    bundle = ROOT / "runs/evaluation_bundles/full-suite-20260414_150705"
    deployment = load_deployment_graph(bundle / "fable_deployment.yaml")
    runtimes = ProviderRuntimeResolver.from_yaml(bundle / "fable_provider_runtimes.yaml")
    now = datetime(2026, 4, 14, 20, 10, 46, tzinfo=UTC)
    demand = PredicateDemand(
        request_id="orin11-same-entity-locality",
        graph_hash="orin11-same-entity-locality",
        hypothesis_id=uuid7(), hypothesis_version=1,
        frontier_id=uuid7(), checkpoint_id=uuid7(), graph_node_id="same",
        semantic_predicate=SemanticPredicate(
            predicate_id="SAME_ENTITY",
            roles=(
                PredicateRole(role_name="left", variable="vehicle", entity_type="vehicle"),
                PredicateRole(role_name="right", variable="returning", entity_type="vehicle"),
            ),
            result_kind=ResultKind.STATE_OBSERVATION,
        ),
        bound_roles={
            "left": "dvpg_gq_orin_11:session:0",
            "right": "dvpg_gq_orin_11:session:2",
        },
        event_time_interval=EventTimeInterval(start=now - timedelta(seconds=90), end=now),
        deadline=DeadlineSpec(latest_useful_completion=now + timedelta(minutes=5)),
        eligible_source_ids=("orin11_camera",),
        required_capabilities=("vehicle_identity", "cross_sensor_identity"),
        acceptable_output_types=("canonical_entity_map.v1",),
        hard_constraints=DataMovementConstraints(
            raw_data_must_remain_local=True,
            allowed_node_ids=tuple(deployment.nodes),
        ),
        binding_policy=DemandBindingPolicy(role_modes={
            "left": BindingCapability.CONSUME,
            "right": BindingCapability.CONSUME,
        }),
    )
    graph = PhysicalAlternativeGraphBuilder(
        provider_registry=registry,
        artifact_catalog=ArtifactCatalog(),
        deployment=deployment,
        config=AlternativeBuildConfig(max_candidate_nodes_per_step=16),
        placement_eligible=runtimes.has,
    ).build((demand,), now=now)

    cross = [item for item in graph.alternatives if item.chain_id == "same_entity_cross_camera_reid"]
    assert cross
    for alternative in cross:
        placements = {step.step_id: step.node_id for step in alternative.step_placements}
        assert placements["track_a"] == placements["crops_a"] == "dvpg_gq_orin_11"
        assert placements["track_b"] == placements["crops_b"] == "dvpg_gq_orin_11"
