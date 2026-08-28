from __future__ import annotations

from pathlib import Path
import ast

from fable.common.enums import ResultKind
from fable.common.schemas import PredicateDemand, SemanticPredicate
from fable.planning import (
    BoundedLabelPlanner,
    PhysicalAlternativeGraphBuilder,
    PhysicalPlanner,
)
from fable.planning.provider_registry import ProviderRegistry
from fable.planning.testing import (
    fake_artifact_catalog,
    fake_deployment,
    fake_follow_demand,
    fake_follow_frontier,
    fake_provider_registry,
)
from fable.semantic import ComplexEvent, SemanticRuntime, SemanticRuntimeConfig
from fable.semantic.definitions import package_exchange_graph
from fable.semantic.definitions.example_event import authored_api_example_graph
from fable.semantic.definitions.multimodal import package_exchange_graph as compatibility_package_exchange
from fable.semantic.definitions.registry import PRODUCTION_DEFINITIONS


ROOT = Path(__file__).resolve().parents[1]


def test_readable_authoring_infers_roles_and_compiles_dag() -> None:
    event = ComplexEvent("readable_test")
    event.role("vehicle_a", "vehicle")
    event.role("vehicle_b", "vehicle")
    first = event.predicate("ENTERS", bind={"vehicle": "vehicle_a"})
    second = event.predicate("ENTERS", bind={"vehicle": "vehicle_b"})
    graph = event.all_of(first, second).build()

    assert {role.role_name for role in graph.roles} == {"vehicle_a", "vehicle_b"}
    predicates = [node for node in graph.nodes if node.predicate is not None]
    assert len(predicates) == 2
    assert {item.predicate.roles[0].entity_type for item in predicates} == {"vehicle"}


def test_authored_example_can_narrow_generic_entity_role() -> None:
    graph = authored_api_example_graph()
    assert {role.role_name: role.entity_type for role in graph.roles} == {
        "destination_holder": "entity",
        "reference": "location",
    }
    assert {
        node.predicate.predicate_id
        for node in graph.nodes
        if node.predicate is not None
    } == {"PASSES", "EXITS"}


def test_one_frontier_can_expose_two_static_predicate_nodes() -> None:
    event = ComplexEvent("frontier_test")
    event.role("vehicle_a", "vehicle")
    event.role("vehicle_b", "vehicle")
    root = event.all_of(
        event.predicate("ENTERS", bind={"vehicle": "vehicle_a"}),
        event.predicate("ENTERS", bind={"vehicle": "vehicle_b"}),
    )
    runtime = SemanticRuntime(
        root.build(), config=SemanticRuntimeConfig(request_id="frontier-test")
    )
    transition = runtime.start()
    frontier = transition.frontiers[0]
    assert len(frontier.active_predicate_node_ids) == 2
    assert len(frontier.active_predicates(runtime.graph)) == 2


def test_production_registry_has_one_canonical_module_per_family() -> None:
    assert len({item.family_id for item in PRODUCTION_DEFINITIONS}) == len(
        PRODUCTION_DEFINITIONS
    )
    assert all(item.module.startswith("fable.semantic.definitions.") for item in PRODUCTION_DEFINITIONS)
    assert all(item.module.rsplit(".", 1)[-1] not in {"vehicle", "multimodal"} for item in PRODUCTION_DEFINITIONS)
    assert all(item.factory.__module__ == item.module for item in PRODUCTION_DEFINITIONS)
    assert compatibility_package_exchange().graph_hash == package_exchange_graph().graph_hash

    for compatibility_module in ("vehicle.py", "multimodal.py"):
        tree = ast.parse(
            (ROOT / "fable/semantic/definitions" / compatibility_module).read_text()
        )
        assert not any(isinstance(item, ast.FunctionDef) for item in tree.body)


def test_physical_planner_facade_matches_existing_build_then_search() -> None:
    demand = fake_follow_demand()
    demands = (demand,)
    providers = fake_provider_registry()
    artifacts = fake_artifact_catalog()
    deployment = fake_deployment()
    generator = PhysicalAlternativeGraphBuilder(
        provider_registry=providers, artifact_catalog=artifacts, deployment=deployment
    )
    search = BoundedLabelPlanner(
        provider_registry=providers, artifact_catalog=artifacts, deployment=deployment
    )
    now = demand.event_time_interval.end
    expected_graph = generator.build(demands, now=now)
    expected = search.search(expected_graph, demands, now=now)
    actual = PhysicalPlanner(
        alternative_generator=generator, plan_search=search
    ).plan(demands, now=now)
    assert actual.search.selected == expected.selected
    assert actual.execution_plan is not None and expected.execution_plan is not None
    assert actual.execution_plan.label_id == expected.execution_plan.label_id
    assert actual.execution_plan.steps == expected.execution_plan.steps


def test_fake_provider_onboarding_catalog_validates_and_is_discoverable() -> None:
    registry = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/examples/fake_object_detector.catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
        profiles_path=ROOT / "providers/registry/examples/fake_object_detector.profile.json",
    )
    base = fake_follow_demand()
    payload = base.model_dump(mode="python", exclude={"sharing_key"})
    payload.update(
        semantic_predicate=SemanticPredicate(
            predicate_id="OBJECT_PRESENT", roles=(), result_kind=ResultKind.INSTANT_MATCH
        ),
        bound_roles={},
        unbound_roles=(),
        acceptable_output_types=("predicate_match.v1",),
        binding_policy={},
    )
    demand = PredicateDemand.model_validate(payload)
    assert [item.chain_id for item in registry.candidate_chains(demand)] == [
        "fake_object_present"
    ]


def test_demand_compiler_can_compile_explicit_sources_without_deployment() -> None:
    from fable.planning import DemandCompileContext, DemandCompiler, default_predicate_registry

    runtime, hypothesis, frontier = fake_follow_frontier()
    node_id = runtime.graph.nodes_by_key["follower_follows"].node_id
    demands = DemandCompiler(
        predicate_registry=default_predicate_registry()
    ).compile(
        graph=runtime.graph,
        hypothesis=hypothesis,
        frontier=frontier,
        context=DemandCompileContext(
            eligible_source_ids_by_node={node_id: ("camera_mobile",)}
        ),
    )
    assert demands[0].eligible_source_ids == ("camera_mobile",)
