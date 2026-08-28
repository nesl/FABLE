from evaluation.task_universe import TaskDemandUniverseBuilder
from fable.common.examples import BASE_TIME
from fable.planning import (
    DemandCompileContext,
    DemandCompiler,
    default_predicate_registry,
)
from fable.planning.testing import fake_deployment
from fable.semantic import (
    EventRequestCompiler,
    SemanticRuntime,
    SemanticRuntimeConfig,
    seed_result_from_spec,
)
from fable.semantic.models import ScriptedResultSpec
from fable.common.time import EventTimeInterval


def test_task_universe_covers_every_remaining_rendezvous_branch() -> None:
    graph = EventRequestCompiler().compile(
        {"family_id": "rendezvous", "parameters": {"interaction": "either"}}
    ).graph
    runtime = SemanticRuntime(
        graph,
        config=SemanticRuntimeConfig(request_id="universe-rendezvous"),
    )
    seed = seed_result_from_spec(
        runtime,
        ScriptedResultSpec(
            node_key="arrival",
            source_id="camera_a",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME,
            ),
            introduced={"arrival_vehicle": "vehicle-1"},
        ),
    )
    transition = runtime.seed(seed)
    hypothesis = runtime.get_hypothesis(transition.hypothesis_ids[0])
    compiler = DemandCompiler(
        predicate_registry=default_predicate_registry(),
        deployment=fake_deployment(),
    )
    universe = TaskDemandUniverseBuilder(compiler).build(
        graph=runtime.graph,
        hypothesis=hypothesis,
        context=DemandCompileContext(
            eligible_source_ids_by_node={
                node_id: tuple(sorted(fake_deployment().sources))
                for node_id in runtime.graph.executable_predicate_nodes()
            }
        ),
    )
    by_key = {
        runtime.graph.nodes_by_id[item.graph_node_id].authored_key
        for item in universe
    }
    assert {"conversation", "visual_proximity", "arrival_vehicle_exits"} <= by_key
    assert "arrival" not in by_key


def test_retrospective_universe_carries_valid_anchor_context() -> None:
    graph = EventRequestCompiler().compile(
        {"family_id": "robbery", "parameters": {}}
    ).graph
    runtime = SemanticRuntime(
        graph,
        config=SemanticRuntimeConfig(request_id="universe-robbery"),
    )
    seed = seed_result_from_spec(
        runtime,
        ScriptedResultSpec(
            node_key="gunshot_branch",
            source_id="microphone_a",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME,
            ),
            introduced={"location": "site"},
        ),
    )
    transition = runtime.seed(seed)
    hypothesis = runtime.get_hypothesis(transition.hypothesis_ids[0])
    compiler = DemandCompiler(
        predicate_registry=default_predicate_registry(),
        deployment=fake_deployment(),
    )
    universe = TaskDemandUniverseBuilder(compiler).build(
        graph=runtime.graph,
        hypothesis=hypothesis,
    )
    retrospective = [
        item for item in universe if item.retrospective_context is not None
    ]
    assert retrospective
    assert all(
        item.retrospective_context.get("structural_template")
        or item.retrospective_context.get("anchor_event_time")
        for item in retrospective
    )


def test_structural_universe_represents_consumer_only_roles_symbolically() -> None:
    graph = EventRequestCompiler().compile(
        {"family_id": "robbery", "parameters": {}}
    ).graph
    runtime = SemanticRuntime(
        graph,
        config=SemanticRuntimeConfig(request_id="universe-unseeded-robbery"),
    )
    transition = runtime.start(observed_at=BASE_TIME)
    hypothesis = runtime.get_hypothesis(transition.hypothesis_ids[0])
    compiler = DemandCompiler(
        predicate_registry=default_predicate_registry(),
        deployment=fake_deployment(),
    )

    universe = TaskDemandUniverseBuilder(compiler).build(
        graph=runtime.graph,
        hypothesis=hypothesis,
        skip_uncompilable=False,
    )

    assert universe
    symbolic = {
        demand.semantic_predicate.predicate_id: tuple(demand.bound_roles.values())
        for demand in universe
        if any(
            value.startswith("__structural_unbound__:")
            for value in demand.bound_roles.values()
        )
    }
    assert "SAME_ENTITY" in symbolic
