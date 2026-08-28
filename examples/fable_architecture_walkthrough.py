"""Print one deterministic trip through FABLE's architecture without hardware."""

from datetime import timedelta

from fable.common.examples import BASE_TIME
from fable.common.time import EventTimeInterval
from fable.planning import (
    ArtifactCatalog,
    BoundedLabelPlanner,
    DemandCompileContext,
    DemandCompiler,
    PhysicalAlternativeGraphBuilder,
    PhysicalPlanner,
    default_predicate_registry,
)
from fable.planning.testing import (
    fake_artifact_catalog,
    fake_deployment,
    fake_follow_frontier,
    fake_provider_registry,
)
from fable.scheduling.adapters import candidate_from_search_result
from fable.scheduling.capacity import CapacityLedger
from fable.scheduling.lifecycle import ProviderLifecycleManager
from fable.scheduling.models import TaskSchedulingPolicy
from fable.semantic.models import ScriptedResultSpec
from fable.semantic.testing import predicate_result_from_spec


def show(number, label, value, module):
    print(f"{number}. {label}: {type(value).__name__} [{module}]")
    print(f"   {value}")


def main() -> None:
    deployment = fake_deployment()
    providers = fake_provider_registry()
    artifacts: ArtifactCatalog = fake_artifact_catalog()
    runtime, hypothesis, frontier = fake_follow_frontier()

    show(1, "selected SemanticGraph", runtime.graph.graph, "fable.contracts.semantic")
    show(2, "initial Hypothesis", hypothesis, "fable.contracts.hypothesis")
    show(3, "Active Frontier", frontier, "fable.semantic.models")

    compiler = DemandCompiler(
        predicate_registry=default_predicate_registry(), deployment=deployment
    )
    node_id = runtime.graph.nodes_by_key["follower_follows"].node_id
    demands = compiler.compile(
        graph=runtime.graph,
        hypothesis=hypothesis,
        frontier=frontier,
        context=DemandCompileContext(
            eligible_source_ids_by_node={
                node_id: ("camera_mobile", "camera_downstream")
            }
        ),
    )
    show(4, "PredicateDemand", demands[0], "fable.planning.demand_compiler")

    generator = PhysicalAlternativeGraphBuilder(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
    )
    search = BoundedLabelPlanner(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
    )
    planned = PhysicalPlanner(
        alternative_generator=generator, plan_search=search
    ).plan(demands, now=BASE_TIME + timedelta(seconds=2))
    show(5, "physical alternatives", len(planned.alternatives.alternatives), "fable.planning.alternative_graph")
    show(6, "selected ExecutionPlan", planned.execution_plan, "fable.planning.planner")

    candidate = candidate_from_search_result(
        planned.search,
        planned.alternatives,
        demands,
        task_policy=TaskSchedulingPolicy(request_id=hypothesis.request_id),
    )
    lifecycle = ProviderLifecycleManager(
        provider_registry=providers, capacity=CapacityLedger(deployment)
    )
    admitted = lifecycle.attach_candidate(candidate, now=BASE_TIME + timedelta(seconds=2))
    show(7, "PlanCandidate/admission", candidate, "fable.scheduling.adapters/admission")
    lease = lifecycle.leases[admitted.lease_ids[0]].lease
    show(8, "ProviderLease", lease, "fable.contracts.scheduling")

    result = predicate_result_from_spec(
        runtime,
        hypothesis.hypothesis_id,
        ScriptedResultSpec(
            node_key="follower_follows",
            source_id="camera_mobile",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME + timedelta(seconds=2),
                end=BASE_TIME + timedelta(seconds=3),
            ),
            introduced={"follower": "track_18"},
        ),
    )
    show(9, "PredicateResult", result, "fable.contracts.result")
    transition = runtime.apply(result)
    updated = runtime.get_hypothesis(transition.hypothesis_ids[0])
    show(10, "updated Hypothesis/Frontier", (updated, transition.frontiers), "fable.semantic.runtime")
    if updated.lifecycle.value == "COMPLETED":
        show(11, "TerminalComplexEvent", updated, "fable.contracts.result")
    else:
        print("11. TerminalComplexEvent: not yet complete; the printed frontier shows the next evidence need")


if __name__ == "__main__":
    main()
