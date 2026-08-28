from evaluation.baselines import (
    AlwaysOnPolicy,
    BaselinePlanningCase,
    FablePolicy,
    ProduceAllPolicy,
)
from evaluation.live_orchestration import LivePlanningBridge
from evaluation.orchestration import ControlledPlanningCoordinator, PlanningTrigger
from evaluation.schemas import BaselineId
from fable.common.examples import BASE_TIME
from fable.planning import BoundedLabelPlanner
from fable.planning.models import ExternalInputKind
from fable.planning.phase4_testing import fake_follow_alternative_graph
from fable.planning.testing import fake_artifact_catalog, fake_deployment, fake_provider_registry
from fable.scheduling.models import TaskSchedulingPolicy


class RecordingDispatcher:
    def __init__(self) -> None:
        self.submissions = []
        self.cancelled_leases = []
        self.request_sweeps = []

    def submit_candidates(
        self,
        candidates,
        *,
        runtime_overrides=None,
        now=None,
        allow_capacity_overcommit=False,
    ):
        self.submissions.append(tuple(candidates))
        return {"count": len(candidates)}, tuple(
            f"command:{candidate.candidate_id}" for candidate in candidates
        )

    def cancel_leases(self, leases, *, reason):
        self.cancelled_leases.extend(leases)
        return tuple(f"cancel:{item.lease.lease_id}" for item in leases)

    def sweep_request(self, node_ids, *, request_id, reason):
        commands = tuple(f"sweep:{node_id}:{request_id}" for node_id in node_ids)
        self.request_sweeps.extend(commands)
        return commands


def _fixture():
    providers = fake_provider_registry()
    artifacts = fake_artifact_catalog()
    deployment = fake_deployment()
    graph, demand = fake_follow_alternative_graph(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
    )
    case = BaselinePlanningCase(
        run_id="run",
        trace_id="trace",
        request_id=demand.request_id,
        event_family="route_convoy",
        frontier_demands=(demand,),
        all_task_demands=(demand,),
        frontier_graph=graph,
        whole_event_graph=graph,
        now=BASE_TIME,
    )
    planner = BoundedLabelPlanner(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
    )
    return providers, planner, case


def test_fable_decision_becomes_one_normal_dispatch_candidate() -> None:
    providers, planner, case = _fixture()
    dispatcher = RecordingDispatcher()
    bridge = LivePlanningBridge(
        coordinator=ControlledPlanningCoordinator(FablePolicy(planner)),
        provider_registry=providers,
        dispatcher=dispatcher,
    )
    result = bridge.plan_and_dispatch(
        case,
        trigger=PlanningTrigger.ADMISSION,
        task_policy=TaskSchedulingPolicy(request_id=case.request_id),
    )
    assert result.decision.pruning_counts
    assert result.decision.pruning_samples
    assert len(result.candidates) == 1
    assert len(dispatcher.submissions) == 1
    assert result.candidates[0].alternatives[0].alternative_id in (
        result.decision.selected_alternative_ids
    )


def test_b0_dispatches_every_realization_without_fallback_collapsing() -> None:
    providers, _, case = _fixture()
    dispatcher = RecordingDispatcher()
    bridge = LivePlanningBridge(
        coordinator=ControlledPlanningCoordinator(ProduceAllPolicy()),
        provider_registry=providers,
        dispatcher=dispatcher,
    )
    result = bridge.plan_and_dispatch(
        case,
        trigger=PlanningTrigger.ADMISSION,
        task_policy=TaskSchedulingPolicy(request_id=case.request_id),
    )
    assert len(result.candidates) == len(result.decision.selected_alternative_ids)
    assert len(dispatcher.submissions) == len(result.candidates)
    assert all(len(batch) == 1 for batch in dispatcher.submissions)


def test_b0_sensor_fanout_adds_coverage_without_replacing_provider_union() -> None:
    providers, _, case = _fixture()
    original = case.whole_event_graph.alternatives[0]
    second_inputs = tuple(
        item.model_copy(update={"source_id": "second-camera-source"})
        if item.kind == ExternalInputKind.LIVE_SOURCE
        else item
        for item in original.external_inputs
    )
    second_chain = original.model_copy(
        update={
            "alternative_id": "second-provider-realization",
            "external_inputs": second_inputs,
            "estimated_completion_ms": original.estimated_completion_ms + 100,
        }
    )
    graph = case.whole_event_graph.model_copy(
        update={"alternatives": (*case.whole_event_graph.alternatives, second_chain)}
    )
    case = BaselinePlanningCase(
        **{
            **case.__dict__,
            "frontier_graph": graph,
            "whole_event_graph": graph,
        }
    )
    bridge = LivePlanningBridge(
        coordinator=ControlledPlanningCoordinator(ProduceAllPolicy()),
        provider_registry=providers,
        dispatcher=RecordingDispatcher(),
        fanout_predicate_ids=frozenset({"FOLLOWS"}),
        fanout_node_ids=frozenset({"sensor_a", "sensor_b"}),
    )

    result = bridge.plan_and_dispatch(
        case,
        trigger=PlanningTrigger.ADMISSION,
        task_policy=TaskSchedulingPolicy(request_id=case.request_id),
    )

    assert "second-provider-realization" in result.decision.selected_alternative_ids
    assert len(result.decision.selected_alternative_ids) >= len(
        ProduceAllPolicy().plan(case).selected_alternative_ids
    )


def test_b0_refreshes_on_semantic_frontier_but_not_resource_epoch() -> None:
    _, _, case = _fixture()
    coordinator = ControlledPlanningCoordinator(ProduceAllPolicy())
    admitted = coordinator.decide(case, trigger=PlanningTrigger.ADMISSION)

    resource_only = BaselinePlanningCase(
        **{**case.__dict__, "resource_epoch": case.resource_epoch + 1}
    )
    unchanged = coordinator.decide(
        resource_only, trigger=PlanningTrigger.RESOURCE_EPOCH
    )
    assert unchanged is admitted

    semantic = BaselinePlanningCase(
        **{**resource_only.__dict__, "semantic_epoch": case.semantic_epoch + 1}
    )
    refreshed = coordinator.decide(
        semantic, trigger=PlanningTrigger.SEMANTIC_FRONTIER
    )
    assert refreshed is not admitted
    assert refreshed.semantic_epoch == semantic.semantic_epoch


def test_b0_semantic_dispatch_skips_strict_whole_event_revalidation() -> None:
    providers, _, case = _fixture()

    class LateBoundOnlyB0:
        baseline_id = BaselineId.B0_PRODUCE_ALL

        def plan(self, case):
            raise AssertionError("whole-event plan must not run on semantic refresh")

        def plan_late_bound(self, case, *, excluded_demand_ids=frozenset()):
            return ProduceAllPolicy().plan(case).model_copy(
                update={"baseline_id": BaselineId.B0_PRODUCE_ALL}
            )

    bridge = LivePlanningBridge(
        coordinator=ControlledPlanningCoordinator(LateBoundOnlyB0()),
        provider_registry=providers,
        dispatcher=RecordingDispatcher(),
    )

    result = bridge.plan_and_dispatch(
        case,
        trigger=PlanningTrigger.SEMANTIC_FRONTIER,
        task_policy=TaskSchedulingPolicy(request_id=case.request_id),
    )

    assert result.decision.baseline_id == BaselineId.B0_PRODUCE_ALL
    assert result.commands


def test_sensor_local_fanout_is_keyed_by_colocated_live_source_node() -> None:
    providers, planner, case = _fixture()
    bridge = LivePlanningBridge(
        coordinator=ControlledPlanningCoordinator(FablePolicy(planner)),
        provider_registry=providers,
        dispatcher=RecordingDispatcher(),
        fanout_predicate_ids=frozenset({"FOLLOWS"}),
        fanout_node_ids=frozenset({"sensor_a", "sensor_b"}),
    )

    alternatives = bridge._fanout_alternatives(
        case, case.frontier_graph
    )

    assert alternatives
    source_nodes = set()
    for alternative in alternatives:
        live_nodes = {
            item.node_id
            for item in alternative.external_inputs
            if item.kind == ExternalInputKind.LIVE_SOURCE
        }
        if live_nodes:
            result_node = alternative.step_placements[-1].node_id
            assert result_node in live_nodes
            source_nodes.add(result_node)
    assert source_nodes == {"sensor_a", "sensor_b"}


def test_controlled_baseline_fanout_cannot_change_selected_chain() -> None:
    providers, planner, case = _fixture()
    bridge = LivePlanningBridge(
        coordinator=ControlledPlanningCoordinator(FablePolicy(planner)),
        provider_registry=providers,
        dispatcher=RecordingDispatcher(),
        fanout_predicate_ids=frozenset({"FOLLOWS"}),
        fanout_node_ids=frozenset({"sensor_a", "sensor_b"}),
    )
    # Alternative ordering is not a treatment contract and may change as new
    # provider chains are catalogued. Select a chain already proven to have a
    # source-local fan-out realization, then verify the control restriction.
    unrestricted = bridge._fanout_alternatives(case, case.frontier_graph)
    assert unrestricted
    selected_chain = unrestricted[0].chain_id

    alternatives = bridge._fanout_alternatives(
        case,
        case.frontier_graph,
        allowed_chain_ids_by_demand={
            case.frontier_demands[0].demand_id: frozenset({selected_chain})
        },
    )

    assert alternatives
    assert {item.chain_id for item in alternatives} == {selected_chain}


def test_sensor_local_fanout_honors_bound_camera_reference() -> None:
    providers, planner, case = _fixture()
    demand = case.frontier_demands[0].model_copy(
        update={"bound_roles": {"reference": "camera_fov:sensor_a"}}
    )
    scoped_case = BaselinePlanningCase(
        **{
            **case.__dict__,
            "frontier_demands": (demand,),
            "all_task_demands": (demand,),
        }
    )
    bridge = LivePlanningBridge(
        coordinator=ControlledPlanningCoordinator(FablePolicy(planner)),
        provider_registry=providers,
        dispatcher=RecordingDispatcher(),
        fanout_predicate_ids=frozenset({"FOLLOWS"}),
        fanout_node_ids=frozenset({"sensor_a", "sensor_b"}),
    )

    alternatives = bridge._fanout_alternatives(scoped_case, case.frontier_graph)

    assert alternatives
    assert {
        alternative.step_placements[-1].node_id for alternative in alternatives
    } == {"sensor_a"}


def test_sensor_local_fanout_is_one_recorded_atomic_coverage_plan() -> None:
    providers, planner, case = _fixture()
    dispatcher = RecordingDispatcher()
    bridge = LivePlanningBridge(
        coordinator=ControlledPlanningCoordinator(FablePolicy(planner)),
        provider_registry=providers,
        dispatcher=dispatcher,
        fanout_predicate_ids=frozenset({"FOLLOWS"}),
        fanout_node_ids=frozenset({"sensor_a", "sensor_b"}),
        fanout_batch_size=1,
        fanout_batch_interval_seconds=0.01,
        staged_fanout_predicate_ids=frozenset({"FOLLOWS"}),
    )

    result = bridge.plan_and_dispatch(
        case,
        trigger=PlanningTrigger.ADMISSION,
        task_policy=TaskSchedulingPolicy(request_id=case.request_id),
    )

    # One demand spanning two sensors is represented and submitted as one
    # capacity-accounted plan, rather than silently expanding after planning.
    assert len(result.candidates) == 1
    assert result.candidates[0].replicated_demand_execution is True
    assert len(result.candidates[0].alternatives) == 2
    assert set(result.decision.selected_node_ids) == {"sensor_a", "sensor_b"}
    assert len(dispatcher.submissions) == 1
    assert len(result.commands) == 1
