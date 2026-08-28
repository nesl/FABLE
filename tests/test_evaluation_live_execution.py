from datetime import timedelta

from evaluation.baselines import BaselinePlanningCase, FablePolicy
from evaluation.live_execution import (
    AuthoritativeLiveExecution,
    LiveRequestState,
    transition_changes_frontier,
)
from evaluation.live_orchestration import LivePlanningBridge
from evaluation.orchestration import ControlledPlanningCoordinator, PlanningTrigger
from fable.common.examples import BASE_TIME
from fable.planning import (
    BoundedLabelPlanner,
    DemandCompiler,
    PhysicalAlternativeGraphBuilder,
    default_predicate_registry,
)
from fable.planning.phase4_testing import fake_follow_alternative_graph
from fable.planning.testing import (
    fake_artifact_catalog,
    fake_deployment,
    fake_follow_frontier,
    fake_provider_registry,
)
from fable.scheduling.capacity import CapacityLedger
from fable.scheduling.control import CheckpointController
from fable.scheduling.lifecycle import ProviderLifecycleManager
from fable.scheduling.models import TaskSchedulingPolicy
from fable.semantic import ScriptedResultSpec, predicate_result_from_spec
from fable.common.schemas import NodeCapacity, NodeHeartbeat, SourceHeartbeat
from fable.common.time import EventTimeInterval

from tests.test_evaluation_live_orchestration import RecordingDispatcher


def test_evidence_only_merge_does_not_trigger_successor_planning() -> None:
    assert transition_changes_frontier("APPLIED")
    assert transition_changes_frontier("FORKED")
    assert not transition_changes_frontier("MERGED")
    assert not transition_changes_frontier("NOOP")


def test_authoritative_result_advances_runtime_and_dispatches_next_frontier() -> None:
    providers = fake_provider_registry()
    artifacts = fake_artifact_catalog()
    deployment = fake_deployment()
    runtime, hypothesis, _ = fake_follow_frontier()
    initial_graph, initial_demand = fake_follow_alternative_graph(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
    )
    dispatcher = RecordingDispatcher()
    planner = BoundedLabelPlanner(
        provider_registry=providers,
        artifact_catalog=artifacts,
        deployment=deployment,
    )
    bridge = LivePlanningBridge(
        coordinator=ControlledPlanningCoordinator(FablePolicy(planner)),
        provider_registry=providers,
        dispatcher=dispatcher,
    )
    initial_case = BaselinePlanningCase(
        run_id="run",
        trace_id="trace",
        request_id=runtime.config.request_id,
        event_family="route_convoy",
        frontier_demands=(initial_demand,),
        all_task_demands=(initial_demand,),
        frontier_graph=initial_graph,
        whole_event_graph=initial_graph,
        now=BASE_TIME,
    )
    policy = TaskSchedulingPolicy(request_id=runtime.config.request_id)
    bridge.plan_and_dispatch(
        initial_case,
        trigger=PlanningTrigger.ADMISSION,
        task_policy=policy,
    )
    lifecycle = ProviderLifecycleManager(
        provider_registry=providers,
        capacity=CapacityLedger(deployment),
    )
    live = AuthoritativeLiveExecution(
        demand_compiler=DemandCompiler(
            predicate_registry=default_predicate_registry(),
            deployment=deployment,
        ),
        graph_builder=PhysicalAlternativeGraphBuilder(
            provider_registry=providers,
            artifact_catalog=artifacts,
            deployment=deployment,
        ),
        deployment=deployment,
        bridge=bridge,
        checkpoint_controller=CheckpointController(
            lifecycle=lifecycle,
            artifact_catalog=artifacts,
        ),
    )
    live.register(
        LiveRequestState(
            run_id="run",
            trace_id="trace",
            event_family="route_convoy",
            runtime=runtime,
            whole_event_demands=(initial_demand,),
            whole_event_graph=initial_graph,
            task_policy=policy,
            coverage_node_id="sensor_a",
        )
    )
    result = predicate_result_from_spec(
        runtime,
        hypothesis.hypothesis_id,
        ScriptedResultSpec(
            node_key="follower_follows",
            source_id="camera_mobile",
            event_time_interval=EventTimeInterval(
                start=BASE_TIME,
                end=BASE_TIME,
            ),
            introduced={"follower": "vehicle_18"},
            validated={"leader": "vehicle_17"},
        ),
    ).model_copy(update={"demand_id": initial_demand.demand_id})
    progression = live.handle_result(result)
    assert progression.transition.status in {"APPLIED", "FORKED"}
    assert progression.planning
    assert len(dispatcher.submissions) == 2

    duplicate = live.handle_result(result)
    assert duplicate.transition.status == "DUPLICATE"
    assert duplicate.planning == ()
    assert len(dispatcher.submissions) == 2

    watermark_progressions = live.handle_heartbeat(
        NodeHeartbeat(
            node_id="sensor_a",
            session_id="session",
            sequence=1,
            sent_at=BASE_TIME + timedelta(seconds=31),
            sources={
                "camera_mobile": SourceHeartbeat(
                    source_id="camera_mobile",
                    latest_sequence=100,
                    latest_event_time=BASE_TIME + timedelta(seconds=31),
                    operational_coverage=True,
                ),
                # Agents observe the shared MQTT bus, so their heartbeat may
                # contain another node's newer camera. It must not satisfy the
                # seed camera's logical absence watermark.
                "camera_downstream": SourceHeartbeat(
                    source_id="camera_downstream",
                    latest_sequence=200,
                    latest_event_time=BASE_TIME + timedelta(seconds=90),
                    operational_coverage=True,
                ),
            },
            capacity=NodeCapacity(cpu_free_cores=1, memory_free_mb=1024),
        )
    )
    assert len(watermark_progressions) == 1
    assert watermark_progressions[0].transition.status == "WINDOW_CLOSED"
    assert watermark_progressions[0].terminal_lifecycles
    assert len(watermark_progressions[0].detections) == 1
