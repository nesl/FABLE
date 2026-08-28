from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from fable.common.examples import BASE_TIME
from fable.common.ids import uuid7
from fable.common.schemas import PlanStep
from fable.common.time import EventTimeInterval
from fable.distributed.codec import decode_model, encode_model
from fable.distributed.config import ProviderRuntimeResolver
from fable.distributed.models import (
    EventRequestResponse,
    EventRequestSubmission,
    ExecutionProfile,
    ProviderRuntimeSpec,
    ResourceLimits,
    RuntimeMode,
)
from fable.distributed.plan_execution import PlanExecutionTracker
from fable.distributed.topics import event_request_topic, event_response_topic
from fable.distributed.transport import InMemoryTransport
from fable.orchestration import FableController
from fable.planning import ArtifactCatalog, RuntimeDeploymentView
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import SensorSource
from fable.planning.testing import fake_deployment
from fable.semantic import (
    AuthoredEventFamilyRegistry,
    AuthoredGraphBuilder,
    EventRequestCompiler,
)

from .fake_phase6_data import make_stack, wait_until


def _two_audio_graph(_parameters):
    builder = AuthoredGraphBuilder(namespace="tests.closed_loop", name="Two audio checkpoints")
    location = (("location", "store", "zone"),)
    first = builder.primitive(
        "first",
        name="First audio event",
        predicate_id="AUDIO_EVENT",
        roles=location,
        parameters={"label": "first"},
        checkpoint=True,
    )
    second = builder.primitive(
        "second",
        name="Second audio event",
        predicate_id="AUDIO_EVENT",
        roles=location,
        parameters={"label": "second"},
        checkpoint=True,
    )
    root = builder.sequence("root", (first, second), name="First then second")
    return builder.root(root).compile()


def _controller(
    stack,
    *,
    execution_profile=ExecutionProfile.DEVELOPMENT,
    deployment=None,
):
    families = AuthoredEventFamilyRegistry()
    families.register("two_audio", _two_audio_graph)
    controller = FableController(
        orchestrator=stack.orchestrator,
        provider_registry=stack.registry,
        deployment_view=RuntimeDeploymentView(deployment or fake_deployment()),
        artifact_catalog=ArtifactCatalog(),
        request_compiler=EventRequestCompiler(registry=families),
        execution_profile=execution_profile,
    )
    controller.bind()
    return controller




def _dual_audio_deployment() -> DeploymentGraph:
    base = fake_deployment()
    nodes = []
    for node in base.nodes.values():
        if node.node_id == "sensor_b":
            node = node.model_copy(
                update={"capabilities": tuple(sorted(set(node.capabilities) | {"audio"}))}
            )
        nodes.append(node)
    sources = list(base.sources.values())
    sources.append(
        SensorSource(
            source_id="microphone_store_b",
            node_id="sensor_b",
            region="store",
            modalities=("audio",),
            live_data_types=("audio_segment.v1",),
            coverage_regions=("store",),
            raw_buffer_interval=EventTimeInterval(
                start=BASE_TIME - timedelta(minutes=5),
                end=BASE_TIME + timedelta(minutes=5),
            ),
        )
    )
    return DeploymentGraph(nodes=tuple(nodes), sources=tuple(sources), links=base.links)


def _submission(request_id: str) -> EventRequestSubmission:
    return EventRequestSubmission(
        submitter_id=f"submitter-{request_id}",
        request_id=request_id,
        family_id="two_audio",
        event_time_window=EventTimeInterval(
            start=BASE_TIME - timedelta(minutes=1),
            end=BASE_TIME + timedelta(minutes=1),
        ),
    )


def test_event_request_api_runs_semantic_planning_feedback_loop(tmp_path: Path):
    runtime = ProviderRuntimeSpec(
        provider_id="audio_event_classifier",
        provider_contract_version=1,
        node_id="sensor_a",
        mode=RuntimeMode.REFERENCE,
        reference_delay_ms=20,
    )
    stack = make_stack(
        tmp_path,
        runtimes={("sensor_a", "audio_event_classifier"): runtime},
    )
    submitter = InMemoryTransport(stack.broker)
    try:
        controller = _controller(stack)
        request = _submission("closed-loop")
        responses: list[EventRequestResponse] = []
        submitter.subscribe(
            event_response_topic(request.submitter_id),
            lambda _topic, payload: responses.append(decode_model(payload, EventRequestResponse)),
            qos=1,
        )
        submitter.start()
        stack.broker.publish(
            event_request_topic("orchestrator"),
            encode_model(request),
            qos=1,
            retain=False,
        )
        assert wait_until(lambda: len(responses) == 1)
        assert responses[0].accepted
        assert len(responses[0].admitted_plan_ids) == 1

        # The first result advances the semantic frontier and causes a second
        # plan; the second result completes the event exactly once.
        assert wait_until(lambda: len(stack.store.list_raw("results")) == 2)
        assert wait_until(lambda: len(stack.store.list_raw("emitted_events")) == 1)
        state = controller.requests[request.request_id]
        hypothesis = state.runtime.hypotheses[0]
        assert hypothesis.lifecycle.value == "COMPLETED"
        assert hypothesis.version == 2
        assert len(hypothesis.provenance_result_ids) == 2
        assert len(stack.store.list_raw("artifacts")) >= 2
    finally:
        submitter.stop()
        stack.stop()



def test_closed_loop_replans_active_frontier_after_node_loss(tmp_path: Path):
    runtimes = {
        (node_id, "audio_event_classifier"): ProviderRuntimeSpec(
            provider_id="audio_event_classifier",
            provider_contract_version=1,
            node_id=node_id,
            mode=RuntimeMode.REFERENCE,
            reference_delay_ms=200,
        )
        for node_id in ("sensor_a", "sensor_b")
    }
    stack = make_stack(
        tmp_path,
        nodes=("sensor_a", "sensor_b"),
        runtimes=runtimes,
    )
    try:
        controller = _controller(stack, deployment=_dual_audio_deployment())
        response = controller.submit_event(_submission("closed-loop-replan"))
        assert response.accepted
        assert len(response.admitted_plan_ids) == 1

        # The first checkpoint completes normally. The second frontier should
        # then have exactly one active logical lease.
        assert wait_until(
            lambda: len(stack.store.list_raw("results")) == 1
            and len(stack.lifecycle.active_leases) == 1,
            timeout=2.0,
        )
        prior = stack.lifecycle.active_leases[0]
        failed_node = prior.lease.node_id
        demand_id = prior.lease.demand_id

        # Simulate loss of the selected compute node while the second checkpoint
        # is in flight. The controller must cancel that invocation and replan the
        # same frontier on the other feasible node.
        controller.handle_replan(failed_node, (demand_id,), "test node capacity loss")
        assert wait_until(
            lambda: any(
                item.request_id == "closed-loop-replan"
                and item.hypothesis_id == prior.hypothesis_id
                and item.lease.node_id != failed_node
                for item in stack.lifecycle.active_leases
            ),
            timeout=1.0,
        )

        # The cancelled delayed reference invocation is suppressed; only the
        # replacement result completes the semantic event, exactly once.
        assert wait_until(lambda: len(stack.store.list_raw("results")) == 2, timeout=2.0)
        assert wait_until(lambda: len(stack.store.list_raw("emitted_events")) == 1)
        hypothesis = controller.requests["closed-loop-replan"].runtime.hypotheses[0]
        assert hypothesis.lifecycle.value == "COMPLETED"
        assert len(hypothesis.provenance_result_ids) == 2
    finally:
        stack.stop()


def test_soft_network_replan_keeps_evidence_path_until_replacement_is_admitted(
    tmp_path: Path,
):
    runtimes = {
        (node_id, "audio_event_classifier"): ProviderRuntimeSpec(
            provider_id="audio_event_classifier",
            provider_contract_version=1,
            node_id=node_id,
            mode=RuntimeMode.REFERENCE,
            reference_delay_ms=250,
        )
        for node_id in ("sensor_a", "sensor_b")
    }
    stack = make_stack(
        tmp_path,
        nodes=("sensor_a", "sensor_b"),
        runtimes=runtimes,
    )
    try:
        controller = _controller(stack, deployment=_dual_audio_deployment())
        response = controller.submit_event(_submission("soft-network-replan"))
        assert response.accepted
        assert wait_until(
            lambda: len(stack.store.list_raw("results")) == 1
            and len(stack.lifecycle.active_leases) == 1,
            timeout=2.0,
        )
        prior = stack.lifecycle.active_leases[0]

        # A link-cost/resource epoch is a soft disturbance. The existing
        # provider may still return valid evidence, so replacement admission
        # must happen before the old demand is retired.
        controller.handle_replan(
            "network",
            (prior.lease.demand_id,),
            "test soft link degradation",
        )

        assert wait_until(lambda: len(stack.store.list_raw("emitted_events")) == 1)
        hypothesis = controller.requests["soft-network-replan"].runtime.hypotheses[0]
        assert hypothesis.lifecycle.value == "COMPLETED"
        assert len(hypothesis.provenance_result_ids) == 2
    finally:
        stack.stop()




def test_real_execution_profile_will_not_dispatch_reference_runtime(tmp_path: Path):
    runtime = ProviderRuntimeSpec(
        provider_id="audio_event_classifier",
        provider_contract_version=1,
        node_id="sensor_a",
        mode=RuntimeMode.REFERENCE,
        reference_delay_ms=1,
    )
    stack = make_stack(
        tmp_path,
        runtimes={("sensor_a", "audio_event_classifier"): runtime},
    )
    try:
        controller = _controller(stack, execution_profile=ExecutionProfile.REAL)
        response = controller.submit_event(_submission("real-no-oracle"))
        assert response.accepted
        assert response.admitted_plan_ids == ()
        assert response.command_message_ids == ()
        assert stack.store.list_raw("results") == ()
    finally:
        stack.stop()


def test_plan_execution_tracker_gates_cross_worker_dependencies():
    runtimes = {
        ("sensor_a", "upstream"): ProviderRuntimeSpec(
            provider_id="upstream",
            provider_contract_version=1,
            node_id="sensor_a",
            mode=RuntimeMode.REFERENCE,
            worker_id="worker-a",
            worker_resource_limits=ResourceLimits(cpu_cores=1, memory_mb=128),
        ),
        ("sensor_a", "downstream"): ProviderRuntimeSpec(
            provider_id="downstream",
            provider_contract_version=1,
            node_id="sensor_a",
            mode=RuntimeMode.REFERENCE,
            worker_id="worker-b",
            worker_resource_limits=ResourceLimits(cpu_cores=1, memory_mb=128),
        ),
    }
    tracker = PlanExecutionTracker(ProviderRuntimeResolver(runtimes))
    plan_id = uuid7()
    candidate = SimpleNamespace(
        plan=SimpleNamespace(
            plan_id=plan_id,
            steps=(
                PlanStep(
                    step_id="a",
                    provider_id="upstream",
                    node_id="sensor_a",
                    output_data_types=("intermediate.v1",),
                ),
                PlanStep(
                    step_id="b",
                    provider_id="downstream",
                    node_id="sensor_a",
                    input_data_types=("intermediate.v1",),
                    output_data_types=("predicate_match.v1",),
                    depends_on_step_ids=("a",),
                ),
            ),
        )
    )
    assert tracker.register(candidate) == ("a",)
    assert tracker.complete_step(plan_id, "a") == ("b",)
    assert tracker.complete_step(plan_id, "b") == ()


def test_plan_execution_tracker_coactivates_typed_broker_pipeline():
    runtimes = {
        ("sensor_a", "upstream"): ProviderRuntimeSpec(
            provider_id="upstream",
            provider_contract_version=1,
            node_id="sensor_a",
            mode=RuntimeMode.REFERENCE,
            worker_id="worker-a",
            artifact_topic_outputs={"intermediate.v1": "/sensor_a/intermediate"},
            artifact_broker_scope_id="evaluation-mqtt",
        ),
        ("site", "downstream"): ProviderRuntimeSpec(
            provider_id="downstream",
            provider_contract_version=1,
            node_id="site",
            mode=RuntimeMode.REFERENCE,
            worker_id="worker-b",
            artifact_topic_inputs={"intermediate.v1": "/sensor_a/intermediate"},
            artifact_broker_scope_id="evaluation-mqtt",
        ),
    }
    tracker = PlanExecutionTracker(ProviderRuntimeResolver(runtimes))
    candidate = SimpleNamespace(
        plan=SimpleNamespace(
            plan_id=uuid7(),
            steps=(
                PlanStep(
                    step_id="a",
                    provider_id="upstream",
                    node_id="sensor_a",
                    output_data_types=("intermediate.v1",),
                ),
                PlanStep(
                    step_id="b",
                    provider_id="downstream",
                    node_id="site",
                    input_data_types=("intermediate.v1",),
                    depends_on_step_ids=("a",),
                ),
            ),
        )
    )
    assert set(tracker.register(candidate)) == {"a", "b"}


def test_shared_worker_key_coactivates_internal_logical_steps():
    shared_limits = ResourceLimits(cpu_cores=2, memory_mb=512)
    runtimes = {
        ("sensor_a", provider): ProviderRuntimeSpec(
            provider_id=provider,
            provider_contract_version=1,
            node_id="sensor_a",
            mode=RuntimeMode.REFERENCE,
            worker_id="shared-worker",
            worker_resource_limits=shared_limits,
        )
        for provider in ("upstream", "downstream")
    }
    resolver = ProviderRuntimeResolver(runtimes)
    tracker = PlanExecutionTracker(resolver)
    plan_id = uuid7()
    candidate = SimpleNamespace(
        plan=SimpleNamespace(
            plan_id=plan_id,
            steps=(
                PlanStep(step_id="a", provider_id="upstream", node_id="sensor_a"),
                PlanStep(
                    step_id="b",
                    provider_id="downstream",
                    node_id="sensor_a",
                    depends_on_step_ids=("a",),
                ),
            ),
        )
    )
    assert set(tracker.register(candidate)) == {"a", "b"}
    owner_a, reservation_a = resolver.capacity_group(
        "sensor_a", "upstream", fake_reservation("sensor_a"), "fallback-a"
    )
    owner_b, reservation_b = resolver.capacity_group(
        "sensor_a", "downstream", fake_reservation("sensor_a"), "fallback-b"
    )
    assert owner_a == owner_b == "worker:sensor_a:shared-worker"
    assert reservation_a == reservation_b


def test_named_legacy_container_is_a_shared_physical_capacity_group():
    """Logical leases adopting one named container must not reserve it N times."""

    runtimes = {
        ("site", provider): ProviderRuntimeSpec(
            provider_id=provider,
            provider_contract_version=1,
            node_id="site",
            mode=RuntimeMode.ADOPT_EXISTING,
            container_name="identity-site",
        )
        for provider in ("identity-a", "identity-b")
    }
    resolver = ProviderRuntimeResolver(runtimes)

    owner_a, reservation_a = resolver.capacity_group(
        "site", "identity-a", fake_reservation("site"), "logical-a"
    )
    owner_b, reservation_b = resolver.capacity_group(
        "site", "identity-b", fake_reservation("site"), "logical-b"
    )

    assert owner_a == owner_b == "worker:site:identity-site"
    assert reservation_a == reservation_b
    assert resolver.worker_key("site", "identity-a") == "site/identity-site"


def fake_reservation(node_id: str):
    from fable.common.schemas import ResourceReservation

    return ResourceReservation(node_id=node_id, cpu_cores=0.1, memory_mb=32)
