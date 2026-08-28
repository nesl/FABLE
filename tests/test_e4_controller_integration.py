from __future__ import annotations

import ast
from datetime import timedelta
from pathlib import Path

import pytest

from evaluation.e4_worker import E4Worker
from evaluation.live_planning import LiveBaselinePlanningPolicy
from evaluation.schemas import BaselineId
from fable.common.examples import BASE_TIME
from fable.common.schemas import RuntimeNodeUpdate
from fable.common.time import EventTimeInterval
from fable.distributed.models import (
    EventRequestSubmission,
    ProviderRuntimeSpec,
    RuntimeDisturbanceRequest,
    RuntimeMode,
)
from fable.distributed.transport import InMemoryTransport
from fable.orchestration import FableController
from fable.planning import ArtifactCatalog, RuntimeDeploymentView
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import SensorSource
from fable.planning.testing import fake_deployment
from fable.semantic import AuthoredEventFamilyRegistry, AuthoredGraphBuilder, EventRequestCompiler
from fable.semantic.models import ScriptedResultSpec
from fable.semantic.testing import predicate_result_from_spec
from fable.common.ids import uuid7

from .fake_phase6_data import make_stack, wait_until


def _two_audio_graph(_parameters):
    builder = AuthoredGraphBuilder(namespace="tests.e4", name="E4 two audio checkpoints")
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


def _deployment() -> DeploymentGraph:
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


def _controller(
    stack, deployment: DeploymentGraph, *, planning_event_sink=None
) -> FableController:
    families = AuthoredEventFamilyRegistry()
    families.register("two_audio", _two_audio_graph)
    policies = {
        baseline.value: LiveBaselinePlanningPolicy(baseline)
        for baseline in (
            BaselineId.B0_PRODUCE_ALL,
            BaselineId.B1_HANDWRITTEN_STATIC,
            BaselineId.B2_FRONTIER_FIXED_REALIZATION,
            BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
            BaselineId.B4_GREEDY_FRONTIER,
        )
    }
    controller = FableController(
        orchestrator=stack.orchestrator,
        provider_registry=stack.registry,
        deployment_view=RuntimeDeploymentView(deployment),
        artifact_catalog=ArtifactCatalog(),
        request_compiler=EventRequestCompiler(registry=families),
        planning_policies=policies,
        planning_event_sink=planning_event_sink,
    )
    controller.bind()
    return controller


def test_redesigned_controller_emits_admitted_plan_telemetry(tmp_path: Path):
    events = []
    stack = make_stack(tmp_path)
    try:
        controller = _controller(
            stack, fake_deployment(), planning_event_sink=events.append
        )
        response = controller.submit_event(
            _submission("telemetry", "test", BaselineId.B3_TASK_RESOURCE_ADAPTIVE)
        )
        assert response.accepted
        assert events
        event = events[0]
        assert event.request_id == "telemetry"
        assert event.trigger == "ADMISSION"
        assert event.selected_alternative_ids
        assert event.activated_provider_keys
        assert event.commands
        assert event.planning_scope == "REDESIGNED_CONTROLLER_FRONTIER"
        assert event.frozen is False
    finally:
        stack.stop()


def _submission(request_id: str, submitter_id: str, baseline: BaselineId):
    return EventRequestSubmission(
        submitter_id=submitter_id,
        request_id=request_id,
        family_id="two_audio",
        planning_policy_id=baseline.value,
        event_time_window=EventTimeInterval(
            start=BASE_TIME - timedelta(minutes=1),
            end=BASE_TIME + timedelta(minutes=1),
        ),
    )


@pytest.mark.parametrize(
    "baseline",
    [
        BaselineId.B0_PRODUCE_ALL,
        BaselineId.B1_HANDWRITTEN_STATIC,
        BaselineId.B2_FRONTIER_FIXED_REALIZATION,
        BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
        BaselineId.B4_GREEDY_FRONTIER,
    ],
)
def test_b1_b3_live_policies_drive_new_controller_and_terminal_stream(tmp_path: Path, baseline):
    runtime = ProviderRuntimeSpec(
        provider_id="audio_event_classifier",
        provider_contract_version=1,
        node_id="sensor_a",
        mode=RuntimeMode.REFERENCE,
        reference_delay_ms=10,
    )
    stack = make_stack(tmp_path, runtimes={("sensor_a", "audio_event_classifier"): runtime})
    client = InMemoryTransport(stack.broker)
    try:
        controller = _controller(stack, fake_deployment())
        worker = E4Worker(transport=client, submitter_id=f"worker-{baseline.value}")
        worker.bind()
        client.start()
        response = worker.submit_event(
            _submission(f"e4-{baseline.value}", worker.submitter_id, baseline),
            timeout=2.0,
        )
        assert response.accepted
        if baseline == BaselineId.B1_HANDWRITTEN_STATIC:
            # B1 must ignore resource/network replanning while its current
            # frontier is live. This freezes placement, not the logical
            # demand: solved demands still retire and the already-running
            # authored container is reused by the successor frontier.
            assert wait_until(
                lambda: len(stack.lifecycle.active_leases) == 1,
                timeout=1.0,
            )
            prior = stack.lifecycle.active_leases[0]
            controller.handle_replan(
                "network",
                (prior.lease.demand_id,),
                "test disturbance must not adapt B1",
            )
            assert any(
                lease.lease.lease_id == prior.lease.lease_id
                for lease in stack.lifecycle.active_leases
            )
        terminal = worker.wait_terminal(response.request_id, timeout=3.0)
        assert terminal.request_id == response.request_id
        assert terminal.family_id == "two_audio"
        assert len(terminal.provenance_result_ids) == 2
        assert len(stack.store.list_raw("terminal_events")) == 1
        if baseline == BaselineId.B1_HANDWRITTEN_STATIC:
            assert len(stack.lifecycle.active_leases) == 0
    finally:
        client.stop()
        stack.stop()


def test_b1_defers_early_downstream_evidence_until_authored_frontier(tmp_path: Path):
    """A fixed whole-event pipeline must not lose valid successor evidence."""

    runtime = ProviderRuntimeSpec(
        provider_id="audio_event_classifier",
        provider_contract_version=1,
        node_id="sensor_a",
        mode=RuntimeMode.REFERENCE,
        reference_delay_ms=10_000,
    )
    stack = make_stack(
        tmp_path,
        runtimes={("sensor_a", "audio_event_classifier"): runtime},
    )
    try:
        controller = _controller(stack, fake_deployment())
        response = controller.submit_event(
            _submission("b1-early-evidence", "test", BaselineId.B1_HANDWRITTEN_STATIC)
        )
        state = controller.requests[response.request_id]
        hypothesis_id = response.hypothesis_ids[0]
        initial_frontier = state.runtime.get_frontier(hypothesis_id)
        assert initial_frontier is not None
        initial_node_id = initial_frontier.snapshot.enabled_node_ids[0]
        initial_key = next(
            key
            for key, node in state.runtime.graph.nodes_by_key.items()
            if node.node_id == initial_node_id
        )
        successor_key = "second" if initial_key == "first" else "first"
        first = predicate_result_from_spec(
            state.runtime,
            hypothesis_id,
            ScriptedResultSpec(
                node_key=initial_key,
                event_time_interval=EventTimeInterval(
                    start=BASE_TIME,
                    end=BASE_TIME + timedelta(seconds=1),
                ),
                source_id="microphone_store",
                provider_id="audio_event_classifier",
            ),
        )
        second_node = state.runtime.graph.nodes_by_key[successor_key]
        early_second = first.model_copy(
            update={
                "result_id": uuid7(),
                "occurrence_id": str(uuid7()),
                "graph_node_id": second_node.node_id,
                "semantic_predicate": second_node.predicate,
                "event_time_interval": EventTimeInterval(
                    start=BASE_TIME + timedelta(seconds=2),
                    end=BASE_TIME + timedelta(seconds=3),
                ),
            }
        )

        controller.handle_result(early_second)
        assert state.deferred_static_results == [early_second]
        controller.handle_result(first)

        assert state.deferred_static_results == []
        completed = [
            hypothesis
            for hypothesis in state.runtime.hypotheses
            if hypothesis.lifecycle.value == "COMPLETED"
        ]
        assert len(completed) == 1
    finally:
        stack.stop()


def test_e4_disturbance_is_typed_acknowledged_and_replans_b3(tmp_path: Path):
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
    stack = make_stack(tmp_path, nodes=("sensor_a", "sensor_b"), runtimes=runtimes)
    client = InMemoryTransport(stack.broker)
    try:
        controller = _controller(stack, _deployment())
        worker = E4Worker(transport=client, submitter_id="e4-disturbance-worker")
        worker.bind()
        client.start()
        response = worker.submit_event(
            _submission(
                "e4-disturbance",
                worker.submitter_id,
                BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
            ),
            timeout=2.0,
        )
        assert response.accepted
        assert wait_until(
            lambda: len(stack.store.list_raw("results")) == 1
            and len(stack.lifecycle.active_leases) == 1,
            timeout=2.0,
        )
        prior = stack.lifecycle.active_leases[0]
        prior_epoch = controller.deployment_view.resource_epoch
        ack = worker.inject_disturbance(
            RuntimeDisturbanceRequest(
                submitter_id=worker.submitter_id,
                disturbance_id="drop-selected-node",
                reason="E4 compute disturbance",
                node_updates=(RuntimeNodeUpdate(node_id=prior.lease.node_id, available=False),),
            ),
            timeout=2.0,
        )
        assert ack.accepted
        assert ack.changed
        assert ack.previous_resource_epoch == prior_epoch
        assert ack.resource_epoch == prior_epoch + 1
        assert prior.lease.demand_id in ack.affected_demand_ids
        assert response.request_id in ack.replanned_request_ids
        assert wait_until(
            lambda: any(
                item.request_id == response.request_id
                and item.lease.node_id != prior.lease.node_id
                for item in stack.lifecycle.active_leases
            ),
            timeout=1.0,
        )
        terminal = worker.wait_terminal(response.request_id, timeout=3.0)
        assert terminal.request_id == response.request_id
    finally:
        client.stop()
        stack.stop()


def test_e4_worker_does_not_depend_on_legacy_live_evaluation_normalizer() -> None:
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "evaluation/e4_worker.py").read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "evaluation.runtime_logging" not in imports
    assert "evaluation.baselines.models" not in imports


def test_evaluation_runner_accepts_typed_terminal_event(tmp_path: Path) -> None:
    from evaluation.runner import EvaluationRunner
    from evaluation.schemas import EvaluationMode
    from fable.common.ids import uuid7
    from fable.common.schemas import TerminalComplexEvent

    event = TerminalComplexEvent(
        request_id="runner-terminal",
        family_id="two_audio",
        hypothesis_id=uuid7(),
        graph_hash="sha256:" + "1" * 64,
        event_time_window=EventTimeInterval(
            start=BASE_TIME,
            end=BASE_TIME + timedelta(seconds=1),
        ),
    )
    runner = EvaluationRunner(tmp_path, mode=EvaluationMode.FULL_STACK)
    record = runner.record_terminal_event(
        event,
        run_id="run-terminal",
        trace_id="trace-terminal",
        baseline_id=BaselineId.FABLE,
    )
    assert record.result_id == str(event.message_id)
    assert record.event_family == "two_audio"
    assert len(runner.store.read("complex_event_result")) == 1
