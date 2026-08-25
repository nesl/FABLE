from __future__ import annotations

from datetime import timedelta
import json

from fable.common.time import EventTimeInterval, utc_now
from fable.distributed.demo import build_replay_audio_candidate
from fable.distributed.models import (
    AgentProviderStatus,
    CancelProviderCommand,
    ProviderRuntimeSpec,
    ReadinessProbe,
    ReplayOutputAdapter,
    RuntimeMode,
)
from fable.distributed.topics import ack_topic
from fable.distributed.topics import activate_topic

from .fake_phase6_data import make_stack, wait_until


def _candidate(stack, node_id: str, request_id: str):
    now = utc_now()
    return build_replay_audio_candidate(
        provider_registry=stack.registry,
        node_id=node_id,
        source_id=f"{node_id}:audio",
        event_interval=EventTimeInterval(
            start=now - timedelta(days=365),
            end=now + timedelta(days=365),
        ),
        request_id=request_id,
        now=now,
        deadline_seconds=30,
    )


def test_reference_providers_run_end_to_end_on_two_logical_nodes(tmp_path):
    stack = make_stack(tmp_path, nodes=("sensor_a", "sensor_b"))
    try:
        candidates = (
            _candidate(stack, "sensor_a", "task_a"),
            _candidate(stack, "sensor_b", "task_b"),
        )
        batch, commands = stack.orchestrator.submit_candidates(candidates, now=utc_now())
        assert len(batch.admitted_plan_ids) == 2
        assert {command.node_id for command in commands} == {"sensor_a", "sensor_b"}
        assert wait_until(lambda: len(stack.received_results) == 2)
        assert {result.request_id for result in stack.received_results} == {"task_a", "task_b"}
        assert len(stack.store.list_raw("results")) == 2
        assert stack.orchestrator.messenger.outbox.pending_count == 0
        assert all(agent.messenger.outbox.pending_count == 0 for agent in stack.agents.values())
    finally:
        stack.stop()


def test_duplicate_activate_command_starts_one_logical_invocation(tmp_path):
    stack = make_stack(tmp_path)
    try:
        stack.broker.duplicate_next = 1
        candidate = _candidate(stack, "sensor_a", "duplicate_activation")
        _, commands = stack.orchestrator.submit_candidates((candidate,), now=utc_now())
        assert len(commands) == 1
        assert wait_until(lambda: len(stack.received_results) == 1)
        agent = stack.agents["sensor_a"]
        assert agent.processed.count == 1
        record = next(iter(agent.providers.values()))
        assert len(record.active_leases) == 1
        activation_messages = [
            item for item in stack.broker.published if item[0] == activate_topic("sensor_a")
        ]
        assert len(activation_messages) == 1  # broker duplicated delivery, not publication
    finally:
        stack.stop()


def test_adopted_shared_worker_reactivates_after_last_lease_is_released(tmp_path):
    """A rolling hypothesis must not inherit the prior worker's IDLE state."""

    node_id = "sensor_a"
    runtime = ProviderRuntimeSpec(
        provider_id="audio_event_classifier",
        provider_contract_version=1,
        node_id=node_id,
        mode=RuntimeMode.ADOPT_EXISTING,
        container_name="shared-audio-worker",
        worker_id="shared-worker",
    )
    stack = make_stack(tmp_path, runtimes={(node_id, "audio_event_classifier"): runtime})
    try:
        agent = stack.agents[node_id]
        agent.containers.available_adopted_names.add("shared-audio-worker")
        _, first_commands = stack.orchestrator.submit_candidates(
            (_candidate(stack, node_id, "rolling-1"),), now=utc_now()
        )
        first = first_commands[0]
        assert agent.providers[first.provider_instance_id].status == AgentProviderStatus.READY
        agent.cancel(
            CancelProviderCommand(
                orchestrator_id="orchestrator",
                node_id=node_id,
                provider_instance_id=first.provider_instance_id,
                lease_id=first.lease.lease_id,
                demand_id=first.demand.demand_id,
                application_ack_topic=ack_topic("orchestrator"),
            )
        )
        assert agent.providers[first.provider_instance_id].status == AgentProviderStatus.IDLE

        _, second_commands = stack.orchestrator.submit_candidates(
            (_candidate(stack, node_id, "rolling-2"),), now=utc_now()
        )
        second = second_commands[0]
        assert second.provider_instance_id != first.provider_instance_id
        assert agent.providers[second.provider_instance_id].status == AgentProviderStatus.READY
        assert second.lease.lease_id in agent.providers[second.provider_instance_id].active_leases
    finally:
        stack.stop()


def test_idle_shared_worker_buffers_output_for_next_logical_lease(tmp_path):
    """A live adopted worker must not drop evidence between graph frontiers."""

    node_id = "sensor_a"
    output_topic = f"/{node_id}/audio_detector/detections"
    runtime = ProviderRuntimeSpec(
        provider_id="audio_event_classifier",
        provider_contract_version=1,
        node_id=node_id,
        mode=RuntimeMode.ADOPT_EXISTING,
        container_name="shared-audio-worker",
        worker_id="shared-worker",
        output_topics=(output_topic,),
        output_adapter=ReplayOutputAdapter.AUDIO_DETECTION,
        output_label_aliases={"loud_audio": ("loud_audio",)},
    )
    stack = make_stack(tmp_path, runtimes={(node_id, "audio_event_classifier"): runtime})
    try:
        agent = stack.agents[node_id]
        agent.containers.available_adopted_names.add("shared-audio-worker")
        _, first_commands = stack.orchestrator.submit_candidates(
            (_candidate(stack, node_id, "frontier-1"),), now=utc_now()
        )
        first = first_commands[0]
        agent.cancel(
            CancelProviderCommand(
                orchestrator_id="orchestrator",
                node_id=node_id,
                provider_instance_id=first.provider_instance_id,
                lease_id=first.lease.lease_id,
                demand_id=first.demand.demand_id,
                application_ack_topic=ack_topic("orchestrator"),
            )
        )
        assert agent.providers[first.provider_instance_id].status == AgentProviderStatus.IDLE

        event_time = utc_now().timestamp()
        stack.broker.publish(
            output_topic,
            json.dumps({"t": event_time, "event": "loud_audio", "db": -10}).encode(),
            qos=0,
            retain=False,
        )
        assert not stack.received_results
        assert len(agent._provider_output_cache) == 1

        stack.orchestrator.submit_candidates(
            (_candidate(stack, node_id, "frontier-2"),), now=utc_now()
        )
        assert wait_until(lambda: len(stack.received_results) == 1)
        assert stack.received_results[0].request_id == "frontier-2"
        assert stack.received_results[0].semantic_predicate.predicate_id == "AUDIO_EVENT"
    finally:
        stack.stop()


def test_adopted_replay_audio_container_forwards_only_matching_typed_results(tmp_path):
    node_id = "sensor_a"
    output_topic = f"/{node_id}/audio_detector/detections"
    readiness_topic = f"/readiness/{node_id}/audio_detector"
    runtime = ProviderRuntimeSpec(
        provider_id="audio_event_classifier",
        provider_contract_version=1,
        node_id=node_id,
        mode=RuntimeMode.ADOPT_EXISTING,
        container_name="audio-detector-orin11",
        readiness=ReadinessProbe(
            mqtt_topic=readiness_topic,
            ready_field="ready",
            ready_value=True,
        ),
        output_topics=(output_topic,),
        output_adapter=ReplayOutputAdapter.AUDIO_DETECTION,
        output_label_aliases={"loud_audio": ("loud_audio",)},
    )
    stack = make_stack(tmp_path, runtimes={(node_id, "audio_event_classifier"): runtime})
    try:
        fake_runtime = stack.agents[node_id].containers
        fake_runtime.available_adopted_names.add("audio-detector-orin11")
        candidate = _candidate(stack, node_id, "adopted_audio")
        stack.orchestrator.submit_candidates((candidate,), now=utc_now())
        assert next(iter(stack.agents[node_id].providers.values())).status.value == "STARTING"

        stack.broker.publish(
            readiness_topic,
            json.dumps({"ready": True, "service": "audio_detector"}).encode(),
            qos=1,
            retain=True,
        )
        event_time = utc_now().timestamp()
        stack.broker.publish(
            output_topic,
            json.dumps({"t": event_time, "event": "other_sound"}).encode(),
            qos=0,
            retain=False,
        )
        assert not stack.received_results
        stack.broker.publish(
            output_topic,
            json.dumps({"t": event_time, "event": "loud_audio", "db": -10}).encode(),
            qos=0,
            retain=False,
        )
        assert wait_until(lambda: len(stack.received_results) == 1)
        result = stack.received_results[0]
        assert result.semantic_predicate.predicate_id == "AUDIO_EVENT"
        assert result.provenance.provider_id == "audio_event_classifier"
        # Re-delivery of the same provider occurrence is suppressed at the node.
        stack.broker.publish(
            output_topic,
            json.dumps({"t": event_time, "event": "loud_audio", "db": -10}).encode(),
            qos=0,
            retain=False,
        )
        assert len(stack.received_results) == 1
    finally:
        stack.stop()
