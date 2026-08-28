#!/usr/bin/env python3
"""FABLE Phase-6 distributed orchestrator service for the replay stack."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
import threading
from time import perf_counter_ns
from datetime import UTC, datetime, timedelta

from fable.distributed.config import ProviderRuntimeResolver, load_deployment_graph
from fable.distributed.heartbeat import HeartbeatMonitor
from fable.distributed.models import ExecutionProfile
from fable.distributed.codec import decode_model, encode_model
from fable.distributed.models import (
    EventRequestSubmission,
    ResourceChange,
    ResourceChangeAck,
)
from fable.distributed.topics import (
    live_request_cancel_response_topic,
    live_request_cancel_topic,
    live_request_progress_topic,
    live_request_response_topic,
    live_request_topic,
    resource_change_ack_topic,
    resource_change_topic,
    terminal_event_filter,
)
from fable.common.schemas import TerminalComplexEvent
from fable.common.ids import deterministic_id
from fable.distributed.orchestrator import DistributedOrchestrator
from fable.distributed.outbox import SQLiteOutbox, SQLiteProcessedLedger
from fable.distributed.persistence import MongoStateStore
from fable.distributed.transport import PahoMQTTTransport, ReliableMessenger
from fable.orchestration import FableController
from fable.integrations.netwaggle import NetWaggleTelemetrySource
from fable.planning import ArtifactCatalog, RuntimeDeploymentView
from fable.planning.provider_registry import ProviderRegistry
from fable.scheduling.admission import MultiTenantScheduler
from fable.scheduling.capacity import CapacityLedger
from fable.scheduling.lifecycle import ProviderLifecycleManager
from evaluation.live_planning import LiveBaselinePlanningPolicy
from evaluation.e4_campaign import (
    planning_policy_id_for_baseline,
    runtime_disturbance_for_resource_change,
)
from evaluation.deployment_artifacts import load_deployment_artifacts
from evaluation.schemas import BaselineId, PlanDecision, ProviderCommand
from evaluation.live_requests import (
    LiveComplexEventCancelRequest,
    LiveComplexEventCancelResponse,
    LiveComplexEventDetection,
    LiveComplexEventProgress,
    LiveComplexEventRequest,
    LiveComplexEventResponse,
)


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("fable.orchestrator.service")


def bind_evaluation_compatibility(controller, orchestrator) -> None:
    """Bridge the typed legacy replay driver onto the redesigned controller.

    This adapter is deliberately confined to the evaluation service boundary;
    no legacy request or baseline semantics enter the core controller.
    """

    transport = orchestrator.transport
    submitters: dict[str, str] = {}
    evaluation_requests: dict[str, LiveComplexEventRequest] = {}

    def publish_planning_event(event) -> None:
        request = evaluation_requests.get(event.request_id)
        if request is None:
            return
        now = datetime.now(UTC)
        record = PlanDecision(
            run_id=request.run_id,
            baseline_id=request.baseline_id,
            trace_id=request.trace_id,
            request_id=request.request_id,
            hypothesis_id=str(event.hypothesis_id),
            event_time=now,
            wall_timestamp=now,
            monotonic_timestamp_ns=perf_counter_ns(),
            decision_id=deterministic_id(
                "redesigned_controller_plan",
                {
                    "request_id": request.request_id,
                    "checkpoint_id": str(event.checkpoint_id),
                    "semantic_epoch": event.semantic_epoch,
                    "resource_epoch": event.resource_epoch,
                    "alternatives": event.selected_alternative_ids,
                },
                length=32,
            ),
            checkpoint_id=str(event.checkpoint_id),
            planning_scope="REDESIGNED_CONTROLLER_FRONTIER",
            selected_alternative_ids=event.selected_alternative_ids,
            selected_chain_ids=event.selected_chain_ids,
            selected_node_ids=event.selected_node_ids,
            activated_provider_keys=event.activated_provider_keys,
            predicted_completion_ms=event.predicted_completion_ms,
            predicted_transfer_bytes=event.predicted_transfer_bytes,
            reason=event.reason,
            resource_epoch=event.resource_epoch,
            semantic_epoch=event.semantic_epoch,
            replan_trigger=event.trigger,
            metadata={"policy_id": event.policy_id},
        )
        transport.publish(
            "fable/v1/evaluation/record/plan_decision",
            record.model_dump_json(exclude_none=True).encode(),
            qos=1,
            retain=False,
        )
        for command in event.commands:
            demand = getattr(command, "demand", None)
            command_record = ProviderCommand(
                run_id=request.run_id,
                baseline_id=request.baseline_id,
                trace_id=request.trace_id,
                request_id=request.request_id,
                hypothesis_id=(
                    str(demand.hypothesis_id) if demand is not None else None
                ),
                provider_id=getattr(
                    getattr(command, "runtime", None), "provider_id", None
                ),
                event_time=getattr(command, "sent_at", now),
                wall_timestamp=now,
                monotonic_timestamp_ns=perf_counter_ns(),
                command_id=str(command.message_id),
                command="ACTIVATE",
                provider_instance_id=getattr(
                    command, "provider_instance_id", None
                ),
                demand_ids=(
                    (str(demand.demand_id),) if demand is not None else ()
                ),
                node_id=str(getattr(command, "node_id", "unknown")),
                emitted_at=getattr(command, "sent_at", now),
                metadata={"checkpoint_id": str(event.checkpoint_id)},
            )
            transport.publish(
                "fable/v1/evaluation/record/provider_command",
                command_record.model_dump_json(exclude_none=True).encode(),
                qos=1,
                retain=False,
            )

    controller.planning_event_sink = publish_planning_event

    def on_request(_topic: str, payload: bytes) -> None:
        try:
            request = decode_model(payload, LiveComplexEventRequest)
            submitters[request.request_id] = request.submitter_id
            evaluation_requests[request.request_id] = request
            submission = EventRequestSubmission(
                message_id=request.message_id,
                submitter_id=request.submitter_id,
                request_id=request.request_id,
                trace_id=request.trace_id,
                baseline_placement_id=request.baseline_placement_id,
                family_id=request.family_id,
                parameters=request.parameters,
                event_time_window=request.allowed_seed_event_time_interval,
                hypothesis_horizon_ms=request.hypothesis_horizon_ms,
                deadline_offset_ms=request.deadline_offset_ms,
                raw_data_must_remain_local=not request.allow_raw_to_trusted_site_edge,
                allowed_node_ids=request.allowed_execution_node_ids,
                planning_policy_id=planning_policy_id_for_baseline(
                    request.baseline_id
                ),
                max_seed_hypotheses=request.max_seed_hypotheses,
                seed_admission_strategy=request.seed_admission_strategy,
            )
            result = controller.submit_event(submission)
            response = LiveComplexEventResponse(
                request_message_id=request.message_id,
                request_id=request.request_id,
                accepted=result.accepted,
                # The redesigned controller admits the initial semantic
                # frontier immediately; WATCHING preserves the replay driver's
                # start barrier without claiming that sensor evidence arrived.
                status="WATCHING" if result.accepted else "REJECTED",
                hypothesis_ids=tuple(map(str, result.hypothesis_ids)),
                command_count=len(result.command_message_ids),
                seed_action="WATCH_REGISTERED" if result.accepted else "REJECTED",
                active_seed_hypothesis_count=len(result.hypothesis_ids),
                reason=result.reason,
            )
            transport.publish(
                live_request_response_topic(request.submitter_id),
                encode_model(response), qos=1, retain=False,
            )
        except Exception:
            LOGGER.exception("invalid evaluation compatibility request")

    def on_terminal(_topic: str, payload: bytes) -> None:
        try:
            event = decode_model(payload, TerminalComplexEvent)
            if event.request_id not in submitters:
                return
            progress = LiveComplexEventProgress(
                request_id=event.request_id,
                transition_status="COMPLETED",
                transition_reason="redesigned controller emitted terminal event",
                hypothesis_ids=(str(event.hypothesis_id),),
                terminal=True,
                terminal_lifecycles={str(event.hypothesis_id): "COMPLETED"},
                detections=(LiveComplexEventDetection(
                    hypothesis_id=str(event.hypothesis_id),
                    event_family=event.family_id,
                    event_start_time=event.event_time_window.start,
                    event_end_time=event.event_time_window.end,
                    emitted_at=event.emitted_at,
                    bindings=event.bindings,
                ),),
            )
            transport.publish(
                live_request_progress_topic(event.request_id),
                encode_model(progress), qos=1, retain=False,
            )
        except Exception:
            LOGGER.exception("invalid terminal event at evaluation compatibility boundary")

    def on_cancel(_topic: str, payload: bytes) -> None:
        try:
            request = decode_model(payload, LiveComplexEventCancelRequest)
            outcome = controller.checkpoints.cancel_task(
                request_id=request.request_id, reason=request.reason
            )
            controller.requests.pop(request.request_id, None)
            evaluation_requests.pop(request.request_id, None)
            response = LiveComplexEventCancelResponse(
                request_message_id=request.message_id,
                request_id=request.request_id,
                status="CANCELLED",
                cancelled_active_execution=True,
                cancelled_demand_count=len(outcome.cancelled_demand_ids),
                released_lease_count=len(outcome.released_lease_ids),
                reason=request.reason,
            )
            transport.publish(
                live_request_cancel_response_topic(request.submitter_id),
                encode_model(response), qos=1, retain=False,
            )
        except Exception:
            LOGGER.exception("invalid evaluation compatibility cancellation")

    def on_resource_change(_topic: str, payload: bytes) -> None:
        try:
            change = decode_model(payload, ResourceChange)
            demand_ids = tuple(sorted(
                {item.lease.demand_id for item in orchestrator.lifecycle.active_leases},
                key=str,
            ))
            disturbance_ack = None
            if change.resource_kind.value in {"COMPUTE", "GPU"}:
                if not change.target_id:
                    raise ValueError("compute resource change requires target_id")
                nominal_gpu = controller.deployment_view.base.node(
                    change.target_id
                ).capacity.gpu_memory_mb
                disturbance_ack = controller.apply_disturbance(
                    runtime_disturbance_for_resource_change(
                        change=change,
                        submitter_id="evaluation-compatibility-adapter",
                        nominal_gpu_memory_mb=nominal_gpu,
                    )
                )
                if not disturbance_ack.accepted:
                    raise RuntimeError(disturbance_ack.reason)
            elif demand_ids:
                controller.handle_replan(
                    "evaluation-resource-change", demand_ids,
                    f"{change.resource_kind.value}:{change.action}:{change.condition}",
                )
            ack = ResourceChangeAck(
                request_message_id=change.message_id,
                run_id=change.run_id,
                condition_epoch=change.condition_epoch,
                accepted=True,
                adaptation_status=(
                    "REPLANNED"
                    if disturbance_ack is not None and disturbance_ack.changed
                    else "REPLANNED" if demand_ids else "UNCHANGED"
                ),
                reason=(
                    disturbance_ack.reason
                    if disturbance_ack is not None
                    else "typed evaluation disturbance accepted"
                ),
            )
            transport.publish(
                resource_change_ack_topic(change.run_id, str(change.message_id)),
                encode_model(ack), qos=1, retain=False,
            )
        except Exception:
            LOGGER.exception("invalid evaluation resource change")

    transport.subscribe(live_request_topic(orchestrator.orchestrator_id), on_request, qos=1)
    transport.subscribe(live_request_cancel_topic(orchestrator.orchestrator_id), on_cancel, qos=1)
    transport.subscribe(terminal_event_filter(), on_terminal, qos=1)
    transport.subscribe(resource_change_topic(), on_resource_change, qos=1)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    state_dir = Path(os.environ.get("FABLE_STATE_DIR", "/var/lib/fable/orchestrator"))
    state_dir.mkdir(parents=True, exist_ok=True)
    orchestrator_id = os.environ.get("FABLE_ORCHESTRATOR_ID", "orchestrator")

    registry = ProviderRegistry.from_files(
        catalog_path=os.environ.get(
            "FABLE_PROVIDER_CATALOG", "/workspace/FABLE/providers/registry/catalog.yaml"
        ),
        data_types_path=os.environ.get(
            "FABLE_DATA_TYPES", "/workspace/FABLE/providers/registry/data_types.yaml"
        ),
    )
    deployment = load_deployment_graph(
        os.environ.get(
            "FABLE_DEPLOYMENT_CONFIG", "/workspace/replay/config/fable_deployment.yaml"
        )
    )
    resolver = ProviderRuntimeResolver.from_yaml(
        os.environ.get(
            "FABLE_RUNTIME_CONFIG",
            "/workspace/replay/config/fable_provider_runtimes.yaml",
        )
    )
    deployment_view = RuntimeDeploymentView(deployment)
    lifecycle = ProviderLifecycleManager(
        provider_registry=registry,
        capacity=CapacityLedger(deployment),
        idle_grace_ms=int(os.environ.get("FABLE_IDLE_GRACE_MS", "2000")),
        capacity_group_resolver=resolver.capacity_group,
    )
    scheduler = MultiTenantScheduler(lifecycle=lifecycle)

    transport = PahoMQTTTransport(
        host=os.environ.get("MQTT_HOST_IP", "mqtt"),
        port=int(os.environ.get("MQTT_PORT", "1883")),
        client_id=os.environ.get(
            "FABLE_MQTT_CLIENT_ID", f"fable-{orchestrator_id}"
        ),
        keepalive=int(os.environ.get("MQTT_KEEPALIVE", "60")),
    )
    messenger = ReliableMessenger(
        entity_id=orchestrator_id,
        transport=transport,
        outbox=SQLiteOutbox(state_dir / "mqtt-outbox.sqlite"),
        retry_interval=float(os.environ.get("FABLE_OUTBOX_RETRY_SEC", "1.0")),
    )
    store = MongoStateStore(
        os.environ.get("MONGODB_URI", "mongodb://fable-mongo:27017"),
        database=os.environ.get("MONGODB_DATABASE", "fable"),
    )

    orchestrator = DistributedOrchestrator(
        orchestrator_id=orchestrator_id,
        transport=transport,
        messenger=messenger,
        processed_ledger=SQLiteProcessedLedger(state_dir / "processed.sqlite"),
        store=store,
        scheduler=scheduler,
        lifecycle=lifecycle,
        runtime_resolver=resolver,
        heartbeat_monitor=HeartbeatMonitor(
            interval=timedelta(
                seconds=float(os.environ.get("FABLE_HEARTBEAT_INTERVAL_SEC", "1"))
            ),
            suspect_misses=int(os.environ.get("FABLE_HEARTBEAT_SUSPECT_MISSES", "10")),
            unavailable_misses=int(
                os.environ.get("FABLE_HEARTBEAT_UNAVAILABLE_MISSES", "30")
            ),
            recovery_confirmations=int(
                os.environ.get("FABLE_HEARTBEAT_RECOVERY_CONFIRMATIONS", "2")
            ),
        ),
        monitor_interval=float(os.environ.get("FABLE_MONITOR_INTERVAL_SEC", "1.0")),
    )
    execution_profile = ExecutionProfile(
        os.environ.get("FABLE_EXECUTION_PROFILE", "development").strip().lower()
    )
    # Keep the service boundary aligned with the evaluation/baseline registry
    # contract used by bundle generation and host-side planning.  The former
    # ``FABLE_STATIC_BASELINE_REGISTRY`` spelling is retained only as a
    # compatibility fallback for already-generated deployments.
    static_registry = os.environ.get(
        "FABLE_STATIC_PIPELINE_REGISTRY",
        os.environ.get(
            "FABLE_STATIC_BASELINE_REGISTRY",
            "/workspace/FABLE/evaluation/manifests/baselines/static_pipelines.yaml",
        ),
    )
    planning_policies = {
        BaselineId.B1_HANDWRITTEN_STATIC.value: LiveBaselinePlanningPolicy(
            BaselineId.B1_HANDWRITTEN_STATIC,
            static_registry_path=static_registry,
        ),
        BaselineId.B3_TASK_RESOURCE_ADAPTIVE.value: LiveBaselinePlanningPolicy(
            BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
            static_registry_path=static_registry,
        ),
        BaselineId.B4_GREEDY_FRONTIER.value: LiveBaselinePlanningPolicy(
            BaselineId.B4_GREEDY_FRONTIER,
            static_registry_path=static_registry,
        ),
    }
    artifact_config = os.environ.get("FABLE_ARTIFACT_CONFIG")
    artifact_catalog = (
        load_deployment_artifacts(
            artifact_config,
            repository_root=os.environ.get(
                "FABLE_REPOSITORY_ROOT", "/workspace/FABLE"
            ),
        )
        if artifact_config
        else ArtifactCatalog()
    )
    controller = FableController(
        orchestrator=orchestrator,
        provider_registry=registry,
        deployment_view=deployment_view,
        artifact_catalog=artifact_catalog,
        execution_profile=execution_profile,
        network_telemetry=NetWaggleTelemetrySource(),
        retrospective_policy_id=os.environ.get(
            "FABLE_RETROSPECTIVE_POLICY", "R2_FABLE_TYPED_REPLAY"
        ),
        planning_policies=planning_policies,
    )
    controller.bind()
    bind_evaluation_compatibility(controller, orchestrator)

    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_args: stop.set())

    orchestrator.start()
    if not transport.wait_connected(
        timeout=float(os.environ.get("FABLE_MQTT_CONNECT_TIMEOUT_SEC", "15"))
    ):
        raise RuntimeError("orchestrator could not connect to MQTT")
    report = orchestrator.reconcile(
        require_heartbeats=env_bool("FABLE_REQUIRE_HEARTBEAT_ON_RESTART", False)
    )
    LOGGER.info("restart reconciliation: %s", report.model_dump(mode="json"))
    LOGGER.info("FABLE orchestrator ready id=%s execution_profile=%s", orchestrator_id, execution_profile.value)
    try:
        stop.wait()
    finally:
        orchestrator.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
