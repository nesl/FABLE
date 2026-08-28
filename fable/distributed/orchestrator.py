"""Distributed orchestrator transport, dispatch, persistence, and failure control."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime
import json
import logging
import threading
import time
from typing import Any
from uuid import UUID

from fable.common.enums import NodeAvailability, ProviderLeaseStatus
from fable.common.schemas import NodeHeartbeat, PredicateResult
from fable.common.time import ensure_utc, utc_now
from fable.scheduling.admission import MultiTenantScheduler
from fable.scheduling.lifecycle import ProviderLifecycleManager
from fable.scheduling.models import (
    AdmissionDecision,
    ManagedLease,
    PlanCandidate,
    ProviderInstanceLifecycle,
)

from .codec import decode_model, encode_model
from .config import ProviderRuntimeResolver
from .heartbeat import HeartbeatMonitor, NodeTransition
from .models import (
    AckStatus,
    ActivateProviderCommand,
    AgentProviderStatus,
    ApplicationAck,
    ArtifactAnnouncement,
    CancelProviderCommand,
    ControlEvent,
    ControlEventType,
    PlanDispatchRequest,
    PlanDispatchResponse,
    ProviderRuntimeSpec,
    ProviderStatusEvent,
    ReliablePredicateResult,
    ResourceLimits,
)
from .outbox import SQLiteProcessedLedger
from .plan_execution import PlanExecutionTracker
from .persistence import StateStore
from .reconciliation import EventEmissionLedger, RuntimeReconciler
from .topics import (
    ack_topic,
    activate_topic,
    artifact_filter,
    artifact_topic,
    cancel_topic,
    dispatch_request_topic,
    dispatch_response_topic,
    heartbeat_filter,
    provider_status_filter,
    result_filter,
    result_topic,
)
from .transport import ReliableMessenger, Transport

LOGGER = logging.getLogger(__name__)
ResultCallback = Callable[[PredicateResult], Any]
ReplanCallback = Callable[[str, tuple[UUID, ...], str], Any]
HeartbeatCallback = Callable[[NodeHeartbeat], Any]
ArtifactCallback = Callable[[ArtifactAnnouncement], Any]


class ExecutionDispatcher:
    """Converts admitted Phase-5 leases into node-agent commands."""

    def __init__(
        self,
        *,
        orchestrator_id: str,
        lifecycle: ProviderLifecycleManager,
        runtime_resolver: ProviderRuntimeResolver,
        messenger: ReliableMessenger,
        store: StateStore,
    ) -> None:
        self.orchestrator_id = orchestrator_id
        self.lifecycle = lifecycle
        self.runtime_resolver = runtime_resolver
        self.messenger = messenger
        self.store = store
        self.execution = PlanExecutionTracker(runtime_resolver)
        self._candidates: dict[UUID, PlanCandidate] = {}
        self._runtime_overrides: dict[UUID, dict[str, ProviderRuntimeSpec]] = {}

    def dispatch_candidate(
        self,
        candidate: PlanCandidate,
        *,
        runtime_overrides: dict[str, ProviderRuntimeSpec] | None = None,
    ) -> tuple[ActivateProviderCommand, ...]:
        """Dispatch only dependency-ready physical plan steps.

        Logical steps hosted by one physical worker are co-activated. Cross-worker
        dependencies remain pending until ``complete_step`` is called by a result or
        artifact announcement.
        """

        overrides = dict(runtime_overrides or {})
        # Registration snapshots the dependency DAG and returns only root steps;
        # downstream commands wait for result/artifact completion callbacks.
        self._candidates[candidate.plan.plan_id] = candidate
        self._runtime_overrides[candidate.plan.plan_id] = overrides
        ready_step_ids = self.execution.register(candidate)
        self._persist_execution_state(candidate.plan.plan_id)
        return self._dispatch_steps(candidate, ready_step_ids, overrides)

    def complete_step(self, plan_id: UUID, step_id: str) -> tuple[ActivateProviderCommand, ...]:
        """Mark a physical step complete and dispatch newly unblocked successors."""

        ready = self.execution.complete_step(plan_id, step_id)
        self._persist_execution_state(plan_id)
        candidate = self._candidates.get(plan_id)
        if candidate is None or not ready:
            return ()
        return self._dispatch_steps(
            candidate,
            ready,
            self._runtime_overrides.get(plan_id, {}),
        )

    def _dispatch_steps(
        self,
        candidate: PlanCandidate,
        step_ids: Iterable[str],
        runtime_overrides: dict[str, ProviderRuntimeSpec],
    ) -> tuple[ActivateProviderCommand, ...]:
        """Resolve runtimes and publish commands for dependency-ready steps."""

        selected = set(step_ids)
        # Lifecycle intents reconnect immutable plan steps to the leases created
        # during admission; dispatch never creates an unreserved execution.
        intents = tuple(
            intent
            for intent in self.lifecycle.preview_candidate(candidate)
            if intent.plan_step.step_id in selected
        )
        commands: list[ActivateProviderCommand] = []
        for intent in intents:
            managed = self._managed_lease(
                plan_id=candidate.plan.plan_id,
                demand_id=intent.demand_id,
                step_id=intent.plan_step.step_id,
            )
            instance = self.lifecycle.instances[managed.lease.provider_instance_id]
            override = runtime_overrides.get(
                f"{intent.plan_step.node_id}/{intent.plan_step.provider_id}"
            ) or runtime_overrides.get(intent.plan_step.provider_id)
            runtime = self.runtime_resolver.resolve(
                node_id=intent.plan_step.node_id,
                provider_id=intent.plan_step.provider_id,
                override=override,
            )
            demand = next(
                item for item in candidate.demands if item.demand_id == intent.demand_id
            )
            command = ActivateProviderCommand(
                attempt_id=managed.lease.attempt_id,
                orchestrator_id=self.orchestrator_id,
                node_id=intent.plan_step.node_id,
                provider_instance_id=managed.lease.provider_instance_id,
                lease=managed.lease,
                demand=demand,
                plan_step=intent.plan_step,
                runtime=runtime,
                resource_limits=ResourceLimits.from_reservation(instance.reservation),
                input_artifact_ids=intent.plan_step.input_artifact_ids,
                result_topic=result_topic(
                    demand.request_id, demand.semantic_predicate.predicate_id
                ),
                artifact_topic=artifact_topic(intent.plan_step.node_id),
                provider_status_topic=f"fable/v1/status/{intent.plan_step.node_id}/provider",
                application_ack_topic=ack_topic(self.orchestrator_id),
                issued_hypothesis_version=demand.hypothesis_version,
            )
            # Persist before publish so a fast acknowledgment/result can always
            # be correlated with durable command state.
            self.store.put("commands", str(command.message_id), command)
            self.messenger.send_model(
                activate_topic(command.node_id),
                command,
                message_id=str(command.message_id),
                qos=1,
                require_application_ack=True,
            )
            self.store.append_event(
                ControlEvent(
                    event_type=ControlEventType.COMMAND_SENT,
                    entity_type="activate_command",
                    entity_id=str(command.message_id),
                    request_id=demand.request_id,
                    hypothesis_id=demand.hypothesis_id,
                    node_id=command.node_id,
                    payload={
                        "provider_instance_id": command.provider_instance_id,
                        "provider_id": command.runtime.provider_id,
                        "lease_id": str(command.lease.lease_id),
                        "demand_id": str(command.demand.demand_id),
                        "step_id": command.plan_step.step_id,
                    },
                )
            )
            commands.append(command)
        return tuple(commands)

    def _persist_execution_state(self, plan_id: UUID) -> None:
        snapshot = self.execution.snapshot(plan_id)
        if snapshot is not None:
            self.store.put("plan_execution", str(plan_id), snapshot)

    def send_cancel(
        self,
        managed: ManagedLease,
        *,
        reason: str,
        stop_if_idle: bool = True,
    ) -> CancelProviderCommand:
        """Publish a lease-scoped cancellation to the owning node agent."""

        command = CancelProviderCommand(
            orchestrator_id=self.orchestrator_id,
            node_id=managed.lease.node_id,
            provider_instance_id=managed.lease.provider_instance_id,
            lease_id=managed.lease.lease_id,
            demand_id=managed.lease.demand_id,
            reason=reason,
            stop_if_idle=stop_if_idle,
            application_ack_topic=ack_topic(self.orchestrator_id),
        )
        self.store.put("commands", str(command.message_id), command)
        self.messenger.send_model(
            cancel_topic(command.node_id),
            command,
            message_id=str(command.message_id),
            qos=1,
            require_application_ack=True,
        )
        return command

    def send_identity_demand_cancel(
        self,
        *,
        request_id: str,
        demand_id: UUID,
        reason: str,
    ) -> None:
        """Cancel semantic identity work even after its physical lease ended.

        Identity comparisons can retain bounded crop/retry state after a short
        physical provider invocation has released its lease.  Consequently
        this control message is keyed by the semantic demand rather than by a
        provider instance.
        """

        payload = json.dumps(
            {
                "schema_version": "fable.identity_comparison_cancellation.v1",
                "request_id": request_id,
                "demand_id": str(demand_id),
                "reason": reason,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.messenger.transport.publish(
            "/fable/identity/cancellations",
            payload,
            qos=1,
            retain=False,
        )

    def persist_candidate_state(self, candidate: PlanCandidate) -> None:
        self.store.put("plans", str(candidate.plan.plan_id), self.lifecycle.plans[candidate.plan.plan_id])
        for demand in candidate.demands:
            self.store.put("demands", str(demand.demand_id), demand)
        # Heartbeat-driven replanning and result-driven cancellation can run
        # concurrently. Persist immutable snapshots rather than iterating the
        # live dictionaries while another callback mutates them.
        for instance in tuple(self.lifecycle.instances.values()):
            self.store.put("provider_instances", instance.provider_instance_id, instance)
        for lease in tuple(self.lifecycle.leases.values()):
            self.store.put("leases", str(lease.lease.lease_id), lease)

    def _managed_lease(
        self,
        *,
        plan_id: UUID,
        demand_id: UUID,
        step_id: str,
    ) -> ManagedLease:
        matches = [
            item
            for item in self.lifecycle.leases.values()
            if item.lease.plan_id == plan_id
            and item.lease.demand_id == demand_id
            and item.step_id == step_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one managed lease for plan={plan_id} demand={demand_id} "
                f"step={step_id}; found {len(matches)}"
            )
        return matches[0]


class DistributedOrchestrator:
    """Logically centralized semantic authority's distributed substrate.

    This class persists and transports plans, leases, provider state, results,
    artifacts, and node health.  A caller-provided result callback applies a
    ``PredicateResult`` to the Phase-1 semantic runtime; this class deliberately
    does not infer event truth itself.
    """

    def __init__(
        self,
        *,
        orchestrator_id: str,
        transport: Transport,
        messenger: ReliableMessenger,
        processed_ledger: SQLiteProcessedLedger,
        store: StateStore,
        scheduler: MultiTenantScheduler,
        lifecycle: ProviderLifecycleManager,
        runtime_resolver: ProviderRuntimeResolver,
        heartbeat_monitor: HeartbeatMonitor | None = None,
        on_result: ResultCallback | None = None,
        on_replan_required: ReplanCallback | None = None,
        on_heartbeat: HeartbeatCallback | None = None,
        on_artifact: ArtifactCallback | None = None,
        monitor_interval: float = 1.0,
    ) -> None:
        self.orchestrator_id = orchestrator_id
        self.transport = transport
        self.messenger = messenger
        self.processed = processed_ledger
        self.store = store
        self.scheduler = scheduler
        self.lifecycle = lifecycle
        self.runtime_resolver = runtime_resolver
        self.heartbeats = heartbeat_monitor or HeartbeatMonitor()
        self.on_result = on_result
        self.on_replan_required = on_replan_required
        self.on_heartbeat = on_heartbeat
        self.on_artifact = on_artifact
        self.monitor_interval = monitor_interval
        self.dispatcher = ExecutionDispatcher(
            orchestrator_id=orchestrator_id,
            lifecycle=lifecycle,
            runtime_resolver=runtime_resolver,
            messenger=messenger,
            store=store,
        )
        self.emissions = EventEmissionLedger(store)
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._started = False
        self.received_results: dict[UUID, PredicateResult] = {}
        self.replan_requests: list[tuple[str, tuple[UUID, ...], str]] = []

    def start(self) -> None:
        """Subscribe control/data callbacks before connecting the transport."""

        if self._started:
            return
        self._started = True
        self.transport.subscribe(ack_topic(self.orchestrator_id), self._on_ack, qos=1)
        self.transport.subscribe(result_filter(), self._on_result_message, qos=1)
        self.transport.subscribe(provider_status_filter(), self._on_provider_status, qos=1)
        self.transport.subscribe(artifact_filter(), self._on_artifact, qos=1)
        self.transport.subscribe(heartbeat_filter(), self._on_heartbeat, qos=0)
        self.transport.subscribe(
            dispatch_request_topic(self.orchestrator_id), self._on_dispatch_request, qos=1
        )
        self.transport.start()
        self.messenger.start_retry_loop()
        self._start_monitor_loop()

    def stop(self) -> None:
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=max(1.0, self.monitor_interval * 2))
        self.messenger.stop_retry_loop()
        self.transport.stop()
        self._started = False

    def submit_candidates(
        self,
        candidates: Iterable[PlanCandidate],
        *,
        runtime_overrides: dict[str, ProviderRuntimeSpec] | None = None,
        now: datetime | None = None,
    ) -> tuple[Any, tuple[ActivateProviderCommand, ...]]:
        """Admit candidates, persist admitted state, and dispatch ready roots."""

        # Materialize once because admission and candidate lookup must operate
        # over the exact same batch even when the caller supplied a generator.
        candidate_tuple = tuple(candidates)
        batch = self.scheduler.admit(candidate_tuple, now=now)
        by_id = {candidate.candidate_id: candidate for candidate in candidate_tuple}
        commands: list[ActivateProviderCommand] = []
        for record in batch.records:
            if record.decision != AdmissionDecision.ADMITTED:
                continue
            candidate = by_id[record.candidate_id]
            self.dispatcher.persist_candidate_state(candidate)
            # Dispatch may return fewer commands than plan steps because the
            # execution tracker honors inter-worker data dependencies.
            commands.extend(
                self.dispatcher.dispatch_candidate(
                    candidate,
                    runtime_overrides=runtime_overrides,
                )
            )
            self.store.append_event(
                ControlEvent(
                    event_type=ControlEventType.PLAN_DISPATCHED,
                    entity_type="execution_plan",
                    entity_id=str(candidate.plan.plan_id),
                    request_id=candidate.request_id,
                    payload={
                        "candidate_id": candidate.candidate_id or "",
                        "command_count": len(candidate.plan.steps),
                    },
                )
            )
        return batch, tuple(commands)

    def reconcile(self, *, require_heartbeats: bool = False):
        heartbeat_records = self.store.list("nodes", NodeHeartbeat)
        return RuntimeReconciler(store=self.store, lifecycle=self.lifecycle).restore(
            heartbeats=heartbeat_records,
            require_heartbeat_for_active_instances=require_heartbeats,
        )

    def emit_complex_event_once(self, event_key: str, payload: dict[str, Any]) -> bool:
        return self.emissions.claim(event_key, payload)

    def _on_dispatch_request(self, topic: str, payload: bytes) -> None:
        try:
            request = decode_model(payload, PlanDispatchRequest)
        except Exception:
            LOGGER.exception("invalid plan dispatch request")
            return
        existing = self.processed.get(str(request.message_id))
        if existing is not None and existing.response_payload is not None:
            self.transport.publish(
                dispatch_response_topic(request.submitter_id),
                existing.response_payload,
                qos=1,
                retain=False,
            )
            self._send_ack(
                target_id=request.submitter_id,
                acked_message_id=request.message_id,
                status=AckStatus.DUPLICATE,
                reason=existing.outcome,
            )
            return
        try:
            batch, commands = self.submit_candidates(
                request.candidates,
                runtime_overrides=request.runtime_overrides,
            )
            response = PlanDispatchResponse(
                request_message_id=request.message_id,
                admitted_plan_ids=batch.admitted_plan_ids,
                deferred_candidate_ids=tuple(
                    record.candidate_id
                    for record in batch.records
                    if record.decision == AdmissionDecision.DEFERRED
                ),
                rejected_candidate_ids=tuple(
                    record.candidate_id
                    for record in batch.records
                    if record.decision
                    in (AdmissionDecision.REJECTED, AdmissionDecision.EXPIRED)
                ),
                command_message_ids=tuple(command.message_id for command in commands),
            )
            wire = encode_model(response)
            self.processed.record(
                message_id=str(request.message_id),
                payload=payload,
                outcome="dispatch processed",
                response_payload=wire,
            )
            self.transport.publish(
                dispatch_response_topic(request.submitter_id), wire, qos=1, retain=False
            )
            self._send_ack(
                target_id=request.submitter_id,
                acked_message_id=request.message_id,
                status=AckStatus.ACCEPTED,
                reason="dispatch processed",
            )
        except Exception as exc:
            LOGGER.exception("plan dispatch failed")
            self.processed.record(
                message_id=str(request.message_id),
                payload=payload,
                outcome=f"dispatch failed: {exc}",
            )
            self._send_ack(
                target_id=request.submitter_id,
                acked_message_id=request.message_id,
                status=AckStatus.REJECTED,
                reason=f"dispatch failed: {exc}",
            )

    def _on_ack(self, topic: str, payload: bytes) -> None:
        try:
            ack = self.messenger.accept_ack(payload)
        except Exception:
            LOGGER.exception("invalid application acknowledgment")
            return
        self.store.append_event(
            ControlEvent(
                event_type=ControlEventType.COMMAND_ACKED,
                entity_type="message",
                entity_id=str(ack.acked_message_id),
                node_id=ack.sender_id,
                payload={"status": ack.status.value, "reason": ack.reason},
            )
        )

    def _on_result_message(self, topic: str, payload: bytes) -> None:
        started = time.monotonic()
        try:
            self._process_result_message(topic, payload)
        finally:
            elapsed_ms = (time.monotonic() - started) * 1000.0
            if elapsed_ms >= 250.0:
                LOGGER.warning(
                    "diagnostic slow result callback topic=%s elapsed_ms=%.1f payload_bytes=%d",
                    topic,
                    elapsed_ms,
                    len(payload),
                )

    def _process_result_message(self, topic: str, payload: bytes) -> None:
        """Durably accept one reliable result before invoking semantic feedback."""

        try:
            wrapper = decode_model(payload, ReliablePredicateResult)
        except Exception:
            LOGGER.exception("invalid reliable predicate result topic=%s", topic)
            return
        prior_message = self.processed.get(str(wrapper.message_id))
        if prior_message is not None:
            self._send_ack(
                target_id=wrapper.node_id,
                acked_message_id=wrapper.message_id,
                status=AckStatus.DUPLICATE,
                reason=prior_message.outcome,
            )
            return
        result_key = str(wrapper.result.result_id)
        # Message IDs protect transport retries; result IDs protect equivalent
        # evidence wrapped in a different reliable-delivery envelope.
        if self.store.contains("results", result_key):
            outcome = "duplicate result_id suppressed"
            self.processed.record(
                message_id=str(wrapper.message_id), payload=payload, outcome=outcome
            )
            self.store.append_event(
                ControlEvent(
                    event_type=ControlEventType.RESULT_DUPLICATE,
                    entity_type="predicate_result",
                    entity_id=result_key,
                    request_id=wrapper.result.request_id,
                    hypothesis_id=wrapper.result.hypothesis_id,
                    node_id=wrapper.node_id,
                )
            )
            self._send_ack(
                target_id=wrapper.node_id,
                acked_message_id=wrapper.message_id,
                status=AckStatus.DUPLICATE,
                reason=outcome,
            )
            return
        # Durability precedes acknowledgment and callbacks. A crash after this
        # point can replay downstream handling without asking the node to infer
        # the observation again.
        self.store.put("results", result_key, wrapper)
        self.received_results[wrapper.result.result_id] = wrapper.result
        outcome = "result durably accepted"
        self.processed.record(
            message_id=str(wrapper.message_id), payload=payload, outcome=outcome
        )
        self.store.append_event(
            ControlEvent(
                event_type=ControlEventType.RESULT_RECEIVED,
                entity_type="predicate_result",
                entity_id=result_key,
                request_id=wrapper.result.request_id,
                hypothesis_id=wrapper.result.hypothesis_id,
                node_id=wrapper.node_id,
                payload={
                    "demand_id": str(wrapper.result.demand_id),
                    "provider_instance_id": wrapper.provider_instance_id,
                },
            )
        )
        managed_matches = [
            item
            for item in self.lifecycle.leases.values()
            if item.lease.provider_instance_id == wrapper.provider_instance_id
            and item.lease.demand_id == wrapper.result.demand_id
        ]
        for managed in managed_matches:
            self.dispatcher.complete_step(managed.lease.plan_id, managed.step_id)
        if self.on_result is not None:
            self.on_result(wrapper.result)
        self._send_ack(
            target_id=wrapper.node_id,
            acked_message_id=wrapper.message_id,
            status=AckStatus.ACCEPTED,
            reason=outcome,
        )

    def _on_provider_status(self, topic: str, payload: bytes) -> None:
        try:
            event = decode_model(payload, ProviderStatusEvent)
        except Exception:
            LOGGER.exception("invalid provider status topic=%s", topic)
            return
        prior = self.processed.get(str(event.message_id))
        if prior is not None:
            self._send_ack(
                target_id=event.node_id,
                acked_message_id=event.message_id,
                status=AckStatus.DUPLICATE,
                reason=prior.outcome,
            )
            return
        outcome = f"provider status {event.status.value} accepted"
        instance = self.lifecycle.instances.get(event.provider_instance_id)
        if instance is not None:
            try:
                if event.status in (AgentProviderStatus.READY, AgentProviderStatus.ACTIVE):
                    if instance.lifecycle in (
                        ProviderInstanceLifecycle.WARMING,
                        ProviderInstanceLifecycle.ACTIVE,
                    ):
                        self.lifecycle.mark_ready(event.provider_instance_id)
                        self.store.append_event(
                            ControlEvent(
                                event_type=ControlEventType.PROVIDER_READY,
                                entity_type="provider_instance",
                                entity_id=event.provider_instance_id,
                                node_id=event.node_id,
                            )
                        )
                elif event.status == AgentProviderStatus.FAILED:
                    affected_lease_ids = tuple(instance.active_lease_ids)
                    affected = self.lifecycle.mark_failed(
                        event.provider_instance_id,
                        reason=event.reason or "node agent reported provider failure",
                    )
                    self._request_replan(
                        event.node_id,
                        affected,
                        event.reason or "provider failed",
                    )
                    self.store.append_event(
                        ControlEvent(
                            event_type=ControlEventType.PROVIDER_FAILED,
                            entity_type="provider_instance",
                            entity_id=event.provider_instance_id,
                            node_id=event.node_id,
                            payload={"affected_demand_ids": [str(item) for item in affected]},
                        )
                    )
                self.store.put("provider_instances", instance.provider_instance_id, instance)
                lease_ids_to_persist = set(instance.active_lease_ids)
                if event.status == AgentProviderStatus.FAILED:
                    lease_ids_to_persist.update(affected_lease_ids)
                for lease_id in lease_ids_to_persist:
                    managed = self.lifecycle.leases.get(lease_id)
                    if managed is not None:
                        self.store.put("leases", str(lease_id), managed)
            except Exception as exc:
                outcome = f"provider status recorded but lifecycle update failed: {exc}"
        self.store.put("provider_status_events", str(event.message_id), event)
        self.processed.record(
            message_id=str(event.message_id), payload=payload, outcome=outcome
        )
        self._send_ack(
            target_id=event.node_id,
            acked_message_id=event.message_id,
            status=AckStatus.ACCEPTED,
            reason=outcome,
        )

    def _on_artifact(self, topic: str, payload: bytes) -> None:
        try:
            announcement = decode_model(payload, ArtifactAnnouncement)
        except Exception:
            LOGGER.exception("invalid artifact announcement topic=%s", topic)
            return
        prior = self.processed.get(str(announcement.message_id))
        if prior is not None:
            self._send_ack(
                target_id=announcement.node_id,
                acked_message_id=announcement.message_id,
                status=AckStatus.DUPLICATE,
                reason=prior.outcome,
            )
            return
        self.store.put(
            "artifacts", str(announcement.artifact.artifact_id), announcement.artifact
        )
        outcome = "artifact metadata durably registered"
        self.processed.record(
            message_id=str(announcement.message_id), payload=payload, outcome=outcome
        )
        self.store.append_event(
            ControlEvent(
                event_type=ControlEventType.ARTIFACT_REGISTERED,
                entity_type="artifact",
                entity_id=str(announcement.artifact.artifact_id),
                node_id=announcement.node_id,
                payload={"artifact_type": announcement.artifact.artifact_type},
            )
        )
        if announcement.plan_id is not None and announcement.step_id is not None:
            self.dispatcher.complete_step(announcement.plan_id, announcement.step_id)
        if self.on_artifact is not None:
            self.on_artifact(announcement)
        self._send_ack(
            target_id=announcement.node_id,
            acked_message_id=announcement.message_id,
            status=AckStatus.ACCEPTED,
            reason=outcome,
        )

    def _on_heartbeat(self, topic: str, payload: bytes) -> None:
        """Overlay node telemetry, update capacity, and trigger transitions."""

        try:
            heartbeat = decode_model(payload, NodeHeartbeat)
        except Exception:
            LOGGER.exception("invalid node heartbeat topic=%s", topic)
            return
        received_at = utc_now()
        # Ingress lag is diagnostic only; availability hysteresis below remains
        # authoritative and prevents a delayed packet from immediately flapping.
        ingress_lag_ms = max(
            0.0,
            (received_at - ensure_utc(heartbeat.sent_at)).total_seconds() * 1000.0,
        )
        if ingress_lag_ms >= 2000.0:
            LOGGER.warning(
                "diagnostic delayed heartbeat node=%s sequence=%s sent_at=%s "
                "received_at=%s ingress_lag_ms=%.1f declared=%s",
                heartbeat.node_id,
                heartbeat.sequence,
                heartbeat.sent_at.isoformat(),
                received_at.isoformat(),
                ingress_lag_ms,
                heartbeat.availability.value,
            )
        if heartbeat.node_id not in self.lifecycle.capacity.deployment.nodes:
            LOGGER.debug(
                "ignoring heartbeat outside active logical deployment node=%s",
                heartbeat.node_id,
            )
            return
        transitions = self.heartbeats.record(heartbeat)
        # Persist effective hysteresis-derived availability while retaining the
        # node's original capacity/source telemetry fields.
        effective = heartbeat.model_copy(
            update={
                "availability": self.heartbeats.availability(heartbeat.node_id)
                or heartbeat.availability
            }
        )
        self.store.put("nodes", heartbeat.node_id, effective)
        try:
            # The scheduler maintains its own reservation ledger; heartbeat
            # free capacity is a runtime cap layered on top of those reservations.
            self.lifecycle.capacity.update_runtime_free(
                heartbeat.node_id,
                cpu_free_cores=heartbeat.capacity.cpu_free_cores,
                memory_free_mb=heartbeat.capacity.memory_free_mb,
                gpu_free_mb=heartbeat.capacity.gpu_free_mb,
                available=effective.availability == NodeAvailability.AVAILABLE,
            )
        except Exception:
            LOGGER.exception("failed to overlay heartbeat capacity node=%s", heartbeat.node_id)
        if self.on_heartbeat is not None:
            self.on_heartbeat(effective)
        self.store.append_event(
            ControlEvent(
                event_type=ControlEventType.HEARTBEAT_RECEIVED,
                entity_type="node",
                entity_id=heartbeat.node_id,
                node_id=heartbeat.node_id,
                payload={
                    "session_id": heartbeat.session_id,
                    "sequence": heartbeat.sequence,
                    "availability": effective.availability.value,
                },
            )
        )
        self._handle_node_transitions(transitions)

    def check_heartbeats(self, *, now: datetime | None = None) -> tuple[NodeTransition, ...]:
        transitions = self.heartbeats.tick(now=now)
        self._handle_node_transitions(transitions)
        return transitions

    def _handle_node_transitions(self, transitions: tuple[NodeTransition, ...]) -> None:
        for transition in transitions:
            heartbeat = self.heartbeats.heartbeat(transition.node_id)
            heartbeat_age_ms = (
                None
                if heartbeat is None
                else max(
                    0.0,
                    (
                        ensure_utc(transition.occurred_at)
                        - ensure_utc(heartbeat.sent_at)
                    ).total_seconds()
                    * 1000.0,
                )
            )
            LOGGER.warning(
                "diagnostic node transition node=%s previous=%s current=%s reason=%s "
                "occurred_at=%s last_heartbeat_sequence=%s last_heartbeat_sent_at=%s "
                "heartbeat_age_ms=%s",
                transition.node_id,
                None if transition.previous is None else transition.previous.value,
                transition.current.value,
                transition.reason,
                transition.occurred_at.isoformat(),
                None if heartbeat is None else heartbeat.sequence,
                None if heartbeat is None else heartbeat.sent_at.isoformat(),
                None if heartbeat_age_ms is None else round(heartbeat_age_ms, 1),
            )
            event_type = {
                NodeAvailability.SUSPECT: ControlEventType.NODE_SUSPECT,
                NodeAvailability.UNAVAILABLE: ControlEventType.NODE_UNAVAILABLE,
                NodeAvailability.RECOVERING: ControlEventType.NODE_RECOVERING,
                NodeAvailability.AVAILABLE: ControlEventType.NODE_AVAILABLE,
            }[transition.current]
            self.store.append_event(
                ControlEvent(
                    event_type=event_type,
                    entity_type="node",
                    entity_id=transition.node_id,
                    node_id=transition.node_id,
                    payload={
                        "previous": None
                        if transition.previous is None
                        else transition.previous.value,
                        "current": transition.current.value,
                        "reason": transition.reason,
                    },
                )
            )
            if heartbeat is not None:
                effective = heartbeat.model_copy(update={"availability": transition.current})
                self.store.put("nodes", transition.node_id, effective)
                try:
                    self.lifecycle.capacity.update_runtime_free(
                        transition.node_id,
                        cpu_free_cores=heartbeat.capacity.cpu_free_cores,
                        memory_free_mb=heartbeat.capacity.memory_free_mb,
                        gpu_free_mb=heartbeat.capacity.gpu_free_mb,
                        available=transition.current == NodeAvailability.AVAILABLE,
                    )
                except Exception:
                    LOGGER.exception("failed to update transition capacity node=%s", transition.node_id)
                if self.on_heartbeat is not None:
                    self.on_heartbeat(effective)
            if transition.current == NodeAvailability.UNAVAILABLE:
                affected: set[UUID] = set()
                for instance in tuple(self.lifecycle.active_instances):
                    if instance.share_key.node_id != transition.node_id:
                        continue
                    affected_lease_ids = tuple(instance.active_lease_ids)
                    affected.update(
                        self.lifecycle.mark_failed(
                            instance.provider_instance_id,
                            reason=transition.reason,
                            now=transition.occurred_at,
                        )
                    )
                    self.store.put(
                        "provider_instances", instance.provider_instance_id, instance
                    )
                    for lease_id in affected_lease_ids:
                        managed = self.lifecycle.leases.get(lease_id)
                        if managed is not None:
                            self.store.put("leases", str(lease_id), managed)
                self._request_replan(
                    transition.node_id,
                    tuple(sorted(affected, key=str)),
                    transition.reason,
                )

    def _request_replan(
        self,
        node_id: str,
        demand_ids: tuple[UUID, ...],
        reason: str,
    ) -> None:
        """Record and emit a physical replan request without changing semantics."""

        if not demand_ids:
            return
        request = (node_id, demand_ids, reason)
        self.replan_requests.append(request)
        if self.on_replan_required is not None:
            self.on_replan_required(*request)

    def _send_ack(
        self,
        *,
        target_id: str,
        acked_message_id: UUID,
        status: AckStatus,
        reason: str,
    ) -> None:
        ack = ApplicationAck(
            acked_message_id=acked_message_id,
            receiver_id=target_id,
            sender_id=self.orchestrator_id,
            status=status,
            reason=reason,
        )
        self.transport.publish(ack_topic(target_id), encode_model(ack), qos=1, retain=False)

    def _start_monitor_loop(self) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()

        def loop() -> None:
            while not self._stop_event.wait(self.monitor_interval):
                try:
                    self.check_heartbeats()
                    draining = self.lifecycle.tick()
                    for instance_id in draining:
                        instance = self.lifecycle.instances[instance_id]
                        self.store.put("provider_instances", instance_id, instance)
                except Exception:
                    LOGGER.exception("orchestrator monitor tick failed")

        self._monitor_thread = threading.Thread(
            target=loop,
            name=f"orchestrator-monitor-{self.orchestrator_id}",
            daemon=True,
        )
        self._monitor_thread.start()
