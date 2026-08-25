"""Distributed node agent for provider lifecycle, results, artifacts, and heartbeats."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
from uuid import UUID

from fable.common.enums import (
    ArtifactAccessMode,
    ArtifactLocationKind,
    NodeAvailability,
    TruthValue,
)
from fable.common.ids import uuid7
from fable.common.schemas import (
    ArtifactLocation,
    ArtifactProducer,
    ArtifactRef,
    BindingDelta,
    PredicateResult,
    ResultProvenance,
)
from fable.common.time import ensure_utc, utc_now

from .codec import decode_model, encode_model
from .docker_runtime import ContainerHandle, ContainerRuntime
from .heartbeat import CapacitySampler, ReplaySourceProgressTracker, build_node_heartbeat
from .models import (
    AckStatus,
    ActivateProviderCommand,
    AgentProviderStatus,
    ApplicationAck,
    ArtifactAnnouncement,
    CancelProviderCommand,
    FaultCommand,
    FaultKind,
    ProviderRuntimeSpec,
    ProviderStatusEvent,
    ReplayOutputAdapter,
    ReliablePredicateResult,
    ResourceLimits,
    RuntimeMode,
)
from .outbox import SQLiteProcessedLedger
from .output_adapters import (
    ProviderOutputAdapterRegistry,
    ReferenceExecutionContext,
    ReferenceRuntimeAdapter,
)
from .topics import (
    ack_topic,
    activate_topic,
    artifact_topic,
    cancel_topic,
    fault_topic,
    heartbeat_topic,
    provider_status_topic,
)
from .transport import ReliableMessenger, Transport

LOGGER = logging.getLogger(__name__)


def _identity_comparison_demand_payload(
    command: ActivateProviderCommand,
) -> bytes | None:
    """Build the exact-pair control message required by the identity worker.

    The distributed layer intentionally emits plain typed JSON here instead
    of importing a concrete provider package.  This keeps the generic node
    agent independent while preserving the provider's public demand schema.
    """

    if command.runtime.provider_id != "cross_sensor_identity_association":
        return None
    if command.demand.semantic_predicate.predicate_id != "SAME_ENTITY":
        return None
    left = command.demand.bound_roles.get("left")
    right = command.demand.bound_roles.get("right")
    if not left or not right:
        return None
    entity_kind = next(
        (
            role.entity_type
            for role in command.demand.semantic_predicate.roles
            if role.role_name == "left"
        ),
        "vehicle",
    )
    payload = {
        "request_id": command.demand.request_id,
        "demand_id": str(command.demand.demand_id),
        "left_local_entity_id": left,
        "right_local_entity_id": right,
        "entity_kind": entity_kind,
        "event_time_interval": command.demand.event_time_interval.model_dump(mode="json"),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


@dataclass
class AgentWorkerRecord:
    worker_id: str
    runtime: ProviderRuntimeSpec
    handle: ContainerHandle | None
    status: AgentProviderStatus
    logical_provider_instance_ids: set[str] = field(default_factory=set)


@dataclass
class AgentProviderRecord:
    provider_instance_id: str
    worker_id: str
    provider_id: str
    runtime: ProviderRuntimeSpec
    handle: ContainerHandle | None
    status: AgentProviderStatus
    active_leases: dict[UUID, ActivateProviderCommand] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    failure_reason: str = ""


@dataclass(frozen=True)
class BufferedProviderOutput:
    """One bounded provider-native observation retained for a later lease."""

    adapter: ReplayOutputAdapter
    topic: str
    document: Any
    fingerprint: str
    received_at: datetime


class NodeAgent:
    """Executes provider commands without owning complex-event semantics."""

    def __init__(
        self,
        *,
        node_id: str,
        session_id: str,
        transport: Transport,
        messenger: ReliableMessenger,
        processed_ledger: SQLiteProcessedLedger,
        container_runtime: ContainerRuntime,
        progress: ReplaySourceProgressTracker,
        state_dir: str | Path,
        heartbeat_interval: float = 1.0,
        capacity_sampler: CapacitySampler | None = None,
        allow_fault_injection: bool = False,
        output_adapters: ProviderOutputAdapterRegistry | None = None,
        reference_runtime: ReferenceRuntimeAdapter | None = None,
    ) -> None:
        self.node_id = node_id
        self.session_id = session_id
        self.transport = transport
        self.messenger = messenger
        self.processed = processed_ledger
        self.containers = container_runtime
        self.progress = progress
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = self.state_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_interval = heartbeat_interval
        self.capacity_sampler = capacity_sampler or CapacitySampler()
        self.allow_fault_injection = allow_fault_injection
        self.output_adapters = output_adapters or ProviderOutputAdapterRegistry()
        self.reference_runtime = reference_runtime
        self.providers: dict[str, AgentProviderRecord] = {}
        self.workers: dict[str, AgentWorkerRecord] = {}
        self._activation_keys: dict[str, str] = {}
        self._forwarded_occurrences: set[tuple[UUID, str]] = set()
        # Providers sharing one worker/topic can produce a fact before its
        # semantic frontier is active (for example PASSES is finalized after
        # the corresponding EXITS facts). Retain typed provider output locally
        # so a later bounded demand can adapt it without keeping all semantic
        # predicates leased or forwarding irrelevant observations upstream.
        self._provider_output_cache: deque[BufferedProviderOutput] = deque()
        self._provider_output_fingerprints: set[str] = set()
        self._provider_output_cache_limit = 32_768
        self._provider_output_retention = timedelta(minutes=5)
        self._heartbeat_sequence = 0
        self._last_heartbeat_provider_ids: tuple[str, ...] = ()
        self._last_heartbeat_demand_ids: tuple[UUID, ...] = ()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._pause_heartbeats_until: float = 0.0
        self._lock = threading.RLock()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.transport.subscribe(activate_topic(self.node_id), self._on_activate, qos=1)
        self.transport.subscribe(cancel_topic(self.node_id), self._on_cancel, qos=1)
        self.transport.subscribe(ack_topic(self.node_id), self._on_ack, qos=1)
        if self.allow_fault_injection:
            self.transport.subscribe(fault_topic(self.node_id), self._on_fault, qos=1)
        # Existing replay and detector topics.  These subscriptions only update
        # event-time/source health; semantic interpretation remains in providers.
        for topic_filter in (
            "/replay/status/#",
            "/+/analytics/yolo/bbox",
            "/debug/+/analytics/yolo/frame",
            "/+/audio_detector/detections",
        ):
            self.transport.subscribe(topic_filter, self._on_replay_progress, qos=0)
        self.transport.start()
        self.messenger.start_retry_loop()
        self._start_heartbeat_loop()

    def stop(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
        self.messenger.stop_retry_loop()
        self.transport.stop()
        self._started = False

    def _on_activate(self, topic: str, payload: bytes) -> None:
        try:
            command = decode_model(payload, ActivateProviderCommand)
        except Exception as exc:
            LOGGER.exception("invalid activation command node=%s", self.node_id)
            return
        if command.node_id != self.node_id:
            self._publish_ack(
                command.application_ack_topic,
                command.message_id,
                AckStatus.REJECTED,
                f"command targets node {command.node_id}, not {self.node_id}",
            )
            return
        existing = self.processed.get(str(command.message_id))
        if existing is not None:
            self._publish_ack(
                command.application_ack_topic,
                command.message_id,
                AckStatus.DUPLICATE,
                existing.outcome,
            )
            return
        try:
            status = self.activate(command)
            outcome = f"provider {command.provider_instance_id} {status.value}"
            self.processed.record(
                message_id=str(command.message_id),
                payload=payload,
                outcome=outcome,
            )
            self._publish_ack(
                command.application_ack_topic,
                command.message_id,
                AckStatus.ACCEPTED,
                outcome,
            )
        except Exception as exc:
            outcome = f"activation failed: {exc}"
            self.processed.record(
                message_id=str(command.message_id),
                payload=payload,
                outcome=outcome,
            )
            self._publish_ack(
                command.application_ack_topic,
                command.message_id,
                AckStatus.REJECTED,
                outcome,
            )
            self._publish_provider_status(
                provider_instance_id=command.provider_instance_id,
                provider_id=command.runtime.provider_id,
                status=AgentProviderStatus.FAILED,
                active_lease_ids=(command.lease.lease_id,),
                reason=outcome,
            )

    def activate(self, command: ActivateProviderCommand) -> AgentProviderStatus:
        """Idempotently attach a lease and ensure its provider is running."""
        activation_key = command.idempotency_key
        with self._lock:
            prior_instance = self._activation_keys.get(activation_key)
            if prior_instance is not None:
                record = self.providers[prior_instance]
                return record.status
            record = self.providers.get(command.provider_instance_id)
            if record is None:
                # Keep execution identity consistent with planner capacity
                # grouping.  A named container is one physical worker even
                # when a legacy runtime manifest omits the newer worker_id.
                worker_id = (
                    command.runtime.worker_id
                    or command.runtime.container_name
                    or command.provider_instance_id
                )
                worker = self.workers.get(worker_id)
                new_worker = worker is None
                # IDLE is a lease state, not a failed readiness state.  An
                # adopted worker remains alive after its last logical provider
                # lease is detached.  A later rolling hypothesis may create a
                # new logical provider instance on that same worker; revive the
                # worker before copying its status into the new record.  Without
                # this transition the new lease is born IDLE and every valid
                # output is silently ignored by _on_provider_output.
                if (
                    worker is not None
                    and worker.status == AgentProviderStatus.IDLE
                    and worker.handle is not None
                    and worker.handle.running
                ):
                    worker.status = AgentProviderStatus.READY
                if worker is None:
                    handle: ContainerHandle | None
                    limits = command.runtime.worker_resource_limits or command.resource_limits
                    if command.runtime.mode == RuntimeMode.MANAGED_CONTAINER:
                        handle = self.containers.start(
                            provider_instance_id=worker_id,
                            spec=command.runtime,
                            limits=limits,
                        )
                    elif command.runtime.mode == RuntimeMode.ADOPT_EXISTING:
                        handle = self.containers.adopt(
                            provider_instance_id=worker_id,
                            spec=command.runtime,
                        )
                    else:
                        handle = None
                    ready_immediately = (
                        command.runtime.mode == RuntimeMode.REFERENCE
                        or (
                            handle is not None
                            and handle.running
                            and not command.runtime.readiness.mqtt_topic
                            and (
                                not command.runtime.readiness.container_health_required
                                or handle.healthy is True
                            )
                        )
                    )
                    worker = AgentWorkerRecord(
                        worker_id=worker_id,
                        runtime=command.runtime,
                        handle=handle,
                        status=(
                            AgentProviderStatus.READY
                            if ready_immediately
                            else AgentProviderStatus.STARTING
                        ),
                    )
                    self.workers[worker_id] = worker
                record = AgentProviderRecord(
                    provider_instance_id=command.provider_instance_id,
                    worker_id=worker_id,
                    provider_id=command.runtime.provider_id,
                    runtime=command.runtime,
                    handle=worker.handle,
                    status=worker.status,
                )
                worker.logical_provider_instance_ids.add(command.provider_instance_id)
                self.providers[command.provider_instance_id] = record
                # MQTT subscriptions belong to the physical worker. Registering
                # the same result filter once per logical lease turns one
                # provider result into N callbacks; a callback that correctly
                # fans out across N sibling leases then becomes O(N^2). The
                # first logical record owns the single worker callback, which
                # evaluates every currently attached sibling lease.
                shared_identity_subscription = (
                    command.runtime.provider_id
                    == "cross_sensor_identity_association"
                )
                subscribe_logical_record = (
                    new_worker or not shared_identity_subscription
                )
                if subscribe_logical_record and command.runtime.readiness.mqtt_topic:
                    self.transport.subscribe(
                        command.runtime.readiness.mqtt_topic,
                        lambda topic, payload, provider_instance_id=command.provider_instance_id: self._on_readiness(
                            provider_instance_id, topic, payload
                        ),
                        qos=0,
                    )
                if subscribe_logical_record:
                    for output_topic in command.runtime.output_topics:
                        self.transport.subscribe(
                            output_topic,
                            lambda topic, payload, provider_instance_id=command.provider_instance_id: self._on_provider_output(
                                provider_instance_id, topic, payload
                            ),
                            qos=0,
                        )
            elif (
                record.status == AgentProviderStatus.IDLE
                and record.handle is not None
                and record.handle.running
            ):
                # A lifecycle manager may deliberately reuse a logical
                # provider instance after its prior lease was released.
                record.status = AgentProviderStatus.READY
                worker = self.workers.get(record.worker_id)
                if worker is not None:
                    worker.status = AgentProviderStatus.READY
            record.active_leases.setdefault(command.lease.lease_id, command)
            record.updated_at = utc_now()
            self._activation_keys[activation_key] = command.provider_instance_id
            status = record.status

        self._publish_provider_status_record(record)
        if command.runtime.output_adapter != ReplayOutputAdapter.NONE:
            self._replay_buffered_outputs(command)
        # A lease-controlled worker may not have subscribed to its control
        # topic until after container readiness. Publish immediately only for
        # runtimes which are already ready; STARTING runtimes are delivered by
        # _on_readiness below. This avoids both lost commands and duplicate
        # delivery when the worker was already connected.
        # Adopted workers remain ACTIVE while any prior lease is attached.
        # A later graph checkpoint can add another exact identity demand to
        # that same worker; ACTIVE is already command-ready and must receive
        # the new control message just like READY.
        if status in (AgentProviderStatus.READY, AgentProviderStatus.ACTIVE):
            self._publish_identity_comparison_demand(command)
        if command.runtime.mode == RuntimeMode.REFERENCE:
            self._execute_reference(command)
        return status

    def _on_cancel(self, topic: str, payload: bytes) -> None:
        try:
            command = decode_model(payload, CancelProviderCommand)
        except Exception:
            LOGGER.exception("invalid cancellation command node=%s", self.node_id)
            return
        if command.node_id != self.node_id:
            self._publish_ack(
                command.application_ack_topic,
                command.message_id,
                AckStatus.REJECTED,
                "wrong node",
            )
            return
        existing = self.processed.get(str(command.message_id))
        if existing is not None:
            self._publish_ack(
                command.application_ack_topic,
                command.message_id,
                AckStatus.DUPLICATE,
                existing.outcome,
            )
            return
        try:
            stopped = self.cancel(command)
            outcome = "provider stopped" if stopped else "lease detached; provider preserved"
            self.processed.record(
                message_id=str(command.message_id), payload=payload, outcome=outcome
            )
            self._publish_ack(
                command.application_ack_topic,
                command.message_id,
                AckStatus.ACCEPTED,
                outcome,
            )
        except Exception as exc:
            outcome = f"cancel failed: {exc}"
            self.processed.record(
                message_id=str(command.message_id), payload=payload, outcome=outcome
            )
            self._publish_ack(
                command.application_ack_topic,
                command.message_id,
                AckStatus.REJECTED,
                outcome,
            )

    def cancel(self, command: CancelProviderCommand) -> bool:
        with self._lock:
            record = self.providers.get(command.provider_instance_id)
            if record is None:
                return False
            record.active_leases.pop(command.lease_id, None)
            record.updated_at = utc_now()
            if record.active_leases or not command.stop_if_idle:
                self._publish_provider_status_record(record)
                return False
            stopped = False
            worker = self.workers.get(record.worker_id)
            worker_has_active_leases = any(
                provider.active_leases
                for provider in self.providers.values()
                if provider.worker_id == record.worker_id
                and provider.provider_instance_id != record.provider_instance_id
            )
            if worker is not None and not worker_has_active_leases:
                if record.runtime.mode == RuntimeMode.MANAGED_CONTAINER:
                    stopped = self.containers.stop(
                        record.worker_id,
                        timeout_seconds=max(1, command.force_after_ms // 1000),
                    )
                elif (
                    record.runtime.mode == RuntimeMode.ADOPT_EXISTING
                    and record.runtime.stop_adopted_when_idle
                ):
                    stopped = self.containers.stop(
                        record.worker_id,
                        timeout_seconds=max(1, command.force_after_ms // 1000),
                    )
                worker.status = AgentProviderStatus.STOPPED if stopped else AgentProviderStatus.IDLE
            record.status = AgentProviderStatus.STOPPED if stopped else AgentProviderStatus.IDLE
            self._publish_provider_status_record(record)
            return stopped

    def _on_readiness(self, provider_instance_id: str, topic: str, payload: bytes) -> None:
        with self._lock:
            record = self.providers.get(provider_instance_id)
            if record is None:
                return
            try:
                document = json.loads(payload.decode("utf-8"))
            except Exception:
                return
            probe = record.runtime.readiness
            ready = _readiness_matches(document, probe.ready_field, probe.ready_value)
            if not ready:
                return
            ready_at = utc_now()
            worker = self.workers.get(record.worker_id)
            if worker is not None:
                sibling_records = tuple(
                    item for item in self.providers.values() if item.worker_id == record.worker_id
                )
                transitioned = tuple(
                    sibling
                    for sibling in sibling_records
                    if sibling.status == AgentProviderStatus.STARTING
                )
                if worker.status == AgentProviderStatus.STARTING:
                    worker.status = AgentProviderStatus.READY
                for sibling in transitioned:
                    sibling.status = AgentProviderStatus.READY
                    sibling.updated_at = ready_at
            else:
                sibling_records = (record,)
                transitioned = (
                    (record,)
                    if record.status == AgentProviderStatus.STARTING
                    else ()
                )
                if transitioned:
                    record.status = AgentProviderStatus.READY
                    record.updated_at = ready_at
            # Readiness belongs to the physical worker, not merely to the
            # logical provider record whose subscription received this MQTT
            # message. Several rolling SAME_ENTITY checkpoints can attach
            # logical leases while their shared identity worker is STARTING.
            # Marking every sibling READY but publishing only ``record``'s
            # command strands all other exact-pair demands permanently: they
            # will neither receive another readiness transition nor be sent
            # by the immediate READY/ACTIVE activation path.
            #
            # Deliver every active sibling lease at this worker boundary.
            # A plan may share one demand across internal logical steps, so
            # deduplicate by demand ID within the readiness event.
            commands_by_demand = {}
            for sibling in transitioned:
                for command in sibling.active_leases.values():
                    commands_by_demand.setdefault(command.demand.demand_id, command)
            commands = tuple(commands_by_demand.values())
        # MQTT readiness is retained, so each new logical subscription can
        # receive the same ready document. Publishing status and control for
        # siblings which were already READY/ACTIVE creates an O(n^2) feedback
        # storm as rolling hypotheses accumulate. Only real state transitions
        # have work to release.
        for sibling in transitioned:
            self._publish_provider_status_record(sibling)
        # A reference runtime is immediate and never reaches this callback.
        # Managed providers publish their own typed results after readiness.
        for command in commands:
            self._publish_identity_comparison_demand(command)
            if command.runtime.mode == RuntimeMode.REFERENCE:
                self._execute_reference(command)

    def _publish_identity_comparison_demand(
        self,
        command: ActivateProviderCommand,
    ) -> bool:
        payload = _identity_comparison_demand_payload(command)
        if payload is None:
            return False
        self.transport.publish(
            "/fable/identity/demands",
            payload,
            qos=1,
            retain=False,
        )
        return True

    def _execute_reference(self, command: ActivateProviderCommand) -> None:
        if self.reference_runtime is None:
            raise RuntimeError(
                "REFERENCE runtime requested without an injected reference-runtime adapter"
            )

        def is_active(candidate: ActivateProviderCommand) -> bool:
            with self._lock:
                record = self.providers.get(candidate.provider_instance_id)
                return (
                    record is not None
                    and candidate.lease.lease_id in record.active_leases
                )

        def run() -> None:
            outcome = self.reference_runtime.execute(
                command,
                ReferenceExecutionContext(
                    node_id=self.node_id,
                    artifact_dir=self.artifact_dir,
                    is_active=is_active,
                ),
            )
            if outcome is None or not is_active(command):
                return
            for artifact in outcome.artifacts:
                announcement = ArtifactAnnouncement(
                    node_id=self.node_id,
                    session_id=self.session_id,
                    artifact=artifact,
                    plan_id=command.lease.plan_id,
                    step_id=command.plan_step.step_id,
                    provider_instance_id=command.provider_instance_id,
                    lease_id=command.lease.lease_id,
                )
                self.messenger.send_model(
                    command.artifact_topic,
                    announcement,
                    message_id=str(announcement.message_id),
                    qos=1,
                    require_application_ack=True,
                )
            if outcome.result is not None:
                self.forward_result(
                    result=outcome.result,
                    provider_instance_id=command.provider_instance_id,
                    attempt_id=command.attempt_id,
                    topic=command.result_topic,
                )

        threading.Thread(
            target=run,
            name=f"reference-{command.provider_instance_id}",
            daemon=True,
        ).start()

    def _on_provider_output(
        self,
        provider_instance_id: str,
        topic: str,
        payload: bytes,
    ) -> None:
        """Translate supported replay-provider JSON into typed predicate results.

        The adapter only evaluates the currently attached predicate demands.  It
        therefore preserves the semantic/physical boundary: an adopted YOLO or
        audio container emits provider evidence, while the orchestrator remains
        responsible for graph progress and complex-event completion.
        """

        with self._lock:
            record = self.providers.get(provider_instance_id)
            if record is None:
                return
            worker = self.workers.get(record.worker_id)
            worker_is_live = (
                worker is not None
                and worker.handle is not None
                and worker.handle.running
                and worker.status
                not in (AgentProviderStatus.STOPPED, AgentProviderStatus.FAILED)
            )
            sibling_records = (
                tuple(
                    sibling
                    for sibling in self.providers.values()
                    if sibling.worker_id == record.worker_id
                    and sibling.provider_id
                    == "cross_sensor_identity_association"
                    and sibling.runtime.output_adapter
                    == record.runtime.output_adapter
                )
                if record.provider_id == "cross_sensor_identity_association"
                else (record,)
            )
            forwarding_records = tuple(
                sibling
                for sibling in sibling_records
                if sibling.status
                in (AgentProviderStatus.READY, AgentProviderStatus.ACTIVE)
            )
            # An adopted/shared worker can remain alive while no logical lease
            # is attached. Its output must still enter the bounded local cache
            # so a later graph frontier can recover it. Only active logical
            # leases may forward results upstream.
            if not forwarding_records and not worker_is_live:
                return
            command_owners = tuple(
                (sibling.provider_instance_id, command)
                for sibling in forwarding_records
                for command in sibling.active_leases.values()
            )
            adapter = record.runtime.output_adapter
        if adapter == ReplayOutputAdapter.NONE:
            return
        try:
            document = json.loads(payload.decode("utf-8"))
        except Exception:
            LOGGER.debug("ignored non-JSON provider output topic=%s", topic)
            return

        self._buffer_provider_output(adapter, topic, document)

        forwarded_count = 0
        for owner_provider_instance_id, command in command_owners:
            forwarded_count += int(bool(self._forward_provider_document(
                command=command,
                adapter=adapter,
                document=document,
                provider_instance_id=owner_provider_instance_id,
            )))
        if adapter == ReplayOutputAdapter.IDENTITY_ASSOCIATION:
            LOGGER.debug(
                "identity association dispatch worker=%s logical_leases=%d "
                "forwarded=%d associations=%d",
                record.worker_id,
                len(command_owners),
                forwarded_count,
                len(document.get("associations") or ())
                if isinstance(document, dict)
                else 0,
            )

    def _buffer_provider_output(
        self,
        adapter: ReplayOutputAdapter,
        topic: str,
        document: Any,
    ) -> None:
        """Retain bounded typed output even when it matches no current lease."""

        canonical = json.dumps(document, separators=(",", ":"), sort_keys=True)
        fingerprint = f"{adapter.value}:{topic}:{canonical}"
        now = utc_now()
        with self._lock:
            self._expire_provider_output_cache(now)
            if fingerprint in self._provider_output_fingerprints:
                return
            self._provider_output_cache.append(
                BufferedProviderOutput(
                    adapter=adapter,
                    topic=topic,
                    document=document,
                    fingerprint=fingerprint,
                    received_at=now,
                )
            )
            self._provider_output_fingerprints.add(fingerprint)
            while len(self._provider_output_cache) > self._provider_output_cache_limit:
                removed = self._provider_output_cache.popleft()
                self._provider_output_fingerprints.discard(removed.fingerprint)

    def _expire_provider_output_cache(self, now: datetime) -> None:
        cutoff = now - self._provider_output_retention
        while (
            self._provider_output_cache
            and self._provider_output_cache[0].received_at < cutoff
        ):
            removed = self._provider_output_cache.popleft()
            self._provider_output_fingerprints.discard(removed.fingerprint)

    def _replay_buffered_outputs(self, command: ActivateProviderCommand) -> None:
        """Project distinct retained occurrences up to a defensive bound.

        Bindings are not a valid deduplication key: temporal graphs may require
        several distinct PASSES or EXITS occurrences for the same entity. The
        provider's occurrence ID is authoritative, while ``_forwarded_occurrences``
        still suppresses exact redelivery for an individual demand.
        """

        now = utc_now()
        with self._lock:
            self._expire_provider_output_cache(now)
            buffered = tuple(
                item
                for item in self._provider_output_cache
                if item.adapter == command.runtime.output_adapter
            )
        replayed_occurrences: set[str] = set()
        for item in buffered:
            adapted = self._adapt_provider_output(
                command,
                item.adapter,
                item.document,
            )
            if adapted is None:
                continue
            occurrence_id = adapted[0]
            if occurrence_id in replayed_occurrences:
                continue
            replayed_occurrences.add(occurrence_id)
            self._forward_provider_document(
                command=command,
                adapter=item.adapter,
                document=item.document,
                provider_instance_id=command.provider_instance_id,
            )
            if len(replayed_occurrences) >= 32:
                LOGGER.warning(
                    "bounded buffered replay demand=%s after %d occurrences",
                    command.demand.demand_id,
                    len(replayed_occurrences),
                )
                break

    def _forward_provider_document(
        self,
        *,
        command: ActivateProviderCommand,
        adapter: ReplayOutputAdapter,
        document: Any,
        provider_instance_id: str,
    ) -> bool:
        adapted = self._adapt_provider_output(command, adapter, document)
        if adapted is None:
            return False
        occurrence_id, event_interval, introduced, confidence, source_ids = adapted
        dedup_key = (command.demand.demand_id, occurrence_id)
        with self._lock:
            if dedup_key in self._forwarded_occurrences:
                return False
            self._forwarded_occurrences.add(dedup_key)
        result = PredicateResult(
            occurrence_id=occurrence_id,
            demand_id=command.demand.demand_id,
            request_id=command.demand.request_id,
            graph_hash=command.demand.graph_hash,
            hypothesis_id=command.demand.hypothesis_id,
            expected_hypothesis_version=command.issued_hypothesis_version,
            frontier_id=command.demand.frontier_id,
            checkpoint_id=command.demand.checkpoint_id,
            graph_node_id=command.demand.graph_node_id,
            semantic_predicate=command.demand.semantic_predicate,
            truth=TruthValue.TRUE,
            confidence=confidence,
            event_time_interval=event_interval,
            binding_delta=BindingDelta(introduced=introduced),
            provenance=ResultProvenance(
                provider_id=command.runtime.provider_id,
                provider_contract_version=command.runtime.provider_contract_version,
                node_id=self.node_id,
                source_ids=source_ids or command.demand.eligible_source_ids,
            ),
            processing_started_at=utc_now(),
            processing_completed_at=utc_now(),
        )
        self.forward_result(
            result=result,
            provider_instance_id=provider_instance_id,
            attempt_id=command.attempt_id,
            topic=command.result_topic,
        )
        return True

    def _adapt_provider_output(
        self,
        command: ActivateProviderCommand,
        adapter: ReplayOutputAdapter,
        document: Any,
    ) -> tuple[str, Any, dict[str, str], float, tuple[str, ...]] | None:
        """Compatibility wrapper around the injected provider-output registry.

        ``NodeAgent`` no longer interprets vehicle/audio/vision schemas itself.
        Concrete replay/testbed adapters are registered by the composition root.
        """

        evidence = self.output_adapters.adapt(adapter, command, document)
        if evidence is None:
            return None
        return (
            evidence.occurrence_id,
            evidence.event_interval,
            evidence.introduced_bindings,
            evidence.confidence,
            evidence.source_ids,
        )

    def forward_result(
        self,
        *,
        result: PredicateResult,
        provider_instance_id: str,
        attempt_id: UUID,
        topic: str,
    ) -> ReliablePredicateResult:
        wrapper = ReliablePredicateResult(
            node_id=self.node_id,
            session_id=self.session_id,
            provider_instance_id=provider_instance_id,
            attempt_id=attempt_id,
            result=result,
        )
        self.messenger.send_model(
            topic,
            wrapper,
            message_id=str(wrapper.message_id),
            qos=1,
            require_application_ack=True,
        )
        return wrapper

    def announce_artifact(self, artifact: ArtifactRef) -> ArtifactAnnouncement:
        announcement = ArtifactAnnouncement(
            node_id=self.node_id,
            session_id=self.session_id,
            artifact=artifact,
        )
        self.messenger.send_model(
            artifact_topic(self.node_id),
            announcement,
            message_id=str(announcement.message_id),
            qos=1,
            require_application_ack=True,
        )
        return announcement

    def emit_heartbeat(self) -> Any:
        if time.monotonic() < self._pause_heartbeats_until:
            return None
        # Container adoption and inspection can hold the agent state lock for
        # several seconds while a batch of provider leases is attached.  Node
        # liveness must not disappear during that local critical section.  If
        # a fresh snapshot cannot be obtained promptly, publish the most recent
        # coherent snapshot; the following heartbeat will expose the new lease
        # set after activation completes.
        acquired = self._lock.acquire(timeout=min(0.05, self.heartbeat_interval / 4))
        try:
            self._heartbeat_sequence += 1
            if acquired:
                active_provider_ids = tuple(
                    sorted(
                        provider_id
                        for provider_id, record in self.providers.items()
                        if record.status
                        not in (AgentProviderStatus.STOPPED, AgentProviderStatus.FAILED)
                    )
                )
                active_demand_ids = tuple(
                    sorted(
                        {
                            command.demand.demand_id
                            for record in self.providers.values()
                            for command in record.active_leases.values()
                        },
                        key=str,
                    )
                )
                self._last_heartbeat_provider_ids = active_provider_ids
                self._last_heartbeat_demand_ids = active_demand_ids
            else:
                active_provider_ids = self._last_heartbeat_provider_ids
                active_demand_ids = self._last_heartbeat_demand_ids
        finally:
            if acquired:
                self._lock.release()
        heartbeat = build_node_heartbeat(
            node_id=self.node_id,
            session_id=self.session_id,
            sequence=self._heartbeat_sequence,
            sources=self.progress.sources,
            active_provider_instance_ids=active_provider_ids,
            active_demand_ids=active_demand_ids,
            capacity=self.capacity_sampler.sample(),
            availability=NodeAvailability.AVAILABLE,
        )
        self.transport.publish(
            heartbeat_topic(self.node_id),
            encode_model(heartbeat),
            qos=0,
            retain=False,
        )
        return heartbeat

    def _start_heartbeat_loop(self) -> None:
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_stop.clear()

        def loop() -> None:
            self.emit_heartbeat()
            while not self._heartbeat_stop.wait(self.heartbeat_interval):
                try:
                    self.emit_heartbeat()
                except Exception:
                    LOGGER.exception("heartbeat publication failed node=%s", self.node_id)

        self._heartbeat_thread = threading.Thread(
            target=loop,
            name=f"heartbeat-{self.node_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _on_ack(self, topic: str, payload: bytes) -> None:
        try:
            self.messenger.accept_ack(payload)
        except Exception:
            LOGGER.exception("invalid application ack node=%s", self.node_id)

    def _on_replay_progress(self, topic: str, payload: bytes) -> None:
        self.progress.update(topic, payload)

    def _on_fault(self, topic: str, payload: bytes) -> None:
        try:
            command = decode_model(payload, FaultCommand)
        except Exception:
            LOGGER.exception("invalid fault command node=%s", self.node_id)
            return
        if command.target_id != self.node_id:
            return
        if command.kind == FaultKind.CRASH_PROVIDER and command.provider_instance_id:
            self.containers.crash(command.provider_instance_id)
            with self._lock:
                record = self.providers.get(command.provider_instance_id)
                if record:
                    record.status = AgentProviderStatus.FAILED
                    record.failure_reason = command.reason
                    self._publish_provider_status_record(record)
        elif command.kind == FaultKind.PAUSE_HEARTBEATS:
            self._pause_heartbeats_until = time.monotonic() + command.duration_ms / 1000.0
        elif command.kind == FaultKind.DUPLICATE_NEXT_MESSAGE:
            broker = getattr(self.transport, "broker", None)
            if broker is not None:
                broker.duplicate_next += command.count
        elif command.kind == FaultKind.DROP_OUTBOUND:
            broker = getattr(self.transport, "broker", None)
            if broker is not None:
                broker.drop_next += command.count

    def _publish_ack(
        self,
        topic: str,
        acked_message_id: UUID,
        status: AckStatus,
        reason: str,
    ) -> ApplicationAck:
        target_id = topic.rstrip("/").split("/")[-1]
        ack = ApplicationAck(
            acked_message_id=acked_message_id,
            receiver_id=target_id,
            sender_id=self.node_id,
            status=status,
            reason=reason,
        )
        self.transport.publish(topic, encode_model(ack), qos=1, retain=False)
        return ack

    def _publish_provider_status_record(self, record: AgentProviderRecord) -> ProviderStatusEvent:
        return self._publish_provider_status(
            provider_instance_id=record.provider_instance_id,
            provider_id=record.provider_id,
            status=record.status,
            active_lease_ids=tuple(sorted(record.active_leases, key=str)),
            container_id=None if record.handle is None else record.handle.container_id,
            adopted=False if record.handle is None else record.handle.adopted,
            reason=record.failure_reason,
        )

    def _publish_provider_status(
        self,
        *,
        provider_instance_id: str,
        provider_id: str,
        status: AgentProviderStatus,
        active_lease_ids: tuple[UUID, ...],
        container_id: str | None = None,
        adopted: bool = False,
        reason: str = "",
    ) -> ProviderStatusEvent:
        event = ProviderStatusEvent(
            node_id=self.node_id,
            session_id=self.session_id,
            provider_instance_id=provider_instance_id,
            provider_id=provider_id,
            status=status,
            active_lease_ids=tuple(active_lease_ids),
            container_id=container_id,
            adopted=adopted,
            reason=reason,
        )
        self.messenger.send_model(
            provider_status_topic(self.node_id),
            event,
            message_id=str(event.message_id),
            qos=1,
            require_application_ack=True,
        )
        return event


def _readiness_matches(document: Any, field_path: str | None, ready_value: Any) -> bool:
    if field_path is None:
        if isinstance(document, dict):
            if "ready" in document:
                return document["ready"] == ready_value
            if "model_loaded" in document:
                return bool(document["model_loaded"]) == bool(ready_value)
        return bool(document) == bool(ready_value)
    current = document
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return current == ready_value
