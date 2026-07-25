"""Distributed node agent for provider lifecycle, results, artifacts, and heartbeats."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
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
from fable.common.ids import occurrence_anchor_id, uuid7
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


@dataclass
class AgentProviderRecord:
    provider_instance_id: str
    provider_id: str
    runtime: ProviderRuntimeSpec
    handle: ContainerHandle | None
    status: AgentProviderStatus
    active_leases: dict[UUID, ActivateProviderCommand] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    failure_reason: str = ""


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
        self.providers: dict[str, AgentProviderRecord] = {}
        self._activation_keys: dict[str, str] = {}
        self._forwarded_occurrences: set[tuple[UUID, str]] = set()
        self._heartbeat_sequence = 0
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
                handle: ContainerHandle | None
                limits = command.resource_limits
                if command.runtime.mode == RuntimeMode.MANAGED_CONTAINER:
                    handle = self.containers.start(
                        provider_instance_id=command.provider_instance_id,
                        spec=command.runtime,
                        limits=limits,
                    )
                elif command.runtime.mode == RuntimeMode.ADOPT_EXISTING:
                    handle = self.containers.adopt(
                        provider_instance_id=command.provider_instance_id,
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
                record = AgentProviderRecord(
                    provider_instance_id=command.provider_instance_id,
                    provider_id=command.runtime.provider_id,
                    runtime=command.runtime,
                    handle=handle,
                    status=(
                        AgentProviderStatus.READY
                        if ready_immediately
                        else AgentProviderStatus.STARTING
                    ),
                )
                self.providers[command.provider_instance_id] = record
                if command.runtime.readiness.mqtt_topic:
                    self.transport.subscribe(
                        command.runtime.readiness.mqtt_topic,
                        lambda topic, payload, provider_instance_id=command.provider_instance_id: self._on_readiness(
                            provider_instance_id, topic, payload
                        ),
                        qos=0,
                    )
                for output_topic in command.runtime.output_topics:
                    self.transport.subscribe(
                        output_topic,
                        lambda topic, payload, provider_instance_id=command.provider_instance_id: self._on_provider_output(
                            provider_instance_id, topic, payload
                        ),
                        qos=0,
                    )
            record.active_leases.setdefault(command.lease.lease_id, command)
            record.updated_at = utc_now()
            self._activation_keys[activation_key] = command.provider_instance_id
            status = record.status

        self._publish_provider_status_record(record)
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
            if record.runtime.mode == RuntimeMode.MANAGED_CONTAINER:
                stopped = self.containers.stop(
                    command.provider_instance_id,
                    timeout_seconds=max(1, command.force_after_ms // 1000),
                )
            elif (
                record.runtime.mode == RuntimeMode.ADOPT_EXISTING
                and record.runtime.stop_adopted_when_idle
            ):
                stopped = self.containers.stop(
                    command.provider_instance_id,
                    timeout_seconds=max(1, command.force_after_ms // 1000),
                )
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
            record.status = AgentProviderStatus.READY
            record.updated_at = utc_now()
            commands = tuple(record.active_leases.values())
        self._publish_provider_status_record(record)
        # A reference runtime is immediate and never reaches this callback.
        # Managed providers publish their own typed results after readiness.
        for command in commands:
            if command.runtime.mode == RuntimeMode.REFERENCE:
                self._execute_reference(command)

    def _execute_reference(self, command: ActivateProviderCommand) -> None:
        def run() -> None:
            started = utc_now()
            if command.runtime.reference_delay_ms:
                time.sleep(command.runtime.reference_delay_ms / 1000.0)
            completed = utc_now()
            source_id = (
                command.demand.eligible_source_ids[0]
                if command.demand.eligible_source_ids
                else self.node_id
            )
            introduced = {
                role: entity
                for role, entity in command.runtime.reference_bindings.items()
                if role in command.demand.unbound_roles
            }
            validated = {
                role: entity
                for role, entity in command.demand.bound_roles.items()
                if role in command.runtime.reference_bindings
                and command.runtime.reference_bindings[role] == entity
            }
            occurrence = command.occurrence_id_hint or occurrence_anchor_id(
                source_id,
                command.demand.semantic_predicate.predicate_id,
                command.demand.event_time_interval.start,
                {**command.demand.bound_roles, **introduced},
            )
            artifacts = self._create_reference_artifacts(command)
            result = PredicateResult(
                occurrence_id=occurrence,
                demand_id=command.demand.demand_id,
                request_id=command.demand.request_id,
                graph_hash=command.demand.graph_hash,
                hypothesis_id=command.demand.hypothesis_id,
                expected_hypothesis_version=command.issued_hypothesis_version,
                frontier_id=command.demand.frontier_id,
                checkpoint_id=command.demand.checkpoint_id,
                graph_node_id=command.demand.graph_node_id,
                semantic_predicate=command.demand.semantic_predicate,
                truth=(
                    TruthValue.TRUE
                    if command.runtime.reference_truth
                    else TruthValue.FALSE
                ),
                confidence=1.0,
                event_time_interval=command.demand.event_time_interval,
                binding_delta=BindingDelta(
                    introduced=introduced,
                    validated=validated,
                ),
                artifact_ids=tuple(item.artifact_id for item in artifacts),
                provenance=ResultProvenance(
                    provider_id=command.runtime.provider_id,
                    provider_contract_version=command.runtime.provider_contract_version,
                    node_id=self.node_id,
                    source_ids=(source_id,),
                ),
                processing_started_at=started,
                processing_completed_at=completed,
            )
            self.forward_result(
                result=result,
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
            if record is None or record.status not in (
                AgentProviderStatus.READY,
                AgentProviderStatus.ACTIVE,
            ):
                return
            commands = tuple(record.active_leases.values())
            adapter = record.runtime.output_adapter
        if adapter == ReplayOutputAdapter.NONE:
            return
        try:
            document = json.loads(payload.decode("utf-8"))
        except Exception:
            LOGGER.debug("ignored non-JSON provider output topic=%s", topic)
            return

        for command in commands:
            adapted = self._adapt_provider_output(command, adapter, document)
            if adapted is None:
                continue
            occurrence_id, event_interval, introduced, confidence = adapted
            dedup_key = (command.demand.demand_id, occurrence_id)
            with self._lock:
                if dedup_key in self._forwarded_occurrences:
                    continue
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
                    source_ids=command.demand.eligible_source_ids,
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

    def _adapt_provider_output(
        self,
        command: ActivateProviderCommand,
        adapter: ReplayOutputAdapter,
        document: Any,
    ) -> tuple[str, Any, dict[str, str], float] | None:
        demand = command.demand
        source_id = demand.eligible_source_ids[0] if demand.eligible_source_ids else self.node_id
        aliases = command.runtime.output_label_aliases

        if adapter == ReplayOutputAdapter.AUDIO_DETECTION:
            if not isinstance(document, dict):
                return None
            event_name = str(document.get("event") or document.get("label") or "")
            requested = str(demand.semantic_predicate.parameters.get("label") or "any")
            accepted = {event_name, *aliases.get(event_name, ())}
            if requested not in ("any", "*") and requested not in accepted:
                return None
            timestamp = _payload_event_time(document.get("t"))
            interval = _instant_interval(timestamp)
            if not demand.event_time_interval.overlaps(interval):
                return None
            confidence = float(document.get("confidence", 1.0))
            occurrence = occurrence_anchor_id(
                source_id,
                demand.semantic_predicate.predicate_id,
                timestamp,
                {**demand.bound_roles, "event": event_name},
            )
            return occurrence, interval, {}, max(0.0, min(1.0, confidence))

        if adapter == ReplayOutputAdapter.VEHICLE_PREDICATE:
            if not isinstance(document, dict):
                return None
            try:
                from providers.vehicle.models import PredicateObservation

                observation = PredicateObservation.model_validate(document)
            except Exception:
                LOGGER.debug("ignored invalid vehicle predicate payload")
                return None
            if observation.predicate_id != demand.semantic_predicate.predicate_id:
                return None
            if not demand.event_time_interval.overlaps(observation.event_time_interval):
                return None
            # Every already-bound semantic role must agree with the provider's
            # evidence. A provider-local result may introduce only roles that
            # are explicitly unbound in this demand.
            for role, entity_id in demand.bound_roles.items():
                if role in observation.bindings and observation.bindings[role] != entity_id:
                    return None
            introduced = {
                role: entity_id
                for role, entity_id in observation.bindings.items()
                if role in demand.unbound_roles
            }
            if demand.unbound_roles and not introduced:
                return None
            return (
                observation.occurrence_id,
                observation.event_time_interval,
                introduced,
                observation.confidence,
            )

        if adapter == ReplayOutputAdapter.MULTIMODAL_PREDICATE:
            if not isinstance(document, dict):
                return None
            schema_version = str(document.get("schema_version") or "")
            if schema_version == "audio_event_observation.v1":
                try:
                    from providers.multimodal.models import AudioEventObservation

                    observation = AudioEventObservation.model_validate(document)
                except Exception:
                    LOGGER.debug("ignored invalid typed audio-event payload")
                    return None
                if demand.semantic_predicate.predicate_id != "AUDIO_EVENT":
                    return None
                requested = str(demand.semantic_predicate.parameters.get("label") or "any")
                if requested not in ("any", "*") and requested != observation.label:
                    return None
                if not demand.event_time_interval.overlaps(observation.event_time_interval):
                    return None
                observed_location = observation.localized_zone_id or observation.source_id
                bound_location = demand.bound_roles.get("location")
                if bound_location is not None and bound_location != observed_location:
                    return None
                introduced = {}
                if "location" in demand.unbound_roles:
                    introduced["location"] = observed_location
                return (
                    observation.occurrence_id,
                    observation.event_time_interval,
                    introduced,
                    observation.confidence,
                )
            if schema_version == "interaction_predicate_observation.v1":
                try:
                    from providers.multimodal.models import InteractionPredicateObservation

                    observation = InteractionPredicateObservation.model_validate(document)
                except Exception:
                    LOGGER.debug("ignored invalid interaction predicate payload")
                    return None
                if observation.predicate_id != demand.semantic_predicate.predicate_id:
                    return None
                if not observation.truth:
                    return None
                if not demand.event_time_interval.overlaps(observation.event_time_interval):
                    return None
                for role, entity_id in demand.bound_roles.items():
                    if role in observation.bindings and observation.bindings[role] != entity_id:
                        return None
                introduced = {
                    role: entity_id
                    for role, entity_id in observation.bindings.items()
                    if role in demand.unbound_roles
                }
                if demand.unbound_roles and not introduced:
                    return None
                return (
                    observation.occurrence_id,
                    observation.event_time_interval,
                    introduced,
                    observation.confidence,
                )
            return None

        if adapter == ReplayOutputAdapter.YOLO_OBJECT_PRESENT:
            rows = document if isinstance(document, list) else [document]
            rows = [row for row in rows if isinstance(row, dict)]
            requested_raw = demand.semantic_predicate.parameters.get(
                "class_allowlist",
                demand.semantic_predicate.parameters.get("class", ()),
            )
            if isinstance(requested_raw, str):
                requested = {requested_raw}
            else:
                requested = {str(item) for item in requested_raw or ()}
            matching = [
                row
                for row in rows
                if not requested or str(row.get("class") or row.get("label")) in requested
            ]
            if not matching:
                return None
            row = max(matching, key=lambda item: float(item.get("conf", 0.0)))
            timestamp = _payload_event_time(row.get("t"))
            interval = _instant_interval(timestamp)
            if not demand.event_time_interval.overlaps(interval):
                return None
            object_label = str(row.get("class") or row.get("label") or "object")
            object_id = str(
                row.get("track_id")
                or row.get("id")
                or occurrence_anchor_id(
                    source_id,
                    f"object:{object_label}",
                    timestamp,
                    {"box": row.get("box", [])},
                )
            )
            introduced: dict[str, str] = {}
            if demand.unbound_roles:
                introduced[demand.unbound_roles[0]] = object_id
            occurrence = occurrence_anchor_id(
                source_id,
                demand.semantic_predicate.predicate_id,
                timestamp,
                {**demand.bound_roles, **introduced, "class": object_label},
            )
            return (
                occurrence,
                interval,
                introduced,
                max(0.0, min(1.0, float(row.get("conf", 1.0)))),
            )
        return None

    def _create_reference_artifacts(
        self, command: ActivateProviderCommand
    ) -> tuple[ArtifactRef, ...]:
        artifacts: list[ArtifactRef] = []
        for artifact_type in command.runtime.reference_artifact_types:
            artifact_id = uuid7()
            path = self.artifact_dir / f"{artifact_id}.json"
            path.write_text(
                json.dumps(
                    {
                        "provider_instance_id": command.provider_instance_id,
                        "demand_id": str(command.demand.demand_id),
                        "artifact_type": artifact_type,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            artifact = ArtifactRef(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                artifact_schema_version="reference.v1",
                producer=ArtifactProducer(
                    provider_id=command.runtime.provider_id,
                    provider_contract_version=command.runtime.provider_contract_version,
                ),
                event_time_interval=command.demand.event_time_interval,
                bindings={
                    **command.demand.bound_roles,
                    **command.runtime.reference_bindings,
                },
                location=ArtifactLocation(
                    kind=ArtifactLocationKind.LOCAL_PATH,
                    node_id=self.node_id,
                    uri=str(path),
                ),
                access_modes=(ArtifactAccessMode.LOCAL, ArtifactAccessMode.REMOTE_REFERENCE),
                compatible_consumer_families=tuple(
                    family
                    for requirement in command.demand.continuation_requirements
                    if requirement.artifact_type == artifact_type
                    for family in requirement.compatible_consumer_families
                ),
                bytes=path.stat().st_size,
                valid_until=command.demand.deadline.latest_useful_completion,
                expires_at=command.demand.deadline.latest_useful_completion,
            )
            artifacts.append(artifact)
            announcement = ArtifactAnnouncement(
                node_id=self.node_id,
                session_id=self.session_id,
                artifact=artifact,
            )
            self.messenger.send_model(
                command.artifact_topic,
                announcement,
                message_id=str(announcement.message_id),
                qos=1,
                require_application_ack=True,
            )
        return tuple(artifacts)

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
        with self._lock:
            self._heartbeat_sequence += 1
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


def _payload_event_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1e12:
            numeric /= 1e6
        return datetime.fromtimestamp(numeric, tz=UTC)
    if value is None:
        return utc_now()
    text = str(value).strip()
    try:
        numeric = float(text)
        if numeric > 1e12:
            numeric /= 1e6
        return datetime.fromtimestamp(numeric, tz=UTC)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        pass
    try:
        return datetime.strptime(text, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC)
    except ValueError:
        return utc_now()


def _instant_interval(timestamp: datetime):
    from fable.common.time import EventTimeInterval

    return EventTimeInterval(start=timestamp, end=timestamp)
