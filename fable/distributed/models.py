"""Typed records for the Phase-6 distributed execution substrate.

The distributed layer transports and persists decisions made by the semantic,
planning, and scheduling layers.  It does not interpret complex-event truth.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Literal, Self
from uuid import UUID

from pydantic import Field, UUID7, field_validator, model_validator

from fable.common.base import FableModel, JSONValue, VersionedModel
from fable.common.ids import deterministic_id, uuid7
from fable.common.schemas import (
    ArtifactRef,
    NodeHeartbeat,
    PlanStep,
    PredicateDemand,
    PredicateResult,
    ProviderLease,
    ResourceReservation,
)
from fable.common.time import EventTimeInterval, ensure_utc, utc_now
from fable.scheduling.models import PlanCandidate


class MessageKind(StrEnum):
    ACTIVATE_PROVIDER = "ACTIVATE_PROVIDER"
    CANCEL_PROVIDER = "CANCEL_PROVIDER"
    APPLICATION_ACK = "APPLICATION_ACK"
    PROVIDER_STATUS = "PROVIDER_STATUS"
    PREDICATE_RESULT = "PREDICATE_RESULT"
    NODE_HEARTBEAT = "NODE_HEARTBEAT"
    ARTIFACT_ANNOUNCEMENT = "ARTIFACT_ANNOUNCEMENT"
    PLAN_DISPATCH_REQUEST = "PLAN_DISPATCH_REQUEST"
    PLAN_DISPATCH_RESPONSE = "PLAN_DISPATCH_RESPONSE"
    FAULT_COMMAND = "FAULT_COMMAND"


class AckStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    DUPLICATE = "DUPLICATE"
    REJECTED = "REJECTED"


class RuntimeMode(StrEnum):
    MANAGED_CONTAINER = "MANAGED_CONTAINER"
    ADOPT_EXISTING = "ADOPT_EXISTING"
    REFERENCE = "REFERENCE"


class ReplayOutputAdapter(StrEnum):
    """Built-in adapters for the existing replay stack's JSON MQTT outputs.

    These adapters are intentionally narrow.  They translate already-produced
    provider evidence into the typed ``PredicateResult`` envelope; they do not
    perform complex-event matching or invent new semantic predicates.
    """

    NONE = "NONE"
    AUDIO_DETECTION = "AUDIO_DETECTION"
    YOLO_OBJECT_PRESENT = "YOLO_OBJECT_PRESENT"
    VEHICLE_PREDICATE = "VEHICLE_PREDICATE"
    MULTIMODAL_PREDICATE = "MULTIMODAL_PREDICATE"


class AgentProviderStatus(StrEnum):
    STARTING = "STARTING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class ControlEventType(StrEnum):
    PLAN_DISPATCHED = "PLAN_DISPATCHED"
    COMMAND_SENT = "COMMAND_SENT"
    COMMAND_ACKED = "COMMAND_ACKED"
    PROVIDER_READY = "PROVIDER_READY"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    RESULT_RECEIVED = "RESULT_RECEIVED"
    RESULT_DUPLICATE = "RESULT_DUPLICATE"
    HEARTBEAT_RECEIVED = "HEARTBEAT_RECEIVED"
    NODE_SUSPECT = "NODE_SUSPECT"
    NODE_UNAVAILABLE = "NODE_UNAVAILABLE"
    NODE_RECOVERING = "NODE_RECOVERING"
    NODE_AVAILABLE = "NODE_AVAILABLE"
    ARTIFACT_REGISTERED = "ARTIFACT_REGISTERED"
    RECONCILIATION_COMPLETED = "RECONCILIATION_COMPLETED"
    CE_EMITTED = "CE_EMITTED"


class ResourceLimits(FableModel):
    cpu_cores: float = Field(default=0.0, ge=0)
    memory_mb: int = Field(default=0, ge=0)
    gpu_memory_mb: int = Field(default=0, ge=0)
    gpu_count: int = Field(default=0, ge=0)
    pids_limit: int | None = Field(default=None, ge=1)

    @classmethod
    def from_reservation(cls, reservation: ResourceReservation) -> "ResourceLimits":
        return cls(
            cpu_cores=reservation.cpu_cores,
            memory_mb=reservation.memory_mb,
            gpu_memory_mb=reservation.gpu_memory_mb,
            gpu_count=1 if reservation.gpu_memory_mb > 0 else 0,
        )


class ReadinessProbe(FableModel):
    mqtt_topic: str | None = None
    ready_field: str | None = None
    ready_value: JSONValue = True
    timeout_ms: int = Field(default=30_000, ge=0)
    container_health_required: bool = False


class ProviderRuntimeSpec(FableModel):
    """Node-executable provider realization.

    A node-local catalog may fill omitted image/container fields.  The command
    carries the resolved provider identity and resource limits so execution is
    auditable even when the node applies a local deployment override.
    """

    provider_id: str = Field(min_length=1)
    provider_contract_version: int = Field(ge=1)
    node_id: str = Field(min_length=1)
    mode: RuntimeMode
    image: str | None = None
    container_name: str | None = None
    command: tuple[str, ...] = ()
    environment: dict[str, str] = Field(default_factory=dict)
    volumes: dict[str, str] = Field(default_factory=dict)
    network_mode: str | None = None
    working_dir: str | None = None
    entrypoint: tuple[str, ...] = ()
    labels: dict[str, str] = Field(default_factory=dict)
    stop_adopted_when_idle: bool = False
    output_topics: tuple[str, ...] = ()
    output_adapter: ReplayOutputAdapter = ReplayOutputAdapter.NONE
    output_label_aliases: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    readiness: ReadinessProbe = Field(default_factory=ReadinessProbe)
    reference_delay_ms: int = Field(default=0, ge=0)
    reference_truth: bool = True
    reference_bindings: dict[str, str] = Field(default_factory=dict)
    reference_artifact_types: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_runtime(self) -> Self:
        if self.mode == RuntimeMode.MANAGED_CONTAINER and not self.image:
            raise ValueError("managed-container runtime requires image")
        if self.mode == RuntimeMode.ADOPT_EXISTING and not self.container_name:
            raise ValueError("adopt-existing runtime requires container_name")
        return self


class ActivateProviderCommand(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.activate_provider_command.v1"
    schema_version: Literal["fable.activate_provider_command.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    attempt_id: UUID7 = Field(default_factory=uuid7)
    sent_at: datetime = Field(default_factory=utc_now)
    orchestrator_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    provider_instance_id: str = Field(min_length=1)
    lease: ProviderLease
    demand: PredicateDemand
    plan_step: PlanStep
    runtime: ProviderRuntimeSpec
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    input_artifact_ids: tuple[UUID, ...] = ()
    result_topic: str = Field(min_length=1)
    artifact_topic: str = Field(min_length=1)
    provider_status_topic: str = Field(min_length=1)
    application_ack_topic: str = Field(min_length=1)
    issued_hypothesis_version: int = Field(ge=0)
    occurrence_id_hint: str | None = None

    @field_validator("sent_at")
    @classmethod
    def _normalize_sent_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _validate_alignment(self) -> Self:
        if self.lease.node_id != self.node_id or self.runtime.node_id != self.node_id:
            raise ValueError("activation node IDs must match")
        if self.lease.provider_instance_id != self.provider_instance_id:
            raise ValueError("lease provider instance does not match command")
        if self.lease.provider_id != self.runtime.provider_id:
            raise ValueError("lease and runtime provider IDs must match")
        if self.lease.demand_id != self.demand.demand_id:
            raise ValueError("lease and demand IDs must match")
        if self.issued_hypothesis_version != self.demand.hypothesis_version:
            raise ValueError("issued hypothesis version must match demand")
        return self

    @property
    def idempotency_key(self) -> str:
        return deterministic_id(
            "activate",
            {
                "demand_id": self.demand.demand_id,
                "attempt_id": self.attempt_id,
                "provider_instance_id": self.provider_instance_id,
            },
            length=32,
        )


class CancelProviderCommand(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.cancel_provider_command.v1"
    schema_version: Literal["fable.cancel_provider_command.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    attempt_id: UUID7 = Field(default_factory=uuid7)
    sent_at: datetime = Field(default_factory=utc_now)
    orchestrator_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    provider_instance_id: str = Field(min_length=1)
    lease_id: UUID7
    demand_id: UUID7
    reason: str = ""
    stop_if_idle: bool = True
    force_after_ms: int = Field(default=5_000, ge=0)
    application_ack_topic: str = Field(min_length=1)

    @field_validator("sent_at")
    @classmethod
    def _normalize_sent_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class ApplicationAck(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.application_ack.v1"
    schema_version: Literal["fable.application_ack.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    acked_message_id: UUID7
    receiver_id: str = Field(min_length=1)
    sender_id: str = Field(min_length=1)
    status: AckStatus
    reason: str = ""
    received_at: datetime = Field(default_factory=utc_now)

    @field_validator("received_at")
    @classmethod
    def _normalize_received_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class ProviderStatusEvent(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.provider_status_event.v1"
    schema_version: Literal["fable.provider_status_event.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    node_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    provider_instance_id: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    status: AgentProviderStatus
    active_lease_ids: tuple[UUID7, ...] = ()
    container_id: str | None = None
    adopted: bool = False
    reason: str = ""
    emitted_at: datetime = Field(default_factory=utc_now)

    @field_validator("emitted_at")
    @classmethod
    def _normalize_emitted_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class ArtifactAnnouncement(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.artifact_announcement.v1"
    schema_version: Literal["fable.artifact_announcement.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    node_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    artifact: ArtifactRef
    durable_local_write: bool = True
    emitted_at: datetime = Field(default_factory=utc_now)

    @field_validator("emitted_at")
    @classmethod
    def _normalize_emitted_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class ReliablePredicateResult(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.reliable_predicate_result.v1"
    schema_version: Literal["fable.reliable_predicate_result.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    node_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    provider_instance_id: str = Field(min_length=1)
    attempt_id: UUID7
    result: PredicateResult
    emitted_at: datetime = Field(default_factory=utc_now)

    @field_validator("emitted_at")
    @classmethod
    def _normalize_emitted_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class ReliableNodeHeartbeat(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.reliable_node_heartbeat.v1"
    schema_version: Literal["fable.reliable_node_heartbeat.v1"] = SCHEMA_VERSION
    heartbeat: NodeHeartbeat


class PlanDispatchRequest(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.plan_dispatch_request.v1"
    schema_version: Literal["fable.plan_dispatch_request.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    submitter_id: str = Field(min_length=1)
    candidates: tuple[PlanCandidate, ...]
    runtime_overrides: dict[str, ProviderRuntimeSpec] = Field(default_factory=dict)
    submitted_at: datetime = Field(default_factory=utc_now)

    @field_validator("submitted_at")
    @classmethod
    def _normalize_submitted_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _require_candidates(self) -> Self:
        if not self.candidates:
            raise ValueError("dispatch request requires at least one candidate")
        return self


class PlanDispatchResponse(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.plan_dispatch_response.v1"
    schema_version: Literal["fable.plan_dispatch_response.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    request_message_id: UUID7
    admitted_plan_ids: tuple[UUID7, ...] = ()
    deferred_candidate_ids: tuple[str, ...] = ()
    rejected_candidate_ids: tuple[str, ...] = ()
    command_message_ids: tuple[UUID7, ...] = ()
    reason: str = ""
    emitted_at: datetime = Field(default_factory=utc_now)

    @field_validator("emitted_at")
    @classmethod
    def _normalize_emitted_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class ControlEvent(FableModel):
    event_id: UUID7 = Field(default_factory=uuid7)
    event_type: ControlEventType
    entity_type: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    request_id: str | None = None
    hypothesis_id: UUID | None = None
    node_id: str | None = None
    payload: dict[str, JSONValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def _normalize_created_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class PersistedRuntimeState(FableModel):
    provider_instances: tuple[dict[str, JSONValue], ...] = ()
    managed_leases: tuple[dict[str, JSONValue], ...] = ()
    managed_plans: tuple[dict[str, JSONValue], ...] = ()
    node_heartbeats: tuple[NodeHeartbeat, ...] = ()
    demand_ids: tuple[UUID, ...] = ()
    result_ids: tuple[UUID, ...] = ()


class ReconciliationReport(FableModel):
    restored_provider_instance_ids: tuple[str, ...] = ()
    restored_lease_ids: tuple[UUID, ...] = ()
    restored_plan_ids: tuple[UUID, ...] = ()
    failed_provider_instance_ids: tuple[str, ...] = ()
    orphan_agent_provider_instance_ids: tuple[str, ...] = ()
    stale_lease_ids: tuple[UUID, ...] = ()
    duplicate_event_ids_suppressed: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class SegmentRef(FableModel):
    segment_id: str | None = None
    source_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    event_time_interval: EventTimeInterval
    bytes: int = Field(default=0, ge=0)
    checksum: str | None = None
    media_type: str = "application/octet-stream"
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @field_validator("created_at", "expires_at")
    @classmethod
    def _normalize_segment_times(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    @model_validator(mode="after")
    def _derive_segment_id(self) -> Self:
        expected = deterministic_id(
            "segment",
            {
                "source_id": self.source_id,
                "path": self.path,
                "event_time_interval": self.event_time_interval,
            },
            length=32,
        )
        if self.segment_id is None:
            self.segment_id = expected
        elif self.segment_id != expected:
            raise ValueError("segment_id does not match segment identity")
        return self


class FaultKind(StrEnum):
    DUPLICATE_NEXT_MESSAGE = "DUPLICATE_NEXT_MESSAGE"
    DROP_OUTBOUND = "DROP_OUTBOUND"
    CRASH_PROVIDER = "CRASH_PROVIDER"
    PAUSE_HEARTBEATS = "PAUSE_HEARTBEATS"
    EXPIRE_ARTIFACT = "EXPIRE_ARTIFACT"
    RESTART_ORCHESTRATOR = "RESTART_ORCHESTRATOR"


class FaultCommand(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.fault_command.v1"
    schema_version: Literal["fable.fault_command.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    target_id: str = Field(min_length=1)
    kind: FaultKind
    provider_instance_id: str | None = None
    artifact_id: UUID | None = None
    count: int = Field(default=1, ge=1)
    duration_ms: int = Field(default=0, ge=0)
    reason: str = "injected fault"
