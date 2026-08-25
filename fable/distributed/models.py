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
    RuntimeLinkUpdate,
    RuntimeNodeUpdate,
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
    EVENT_REQUEST = "EVENT_REQUEST"
    EVENT_REQUEST_RESPONSE = "EVENT_REQUEST_RESPONSE"
    RUNTIME_DISTURBANCE_REQUEST = "RUNTIME_DISTURBANCE_REQUEST"
    RUNTIME_DISTURBANCE_ACK = "RUNTIME_DISTURBANCE_ACK"
    FAULT_COMMAND = "FAULT_COMMAND"


class AckStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    DUPLICATE = "DUPLICATE"
    REJECTED = "REJECTED"


class ExecutionProfile(StrEnum):
    """How strictly the deployed runtime treats physical implementations."""

    DEVELOPMENT = "development"
    PLUMBING = "plumbing"
    REAL = "real"


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
    IDENTITY_ASSOCIATION = "IDENTITY_ASSOCIATION"


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
    # Multiple logical providers may be capabilities of one warm physical worker.
    # ``worker_id`` identifies that shared process/container on this node; resource
    # accounting is then charged once using ``worker_resource_limits``.
    worker_id: str | None = None
    worker_resource_limits: ResourceLimits | None = None
    image: str | None = None
    container_name: str | None = None
    command: tuple[str, ...] = ()
    environment: dict[str, str] = Field(default_factory=dict)
    volumes: dict[str, str] = Field(default_factory=dict)
    network_mode: str | None = None
    working_dir: str | None = None
    entrypoint: tuple[str, ...] = ()
    labels: dict[str, str] = Field(default_factory=dict)
    # Stable device identifiers assigned by the evaluation bundle.  Keeping
    # these in the typed runtime contract lets agents enforce GPU placement
    # without passing an unvalidated Docker-specific environment fragment.
    gpu_device_ids: tuple[str, ...] = ()
    stop_adopted_when_idle: bool = False
    output_topics: tuple[str, ...] = ()
    # Explicit broker-backed intermediate dataflow. Keys are typed artifact
    # names and values are exact MQTT topics. These declarations are distinct
    # from ``output_topics`` above, which are observed by the node agent to
    # adapt a provider's terminal predicate result.
    artifact_topic_inputs: dict[str, str] = Field(default_factory=dict)
    artifact_topic_outputs: dict[str, str] = Field(default_factory=dict)
    # Cross-node topic transfer is permitted only for runtimes that explicitly
    # name the same broker/transport scope.
    artifact_broker_scope_id: str | None = None
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
    # Optional physical-plan provenance allows the distributed executor to release
    # downstream steps only when an upstream artifact really exists.
    plan_id: UUID | None = None
    step_id: str | None = None
    provider_instance_id: str | None = None
    lease_id: UUID | None = None
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


class ReplayReadiness(VersionedModel):
    """Typed replay-service readiness announcement.

    ``generation`` distinguishes a newly started process from retained or late
    readiness messages emitted by an older replay instance.
    """

    SCHEMA_VERSION: ClassVar[str] = "fable.replay_readiness.v1"
    schema_version: Literal["fable.replay_readiness.v1"] = SCHEMA_VERSION
    # Replay processes historically use UUID4; correlation requires uniqueness,
    # not the time-ordering guarantee imposed on durable controller commands.
    message_id: UUID = Field(default_factory=uuid7)
    node_id: str = Field(min_length=1)
    service_id: str = Field(min_length=1)
    process_instance_id: str = Field(min_length=1)
    generation: int = Field(ge=0)
    ready: bool
    reason: str = ""
    state: str = ""
    replay_id: str | None = None
    scenario: str | None = None
    observed_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @field_validator("observed_at")
    @classmethod
    def _normalize_observed_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class ResourceKind(StrEnum):
    NETWORK = "NETWORK"
    NETWORK_PROFILE = "NETWORK_PROFILE"
    LINK_STATE = "LINK_STATE"
    COMPUTE = "COMPUTE"
    GPU = "GPU"
    NODE = "NODE"


class ResourceChange(VersionedModel):
    """Evaluator-to-controller notification for one scoped disturbance epoch."""

    SCHEMA_VERSION: ClassVar[str] = "fable.resource_change.v1"
    schema_version: Literal["fable.resource_change.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    run_id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    action: str = Field(min_length=1)
    condition_epoch: int = Field(ge=0)
    target_id: str | None = None
    resource_kind: ResourceKind
    observed_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @field_validator("observed_at")
    @classmethod
    def _normalize_change_time(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _validate_link_target(self) -> Self:
        if self.resource_kind == ResourceKind.LINK_STATE:
            if not self.target_id or not self.target_id.startswith("link:"):
                raise ValueError("LINK_STATE target must be a canonical sensor link")
            parts = self.target_id.split(":")
            if len(parts) != 3 or not all(parts):
                raise ValueError("LINK_STATE target must be a canonical sensor link")
        return self


class ResourceChangeAck(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.resource_change_ack.v1"
    schema_version: Literal["fable.resource_change_ack.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    request_message_id: UUID
    run_id: str = Field(min_length=1)
    condition_epoch: int = Field(ge=0)
    accepted: bool
    adaptation_status: str = Field(min_length=1)
    reason: str = ""
    observed_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @field_validator("observed_at")
    @classmethod
    def _normalize_ack_time(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class ReliableNodeHeartbeat(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.reliable_node_heartbeat.v1"
    schema_version: Literal["fable.reliable_node_heartbeat.v1"] = SCHEMA_VERSION
    heartbeat: NodeHeartbeat


class EventRequestSubmission(VersionedModel):
    """Normal external API for starting a FABLE complex-event request.

    The caller supplies event semantics and request-level policy, not a physical
    execution plan.  The deployed controller owns compilation, frontier planning,
    admission, and replanning.
    """

    SCHEMA_VERSION: ClassVar[str] = "fable.event_request_submission.v1"
    schema_version: Literal["fable.event_request_submission.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    submitter_id: str = Field(min_length=1)
    request_id: str = Field(default_factory=lambda: str(uuid7()), min_length=1)
    # Evaluation-controlled static policies need the stable trace/placement
    # keys in order to enforce an authored physical contract.  They are audit
    # and policy-selection metadata; FABLE's normal planner ignores them.
    trace_id: str = ""
    baseline_placement_id: str = ""
    family_id: str = Field(min_length=1)
    parameters: dict[str, JSONValue] = Field(default_factory=dict)
    event_time_window: EventTimeInterval | None = None
    hypothesis_horizon_ms: int = Field(default=300_000, ge=1)
    deadline_offset_ms: int = Field(default=300_000, ge=1)
    raw_data_must_remain_local: bool = True
    allowed_node_ids: tuple[str, ...] = ()
    allowed_regions: tuple[str, ...] = ()
    maximum_transfer_bytes: int | None = Field(default=None, ge=0)
    planning_policy_id: str = Field(default="FABLE", min_length=1)
    # A source-discovery frontier may yield several independent bindings. The
    # deployed controller keeps that frontier leased until this bounded pool
    # is full; one camera result must not cancel every sibling camera watch.
    max_seed_hypotheses: int = Field(default=1, ge=1, le=32)
    seed_admission_strategy: Literal[
        "first_distinct", "reference_diverse", "reference_bounded"
    ] = "first_distinct"
    submitted_at: datetime = Field(default_factory=utc_now)

    @field_validator("submitted_at")
    @classmethod
    def _normalize_event_request_time(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class EventRequestResponse(VersionedModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.event_request_response.v1"
    schema_version: Literal["fable.event_request_response.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    request_message_id: UUID7
    request_id: str = Field(min_length=1)
    accepted: bool
    hypothesis_ids: tuple[UUID7, ...] = ()
    admitted_plan_ids: tuple[UUID7, ...] = ()
    command_message_ids: tuple[UUID7, ...] = ()
    reason: str = ""
    emitted_at: datetime = Field(default_factory=utc_now)

    @field_validator("emitted_at")
    @classmethod
    def _normalize_event_response_time(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class RuntimeDisturbanceRequest(VersionedModel):
    """Typed control-plane request for one E4/runtime operating disturbance."""

    SCHEMA_VERSION: ClassVar[str] = "fable.runtime_disturbance_request.v1"
    schema_version: Literal["fable.runtime_disturbance_request.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    submitter_id: str = Field(min_length=1)
    disturbance_id: str = Field(default_factory=lambda: str(uuid7()), min_length=1)
    node_updates: tuple[RuntimeNodeUpdate, ...] = ()
    link_updates: tuple[RuntimeLinkUpdate, ...] = ()
    reason: str = "evaluation/runtime disturbance"
    submitted_at: datetime = Field(default_factory=utc_now)

    @field_validator("submitted_at")
    @classmethod
    def _normalize_disturbance_time(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _require_updates(self) -> Self:
        if not self.node_updates and not self.link_updates:
            raise ValueError("runtime disturbance requires at least one node or link update")
        return self


class RuntimeDisturbanceAck(VersionedModel):
    """Acknowledges the authoritative deployment epoch after a disturbance."""

    SCHEMA_VERSION: ClassVar[str] = "fable.runtime_disturbance_ack.v1"
    schema_version: Literal["fable.runtime_disturbance_ack.v1"] = SCHEMA_VERSION
    message_id: UUID7 = Field(default_factory=uuid7)
    request_message_id: UUID7
    disturbance_id: str = Field(min_length=1)
    accepted: bool
    changed: bool = False
    previous_resource_epoch: int = Field(default=0, ge=0)
    resource_epoch: int = Field(default=0, ge=0)
    affected_demand_ids: tuple[UUID7, ...] = ()
    replanned_request_ids: tuple[str, ...] = ()
    reason: str = ""
    emitted_at: datetime = Field(default_factory=utc_now)

    @field_validator("emitted_at")
    @classmethod
    def _normalize_disturbance_ack_time(cls, value: datetime) -> datetime:
        return ensure_utc(value)


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
