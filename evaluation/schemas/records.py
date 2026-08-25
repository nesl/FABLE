"""Versioned records shared by every FABLE evaluation mode and baseline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import Field, field_validator

from fable.common.base import FableModel, JSONValue
from fable.common.time import ensure_utc, utc_now


class BaselineId(StrEnum):
    # Accuracy/evaluation policy identifiers retained as first-class values;
    # they are not aliases for the newer controlled-planning microbenchmarks.
    B0_PRODUCE_ALL = "B0_PRODUCE_ALL"
    B1_STATIC_WHOLE_EVENT = "B1_STATIC_WHOLE_EVENT"
    B2_FRONTIER_FIXED_REALIZATION = "B2_FRONTIER_FIXED_REALIZATION"
    B0_ALWAYS_ON = "B0_ALWAYS_ON"
    B1_HANDWRITTEN_STATIC = "B1_HANDWRITTEN_STATIC"
    B2_STATIC_WHOLE_EVENT = "B2_STATIC_WHOLE_EVENT"
    B3_TASK_RESOURCE_ADAPTIVE = "B3_TASK_RESOURCE_ADAPTIVE"
    B4_GREEDY_FRONTIER = "B4_GREEDY_FRONTIER"
    FABLE = "FABLE"
    FABLE_NO_SHARING = "FABLE_NO_SHARING"
    O1_EXHAUSTIVE_ORACLE = "O1_EXHAUSTIVE_ORACLE"
    C0_CHEAP_ONLY = "C0_CHEAP_ONLY"
    C1_STRONG_ONLY = "C1_STRONG_ONLY"
    C2_FIXED_CASCADE = "C2_FIXED_CASCADE"
    C3_FABLE_ESCALATION = "C3_FABLE_ESCALATION"
    C4_FABLE_NO_ESCALATION = "C4_FABLE_NO_ESCALATION"
    SPATIAL_BROADCAST = "SPATIAL_BROADCAST"
    SPATIAL_TOPOLOGY_SHORTLIST = "SPATIAL_TOPOLOGY_SHORTLIST"
    SPATIAL_RESOURCE_ONLY = "SPATIAL_RESOURCE_ONLY"
    SPATIAL_FABLE = "SPATIAL_FABLE"
    SPATIAL_ORACLE = "SPATIAL_ORACLE"


class EvaluationMode(StrEnum):
    COMMON_PERCEPTION = "COMMON_PERCEPTION"
    FULL_STACK = "FULL_STACK"


class EvaluationRecord(FableModel):
    SCHEMA_VERSION: ClassVar[str] = "fable.evaluation_record.v1"
    schema_version: str = SCHEMA_VERSION
    record_type: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    baseline_id: BaselineId
    trace_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    hypothesis_id: str | None = None
    sensor_id: str | None = None
    provider_id: str | None = None
    event_time: datetime
    monotonic_timestamp_ns: int = Field(ge=0)
    wall_timestamp: datetime = Field(default_factory=utc_now)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @field_validator("event_time", "wall_timestamp")
    @classmethod
    def _normalize_time(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class PredicateObservation(EvaluationRecord):
    record_type: Literal["predicate_observation"] = "predicate_observation"
    observation_id: str = Field(min_length=1)
    predicate_id: str = Field(min_length=1)
    event_end_time: datetime
    bindings: dict[str, str] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_refs: tuple[str, ...] = ()
    source_sequence: int | None = Field(default=None, ge=0)

    @field_validator("event_end_time")
    @classmethod
    def _normalize_end(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class ComplexEventResult(EvaluationRecord):
    record_type: Literal["complex_event_result"] = "complex_event_result"
    result_id: str = Field(min_length=1)
    event_family: str = Field(min_length=1)
    event_start_time: datetime
    event_end_time: datetime
    emitted_at: datetime
    accepted: bool = True
    bindings: dict[str, str] = Field(default_factory=dict)
    provenance_refs: tuple[str, ...] = ()

    @field_validator("event_start_time", "event_end_time", "emitted_at")
    @classmethod
    def _normalize_event_times(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class ProviderLifecycleEvent(EvaluationRecord):
    record_type: Literal["provider_lifecycle"] = "provider_lifecycle"
    provider_instance_id: str = Field(min_length=1)
    lifecycle_event: str = Field(min_length=1)
    demand_ids: tuple[str, ...] = ()
    node_id: str = Field(min_length=1)
    image_or_version: str = ""
    startup_kind: str = ""


class ArtifactEvent(EvaluationRecord):
    record_type: Literal["artifact_event"] = "artifact_event"
    artifact_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    action: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    bytes: int = Field(default=0, ge=0)
    access_mode: str = ""
    expires_at: datetime | None = None
    bindings: dict[str, str] = Field(default_factory=dict)

    @field_validator("expires_at")
    @classmethod
    def _normalize_expiry(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)


class PlanDecision(EvaluationRecord):
    record_type: Literal["plan_decision"] = "plan_decision"
    decision_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    planning_scope: str = Field(min_length=1)
    selected_alternative_ids: tuple[str, ...] = ()
    selected_chain_ids: tuple[str, ...] = ()
    selected_node_ids: tuple[str, ...] = ()
    selected_source_ids: tuple[str, ...] = ()
    activated_provider_keys: tuple[str, ...] = ()
    continuation_types: tuple[str, ...] = ()
    predicted_completion_ms: int | None = Field(default=None, ge=0)
    predicted_transfer_bytes: int | None = Field(default=None, ge=0)
    predicted_compute_ms: int | None = Field(default=None, ge=0)
    predicted_slack_ms: int | None = None
    planning_latency_ms: float = Field(default=0, ge=0)
    labels_generated: int = Field(default=0, ge=0)
    labels_pruned: int = Field(default=0, ge=0)
    labels_retained: int = Field(default=0, ge=0)
    pruning_counts: dict[str, int] = Field(default_factory=dict)
    pruning_samples: tuple[str, ...] = ()
    oracle_gap_ms: int | None = None
    reason: str = ""
    frozen: bool = False
    resource_epoch: int = Field(default=0, ge=0)
    semantic_epoch: int = Field(default=0, ge=0)
    graph_version: int = Field(default=1, ge=1)
    replan_trigger: str = ""


class PredicateDemandRecord(EvaluationRecord):
    record_type: Literal["predicate_demand"] = "predicate_demand"
    demand_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    graph_version: int = Field(default=1, ge=1)
    predicate_id: str = Field(min_length=1)
    semantic_epoch: int = Field(default=0, ge=0)
    resource_epoch: int = Field(default=0, ge=0)
    bindings: dict[str, str] = Field(default_factory=dict)
    eligible_source_ids: tuple[str, ...] = ()
    deadline: datetime | None = None

    @field_validator("deadline")
    @classmethod
    def _normalize_deadline(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)


class ProviderCommand(EvaluationRecord):
    record_type: Literal["provider_command"] = "provider_command"
    command_id: str = Field(min_length=1)
    command: str = Field(min_length=1)
    provider_instance_id: str | None = None
    demand_ids: tuple[str, ...] = ()
    node_id: str = Field(min_length=1)
    emitted_at: datetime
    received_at: datetime | None = None

    @field_validator("emitted_at", "received_at")
    @classmethod
    def _normalize_command_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)


class ProviderLeaseEvent(EvaluationRecord):
    record_type: Literal["provider_lease"] = "provider_lease"
    lease_id: str = Field(min_length=1)
    provider_instance_id: str = Field(min_length=1)
    demand_id: str = Field(min_length=1)
    lease_event: str = Field(min_length=1)
    attached_at: datetime
    detached_at: datetime | None = None

    @field_validator("attached_at", "detached_at")
    @classmethod
    def _normalize_lease_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)


class DisturbanceEvent(EvaluationRecord):
    record_type: Literal["disturbance_event"] = "disturbance_event"
    disturbance_id: str = Field(min_length=1)
    disturbance_type: str = Field(min_length=1)
    action: str = Field(min_length=1)
    target_ids: tuple[str, ...] = ()
    condition_epoch: int = Field(default=0, ge=0)
    scheduled_trigger: str = ""
    validated: bool = False


class RetrospectiveAttempt(EvaluationRecord):
    record_type: Literal["retrospective_attempt"] = "retrospective_attempt"
    attempt_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    predicate_id: str = Field(min_length=1)
    replay_policy: str = Field(min_length=1)
    retained_interval_start: datetime
    retained_interval_end: datetime
    outcome: str = Field(min_length=1)
    artifact_ids: tuple[str, ...] = ()
    raw_bytes_read: int = Field(default=0, ge=0)
    transferred_bytes: int = Field(default=0, ge=0)
    processing_seconds: float = Field(default=0, ge=0)
    buffer_expiration_reason: str = ""

    @field_validator("retained_interval_start", "retained_interval_end")
    @classmethod
    def _normalize_retrospective_time(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class HypothesisTransition(EvaluationRecord):
    record_type: Literal["hypothesis_transition"] = "hypothesis_transition"
    graph_version: int = Field(ge=1)
    previous_state: str = Field(min_length=1)
    next_state: str = Field(min_length=1)
    transition_kind: str = Field(min_length=1)
    graph_node_ids: tuple[str, ...] = ()
    bindings_before: dict[str, str] = Field(default_factory=dict)
    bindings_after: dict[str, str] = Field(default_factory=dict)


class NetworkCondition(EvaluationRecord):
    record_type: Literal["network_condition"] = "network_condition"
    source_node_id: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    latency_ms: float = Field(ge=0)
    bandwidth_mbps: float = Field(ge=0)
    packet_loss_fraction: float = Field(default=0, ge=0, le=1)
    available: bool = True
    condition_epoch: int = Field(default=0, ge=0)


class ResourceSample(EvaluationRecord):
    record_type: Literal["resource_sample"] = "resource_sample"
    node_id: str = Field(min_length=1)
    cpu_utilization: float = Field(default=0, ge=0, le=1)
    cpu_time_seconds: float = Field(default=0, ge=0)
    memory_bytes: int = Field(default=0, ge=0)
    gpu_utilization: float | None = Field(default=None, ge=0, le=1)
    gpu_memory_bytes: int | None = Field(default=None, ge=0)
    gpu_time_seconds: float = Field(default=0, ge=0)
    gpu_energy_joules: float | None = Field(default=None, ge=0)
    network_tx_bytes: int = Field(default=0, ge=0)
    network_rx_bytes: int = Field(default=0, ge=0)


class CoordinationEpisode(EvaluationRecord):
    record_type: Literal["coordination_episode"] = "coordination_episode"
    episode_id: str = Field(min_length=1)
    campaign_year: int
    spatial_evaluation_eligible: bool
    upstream_sensor_id: str = Field(min_length=1)
    object_binding: str | None = None
    predicted_sensor_ids: tuple[str, ...] = ()
    activated_sensor_ids: tuple[str, ...] = ()
    replay_supported_sensor_ids: tuple[str, ...] = ()
    unavailable_mobile_sensor_ids: tuple[str, ...] = ()
    actual_downstream_sensor_id: str | None = None
    prediction_time: datetime
    downstream_observation_time: datetime | None = None
    deadline: datetime | None = None
    topology_confidence: str = ""
    route_ambiguity: int = Field(default=1, ge=1)

    @field_validator("prediction_time", "downstream_observation_time", "deadline")
    @classmethod
    def _normalize_episode_time(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)
