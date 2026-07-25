"""Typed records for Phase-5 multi-tenant scheduling and checkpoint control.

The scheduling layer consumes checkpoint-bounded physical plans.  It does not
interpret semantic truth and it does not mutate the shared event graph.  Its
responsibilities are admission, capacity reservation, provider-token sharing,
lease ownership, scoped cancellation, and bounded retrospective work.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from fable.common.base import FableModel, FrozenFableModel
from fable.common.enums import (
    CancellationScope,
    ExecutionMode,
    PlanStatus,
    ProviderLeaseStatus,
)
from fable.common.ids import deterministic_id
from fable.common.schemas import (
    ExecutionPlan,
    PredicateDemand,
    ProviderLease,
    ResourceReservation,
)
from fable.common.time import EventTimeInterval, ensure_utc, utc_now
from fable.planning.models import PhysicalAlternative


class TaskPriorityClass(StrEnum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    BACKGROUND = "BACKGROUND"

    @property
    def rank(self) -> int:
        return {
            TaskPriorityClass.CRITICAL: 0,
            TaskPriorityClass.HIGH: 1,
            TaskPriorityClass.NORMAL: 2,
            TaskPriorityClass.BACKGROUND: 3,
        }[self]


class AdmissionDecision(StrEnum):
    ADMITTED = "ADMITTED"
    DEFERRED = "DEFERRED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class EvidenceUrgency(StrEnum):
    LIVE_ONLY = "LIVE_ONLY"
    EXPIRING_RETAINED = "EXPIRING_RETAINED"
    RETAINED = "RETAINED"

    @property
    def rank(self) -> int:
        return {
            EvidenceUrgency.LIVE_ONLY: 0,
            EvidenceUrgency.EXPIRING_RETAINED: 1,
            EvidenceUrgency.RETAINED: 2,
        }[self]


class ProviderInstanceLifecycle(StrEnum):
    COLD = "COLD"
    WARMING = "WARMING"
    ACTIVE = "ACTIVE"
    IDLE_LEASE = "IDLE_LEASE"
    DRAINING = "DRAINING"
    FAILED = "FAILED"


class HistoricalDemandStatus(StrEnum):
    CREATED = "CREATED"
    ADMITTED = "ADMITTED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"


class TaskSchedulingPolicy(FrozenFableModel):
    request_id: str = Field(min_length=1)
    priority_class: TaskPriorityClass = TaskPriorityClass.NORMAL
    historical_priority_override: bool = False


class PlanCandidate(FableModel):
    """One Phase-4 selected plan presented to the Phase-5 scheduler."""

    candidate_id: str | None = None
    plan: ExecutionPlan
    demands: tuple[PredicateDemand, ...]
    alternatives: tuple[PhysicalAlternative, ...]
    task_policy: TaskSchedulingPolicy
    predicted_completion_ms: int = Field(ge=0)
    startup_cost_ms: int = Field(default=0, ge=0)
    incremental_resource_cost_units: float = Field(default=0, ge=0)
    transfer_bytes: int = Field(default=0, ge=0)
    fallback_rank: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def _normalize_created_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _validate_alignment(self) -> Self:
        demand_ids = {demand.demand_id for demand in self.demands}
        if demand_ids != set(self.plan.demand_ids):
            raise ValueError("candidate demands must exactly match execution-plan demands")
        if self.task_policy.request_id not in {demand.request_id for demand in self.demands}:
            raise ValueError("task policy request_id must match the candidate demands")
        if len({demand.request_id for demand in self.demands}) != 1:
            raise ValueError("one plan candidate may belong to only one task/request")
        alt_ids = {alternative.demand_id for alternative in self.alternatives}
        if alt_ids != demand_ids:
            raise ValueError("candidate must contain one selected alternative per demand")
        if len(self.alternatives) != len(demand_ids):
            raise ValueError("candidate alternatives must be unique by demand")
        if self.plan.status not in (PlanStatus.CANDIDATE, PlanStatus.ADMITTED):
            raise ValueError("only candidate/admitted plans may enter scheduling")
        payload = {
            "plan_id": self.plan.plan_id,
            "label_id": self.plan.label_id,
            "demands": tuple(sorted(str(item) for item in demand_ids)),
            "alternatives": tuple(sorted(item.alternative_id for item in self.alternatives)),
            "fallback_rank": self.fallback_rank,
        }
        expected = deterministic_id("candidate", payload, length=32)
        if self.candidate_id is None:
            self.candidate_id = expected
        elif self.candidate_id != expected:
            raise ValueError("candidate_id does not match candidate content")
        return self

    @property
    def request_id(self) -> str:
        return self.demands[0].request_id

    @property
    def hypothesis_ids(self) -> tuple[UUID, ...]:
        return tuple(sorted({demand.hypothesis_id for demand in self.demands}, key=str))

    @property
    def earliest_deadline(self) -> datetime:
        return min(demand.deadline.latest_useful_completion for demand in self.demands)

    @property
    def latest_start(self) -> datetime:
        from datetime import timedelta

        return self.earliest_deadline - timedelta(milliseconds=self.predicted_completion_ms)

    def alternative_for_demand(self, demand_id: UUID) -> PhysicalAlternative:
        for alternative in self.alternatives:
            if alternative.demand_id == demand_id:
                return alternative
        raise KeyError(demand_id)


class ProviderShareKey(FrozenFableModel):
    key_id: str | None = None
    provider_id: str = Field(min_length=1)
    provider_contract_version: int = Field(ge=1)
    node_id: str = Field(min_length=1)
    configuration_hash: str = Field(min_length=1)
    source_signature: tuple[str, ...] = ()
    input_artifact_ids: tuple[UUID, ...] = ()
    input_data_types: tuple[str, ...] = ()
    output_data_types: tuple[str, ...] = ()
    event_time_interval: EventTimeInterval
    execution_mode: ExecutionMode
    policy_hash: str = Field(min_length=1)
    semantic_binding_signature: tuple[tuple[str, str], ...] = ()
    nonshareable_discriminator: str | None = None

    @model_validator(mode="after")
    def _derive_key(self) -> Self:
        payload = self.model_dump(mode="python", exclude={"key_id"}, exclude_none=True)
        expected = deterministic_id("provider_share", payload, length=32)
        if self.key_id is None:
            object.__setattr__(self, "key_id", expected)
        elif self.key_id != expected:
            raise ValueError("provider share key ID does not match key fields")
        return self


class ProviderInstanceRecord(FableModel):
    provider_instance_id: str = Field(min_length=1)
    share_key: ProviderShareKey
    lifecycle: ProviderInstanceLifecycle = ProviderInstanceLifecycle.COLD
    reservation: ResourceReservation
    active_lease_ids: tuple[UUID, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    idle_until: datetime | None = None
    failure_reason: str = ""
    lifecycle_history: tuple[ProviderInstanceLifecycle, ...] = (
        ProviderInstanceLifecycle.COLD,
    )

    @field_validator("created_at", "updated_at", "idle_until")
    @classmethod
    def _normalize_times(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    @model_validator(mode="after")
    def _validate_instance(self) -> Self:
        if self.reservation.node_id != self.share_key.node_id:
            raise ValueError("provider instance reservation and share key must use the same node")
        if not self.lifecycle_history or self.lifecycle_history[-1] != self.lifecycle:
            raise ValueError("provider lifecycle history must end in the current lifecycle")
        return self


class ManagedLease(FableModel):
    lease: ProviderLease
    request_id: str = Field(min_length=1)
    hypothesis_id: UUID
    checkpoint_id: UUID
    graph_node_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    share_key_id: str = Field(min_length=1)
    cancellation_scope: CancellationScope
    execution_mode: ExecutionMode
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def _normalize_created_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @property
    def status(self) -> ProviderLeaseStatus:
        return self.lease.status


class ManagedPlan(FableModel):
    candidate_id: str = Field(min_length=1)
    plan: ExecutionPlan
    active_demand_ids: tuple[UUID, ...]
    cancelled_demand_ids: tuple[UUID, ...] = ()
    completed_demand_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def _validate_partition(self) -> Self:
        groups = [
            set(self.active_demand_ids),
            set(self.cancelled_demand_ids),
            set(self.completed_demand_ids),
        ]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("managed-plan demand status sets must be disjoint")
        if set().union(*groups) != set(self.plan.demand_ids):
            raise ValueError("managed-plan demand status sets must cover plan demands")
        return self


class AdmissionRecord(FableModel):
    candidate_id: str = Field(min_length=1)
    decision: AdmissionDecision
    reason: str = ""
    plan_id: UUID | None = None
    lease_ids: tuple[UUID, ...] = ()
    created_provider_instance_ids: tuple[str, ...] = ()
    reused_provider_instance_ids: tuple[str, ...] = ()
    incremental_reservations: tuple[ResourceReservation, ...] = ()
    evidence_urgency: EvidenceUrgency
    latest_start: datetime
    order_rank: int = Field(ge=0)

    @field_validator("latest_start")
    @classmethod
    def _normalize_latest_start(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class AdmissionBatchResult(FableModel):
    ordered_candidate_ids: tuple[str, ...]
    records: tuple[AdmissionRecord, ...]
    resource_pressure: bool = False
    admitted_plan_ids: tuple[UUID, ...] = ()

    def record_for(self, candidate_id: str) -> AdmissionRecord:
        for record in self.records:
            if record.candidate_id == candidate_id:
                return record
        raise KeyError(candidate_id)


class CancellationRequest(FableModel):
    scope: CancellationScope
    request_id: str = Field(min_length=1)
    hypothesis_id: UUID | None = None
    graph_node_ids: tuple[str, ...] = ()
    reason: str = ""

    @model_validator(mode="after")
    def _validate_scope(self) -> Self:
        if self.scope == CancellationScope.BRANCH:
            if self.hypothesis_id is None or not self.graph_node_ids:
                raise ValueError("branch cancellation requires hypothesis_id and graph_node_ids")
        elif self.scope == CancellationScope.HYPOTHESIS and self.hypothesis_id is None:
            raise ValueError("hypothesis cancellation requires hypothesis_id")
        return self


class CancellationOutcome(FableModel):
    request: CancellationRequest
    released_lease_ids: tuple[UUID, ...] = ()
    cancelled_demand_ids: tuple[UUID, ...] = ()
    preserved_provider_instance_ids: tuple[str, ...] = ()
    idle_provider_instance_ids: tuple[str, ...] = ()
    cancelled_plan_ids: tuple[UUID, ...] = ()


class ReplanRequest(FableModel):
    request_id: str = Field(min_length=1)
    hypothesis_id: UUID
    frontier_id: UUID
    checkpoint_ids: tuple[UUID, ...]
    reason: str = Field(min_length=1)


class ArtifactRetentionUpdate(FableModel):
    artifact_id: UUID
    previous_expires_at: datetime | None = None
    new_expires_at: datetime
    reason: str = Field(min_length=1)

    @field_validator("previous_expires_at", "new_expires_at")
    @classmethod
    def _normalize_times(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)


class CheckpointControlOutcome(FableModel):
    completed_lease_ids: tuple[UUID, ...] = ()
    cancellation: CancellationOutcome | None = None
    retention_updates: tuple[ArtifactRetentionUpdate, ...] = ()
    replan_requests: tuple[ReplanRequest, ...] = ()
    historical_demand_ids: tuple[UUID, ...] = ()


class HistoricalDemand(FableModel):
    historical_id: str | None = None
    original_demand_id: UUID
    demand: PredicateDemand
    source_id: str = Field(min_length=1)
    retained_input_type: str = Field(min_length=1)
    historical_interval: EventTimeInterval
    buffer_expires_at: datetime
    reason: str = Field(min_length=1)
    status: HistoricalDemandStatus = HistoricalDemandStatus.CREATED
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("buffer_expires_at", "created_at")
    @classmethod
    def _normalize_times(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _derive_id(self) -> Self:
        if self.demand.event_time_interval != self.historical_interval:
            raise ValueError("historical demand interval must match the wrapped predicate demand")
        payload = {
            "original_demand_id": self.original_demand_id,
            "demand_id": self.demand.demand_id,
            "source_id": self.source_id,
            "interval": self.historical_interval,
            "retained_input_type": self.retained_input_type,
        }
        expected = deterministic_id("historical", payload, length=32)
        if self.historical_id is None:
            self.historical_id = expected
        elif self.historical_id != expected:
            raise ValueError("historical_id does not match historical demand content")
        return self


class HistoricalDemandRejection(FableModel):
    original_demand_id: UUID
    code: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class HistoricalGenerationResult(FableModel):
    demands: tuple[HistoricalDemand, ...] = ()
    rejections: tuple[HistoricalDemandRejection, ...] = ()
