"""Phase-5 deterministic multi-tenant scheduling and checkpoint control."""

from .adapters import CandidateAdapterError, candidate_from_search_result
from .admission import AdmissionConfig, MultiTenantScheduler
from .capacity import CapacityError, CapacityLedger
from .control import CancellationManager, CheckpointController
from .lifecycle import (
    AttachResult,
    ProviderLifecycleError,
    ProviderLifecycleManager,
    StepLeaseIntent,
)
from .models import (
    AdmissionBatchResult,
    AdmissionDecision,
    AdmissionRecord,
    ArtifactRetentionUpdate,
    CancellationOutcome,
    CancellationRequest,
    CheckpointControlOutcome,
    EvidenceUrgency,
    HistoricalDemand,
    HistoricalDemandRejection,
    HistoricalDemandStatus,
    HistoricalGenerationResult,
    ManagedLease,
    ManagedPlan,
    PlanCandidate,
    ProviderInstanceLifecycle,
    ProviderInstanceRecord,
    ProviderShareKey,
    ReplanRequest,
    TaskPriorityClass,
    TaskSchedulingPolicy,
)
from .order import MultiTenantOrderer
from .retrospective import (
    HistoricalDemandSpec,
    RetrospectiveConfig,
    RetrospectiveDemandGenerator,
)

__all__ = [
    "AdmissionBatchResult",
    "AdmissionConfig",
    "AdmissionDecision",
    "AdmissionRecord",
    "ArtifactRetentionUpdate",
    "AttachResult",
    "CancellationManager",
    "CancellationOutcome",
    "CancellationRequest",
    "CandidateAdapterError",
    "CapacityError",
    "CapacityLedger",
    "CheckpointControlOutcome",
    "CheckpointController",
    "EvidenceUrgency",
    "HistoricalDemand",
    "HistoricalDemandRejection",
    "HistoricalDemandSpec",
    "HistoricalDemandStatus",
    "HistoricalGenerationResult",
    "ManagedLease",
    "ManagedPlan",
    "MultiTenantOrderer",
    "MultiTenantScheduler",
    "PlanCandidate",
    "ProviderInstanceLifecycle",
    "ProviderInstanceRecord",
    "ProviderLifecycleError",
    "ProviderLifecycleManager",
    "ProviderShareKey",
    "ReplanRequest",
    "RetrospectiveConfig",
    "RetrospectiveDemandGenerator",
    "StepLeaseIntent",
    "TaskPriorityClass",
    "TaskSchedulingPolicy",
    "candidate_from_search_result",
]
