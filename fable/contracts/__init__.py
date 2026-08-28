"""Versioned data contracts grouped by architectural ownership.

Behavior lives in semantic/planning/scheduling/execution modules; these modules
contain transportable state, commands, results, and invariants only.
"""

from ._shared import NonEmptyStr
from .semantic import (
    GraphEdge,
    GraphNode,
    PredicateRole,
    RoleDefinition,
    SemanticGraph,
    SemanticPredicate,
    TemporalGuard,
)
from .hypothesis import (
    EntityBinding,
    FrontierSnapshot,
    Hypothesis,
    HypothesisNodeState,
    SemanticCheckpoint,
)
from .demand import (
    ContinuationRequirement,
    DataMovementConstraints,
    DemandBindingPolicy,
    PredicateDemand,
    SourcePreference,
)
from .provider import (
    CompatibilityGroup,
    ParameterSpec,
    ProviderContract,
    ProviderExecutionCapabilities,
    ProviderEvaluationContract,
    ProviderFamily,
    ProviderPort,
    ProviderRoleCapability,
    ProviderSemanticCapabilities,
)
from .artifact import ArtifactLocation, ArtifactProducer, ArtifactRef
from .execution import (
    ExecutionInput,
    ExecutionPlan,
    PhysicalPlanLabel,
    PlanCost,
    PlanStep,
    ResourceReservation,
)
from .result import BindingDelta, PredicateResult, ResultProvenance, TerminalComplexEvent
from .scheduling import ProviderLease
from .telemetry import (
    NodeCapacity,
    NodeHeartbeat,
    RuntimeLinkUpdate,
    RuntimeNodeUpdate,
    SourceHeartbeat,
)

__all__ = [
    "NonEmptyStr",
    "RoleDefinition",
    "PredicateRole",
    "SemanticPredicate",
    "TemporalGuard",
    "GraphNode",
    "GraphEdge",
    "SemanticGraph",
    "EntityBinding",
    "HypothesisNodeState",
    "Hypothesis",
    "SemanticCheckpoint",
    "FrontierSnapshot",
    "DataMovementConstraints",
    "ContinuationRequirement",
    "SourcePreference",
    "DemandBindingPolicy",
    "PredicateDemand",
    "ProviderRoleCapability",
    "ProviderSemanticCapabilities",
    "ProviderPort",
    "ParameterSpec",
    "ProviderExecutionCapabilities",
    "ProviderEvaluationContract",
    "CompatibilityGroup",
    "ProviderContract",
    "ProviderFamily",
    "ArtifactLocation",
    "ArtifactProducer",
    "ArtifactRef",
    "ExecutionInput",
    "PlanStep",
    "PlanCost",
    "PhysicalPlanLabel",
    "ResourceReservation",
    "ExecutionPlan",
    "BindingDelta",
    "ResultProvenance",
    "PredicateResult",
    "TerminalComplexEvent",
    "ProviderLease",
    "SourceHeartbeat",
    "NodeCapacity",
    "NodeHeartbeat",
    "RuntimeNodeUpdate",
    "RuntimeLinkUpdate",
]
