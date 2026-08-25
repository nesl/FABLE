"""Stable enum values shared across FABLE modules."""

from __future__ import annotations

from enum import StrEnum


class GraphNodeKind(StrEnum):
    PREDICATE = "PREDICATE"
    AND = "AND"
    OR = "OR"
    K_OF_N = "K_OF_N"
    DURATION = "DURATION"
    ABSENT = "ABSENT"
    WITHIN = "WITHIN"
    NAMED_SUBGRAPH = "NAMED_SUBGRAPH"


class GraphEdgeKind(StrEnum):
    SEQUENCE = "SEQUENCE"
    DEPENDS_ON = "DEPENDS_ON"
    CHILD = "CHILD"
    ALTERNATIVE = "ALTERNATIVE"


class TemporalGuardKind(StrEnum):
    WITHIN = "WITHIN"
    DURATION = "DURATION"
    ABSENCE_WINDOW = "ABSENCE_WINDOW"
    PRECEDES = "PRECEDES"
    OVERLAPS = "OVERLAPS"
    MAX_GAP = "MAX_GAP"
    REPEAT_WITHIN = "REPEAT_WITHIN"


class BindingCapability(StrEnum):
    CONSUME = "CONSUME"
    INTRODUCE = "INTRODUCE"
    VALIDATE = "VALIDATE"
    INTRODUCE_OR_VALIDATE = "INTRODUCE_OR_VALIDATE"
    OBSERVE_ONLY = "OBSERVE_ONLY"
    AGGREGATE = "AGGREGATE"


class ResultKind(StrEnum):
    INSTANT_MATCH = "INSTANT_MATCH"
    INTERVAL_MATCH = "INTERVAL_MATCH"
    STATE_OBSERVATION = "STATE_OBSERVATION"
    GROUP_MATCH = "GROUP_MATCH"
    ARTIFACT_ONLY = "ARTIFACT_ONLY"


class HypothesisLifecycle(StrEnum):
    ACTIVE = "ACTIVE"
    FORKED = "FORKED"
    MERGED = "MERGED"
    COMPLETED = "COMPLETED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class HypothesisNodeStatus(StrEnum):
    UNRESOLVED = "UNRESOLVED"
    ENABLED = "ENABLED"
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    SATISFIED = "SATISFIED"
    FAILED = "FAILED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class CheckpointKind(StrEnum):
    PRIMITIVE = "PRIMITIVE"
    OR_RESOLUTION = "OR_RESOLUTION"
    AND_COMPLETION = "AND_COMPLETION"
    CARDINALITY = "CARDINALITY"
    WINDOW_CLOSURE = "WINDOW_CLOSURE"


class CheckpointStatus(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class TruthValue(StrEnum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class CancellationScope(StrEnum):
    BRANCH = "BRANCH"
    HYPOTHESIS = "HYPOTHESIS"
    TASK = "TASK"


class ArtifactAccessMode(StrEnum):
    LOCAL = "LOCAL"
    REMOTE_REFERENCE = "REMOTE_REFERENCE"
    TRANSFERRED = "TRANSFERRED"
    INLINE = "INLINE"


class ArtifactLocationKind(StrEnum):
    LOCAL_PATH = "LOCAL_PATH"
    OBJECT_URI = "OBJECT_URI"
    REMOTE_REFERENCE = "REMOTE_REFERENCE"
    INLINE = "INLINE"


class ProviderPortKind(StrEnum):
    INPUT = "INPUT"
    STATE_INPUT = "STATE_INPUT"
    OUTPUT = "OUTPUT"
    STATE_OUTPUT = "STATE_OUTPUT"


class ExecutionMode(StrEnum):
    LIVE = "LIVE"
    RETROSPECTIVE = "RETROSPECTIVE"


class ExecutionInputKind(StrEnum):
    """Origin of a concrete execution-step input.

    This is an execution contract, not a planner-internal concept.  Planning's
    historical ``ExternalInputKind`` name is kept as a compatibility alias.
    """

    LIVE_SOURCE = "LIVE_SOURCE"
    RETAINED_ARTIFACT = "RETAINED_ARTIFACT"
    DEPLOYMENT_ARTIFACT = "DEPLOYMENT_ARTIFACT"
    OMITTED_OPTIONAL = "OMITTED_OPTIONAL"


class PlanStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    ADMITTED = "ADMITTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class ProviderLeaseStatus(StrEnum):
    REQUESTED = "REQUESTED"
    STARTING = "STARTING"
    ACTIVE = "ACTIVE"
    IDLE_LEASE = "IDLE_LEASE"
    DRAINING = "DRAINING"
    RELEASED = "RELEASED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class NodeAvailability(StrEnum):
    AVAILABLE = "AVAILABLE"
    SUSPECT = "SUSPECT"
    UNAVAILABLE = "UNAVAILABLE"
    RECOVERING = "RECOVERING"
