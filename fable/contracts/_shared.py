"""Shared imports and value aliases for versioned FABLE contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, StringConstraints, UUID7, field_validator, model_validator

from fable.common.base import FableModel, FrozenFableModel, FrozenVersionedModel, JSONValue, VersionedModel
from fable.common.enums import (
    ArtifactAccessMode, ArtifactLocationKind, BindingCapability, CancellationScope,
    CheckpointKind, CheckpointStatus, ExecutionMode, ExecutionInputKind, GraphEdgeKind,
    GraphNodeKind, HypothesisLifecycle, HypothesisNodeStatus, NodeAvailability, PlanStatus,
    ProviderLeaseStatus, ProviderPortKind, ResultKind, TemporalGuardKind, TruthValue,
)
from fable.common.ids import canonical_hypothesis_key, demand_sharing_key, physical_plan_label_id, uuid7
from fable.common.time import DeadlineSpec, EventTimeInterval, LatenessPolicy, SourceWatermark, ensure_utc, utc_now

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
