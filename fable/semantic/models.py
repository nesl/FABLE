"""Internal Phase-1 runtime records.

These records are intentionally separate from the cross-module Phase-0 wire
contracts.  They describe deterministic semantic-runtime decisions rather than
provider or distributed-execution choices.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, UUID7, field_validator, model_validator

from fable.common.base import FableModel
from fable.common.enums import TruthValue
from fable.common.ids import uuid7
from fable.common.schemas import (
    BindingDelta,
    FrontierSnapshot,
    PredicateResult,
    ResultProvenance,
    SemanticCheckpoint,
    SemanticPredicate,
)
from fable.common.time import EventTimeInterval, LatenessPolicy, ensure_utc, utc_now


class ApplyStatus(StrEnum):
    CREATED = "CREATED"
    APPLIED = "APPLIED"
    FORKED = "FORKED"
    MERGED = "MERGED"
    DUPLICATE = "DUPLICATE"
    STALE = "STALE"
    REJECTED = "REJECTED"
    NOOP = "NOOP"
    WINDOW_CLOSED = "WINDOW_CLOSED"


class SeedPredicateResult(FableModel):
    """Scripted or externally discovered match for a seed frontier node.

    Seed matches are not tied to an existing hypothesis, demand, or checkpoint.
    Once a hypothesis exists, ordinary ``PredicateResult`` records are used.
    """

    result_id: UUID7 = Field(default_factory=uuid7)
    occurrence_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    graph_hash: str = Field(min_length=1)
    graph_node_id: str = Field(min_length=1)
    semantic_predicate: SemanticPredicate
    truth: TruthValue = TruthValue.TRUE
    confidence: float | None = Field(default=None, ge=0, le=1)
    event_time_interval: EventTimeInterval
    binding_delta: BindingDelta = Field(default_factory=BindingDelta)
    artifact_ids: tuple[UUID7, ...] = ()
    provenance: ResultProvenance
    observed_at: datetime = Field(default_factory=utc_now)

    @field_validator("observed_at")
    @classmethod
    def _normalize_observed_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class SemanticRuntimeConfig(FableModel):
    request_id: str = Field(min_length=1)
    hypothesis_horizon_ms: int = Field(default=300_000, ge=1)
    deadline_offset_ms: int = Field(default=300_000, ge=1)
    lateness_policy: LatenessPolicy = Field(default_factory=LatenessPolicy)
    suppress_duplicate_occurrences: bool = True


class CancellationSet(FableModel):
    node_ids: tuple[str, ...] = ()
    branch_ids: tuple[str, ...] = ()
    reason: str = ""


class DerivedFrontier(FableModel):
    snapshot: FrontierSnapshot
    checkpoints: tuple[SemanticCheckpoint, ...]

    @model_validator(mode="after")
    def _checkpoint_ids_match(self) -> "DerivedFrontier":
        expected = tuple(checkpoint.checkpoint_id for checkpoint in self.checkpoints)
        if self.snapshot.checkpoint_ids != expected:
            raise ValueError("frontier checkpoint IDs do not match checkpoint records")
        return self

    def checkpoint_for_node(self, node_id: str) -> SemanticCheckpoint:
        for checkpoint in self.checkpoints:
            if node_id in checkpoint.node_ids:
                return checkpoint
        raise KeyError(node_id)


class RuntimeTransition(FableModel):
    status: ApplyStatus
    parent_hypothesis_id: UUID7 | None = None
    hypothesis_ids: tuple[UUID7, ...] = ()
    frontiers: tuple[DerivedFrontier, ...] = ()
    cancellation: CancellationSet = Field(default_factory=CancellationSet)
    result_id: UUID7 | None = None
    reason: str = ""


class ForkContext(FableModel):
    parent_hypothesis_id: UUID7
    parent_version: int = Field(ge=0)
    checkpoint_id: UUID7
    graph_node_id: str = Field(min_length=1)
    base_hypothesis_json: str
    child_hypothesis_ids: tuple[UUID7, ...] = ()


class ScriptedResultSpec(FableModel):
    """Compact fake-data instruction used by tests and the Phase-1 demo."""

    node_key: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    event_time_interval: EventTimeInterval
    introduced: dict[str, str] = Field(default_factory=dict)
    validated: dict[str, str] = Field(default_factory=dict)
    truth: TruthValue = TruthValue.TRUE
    confidence: float = Field(default=1.0, ge=0, le=1)
    occurrence_id: str | None = None
    provider_id: str = "scripted_reference_provider"


PredicateLikeResult = SeedPredicateResult | PredicateResult
