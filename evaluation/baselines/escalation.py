"""Controlled provider-escalation policies for E4."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from fable.common.base import FrozenFableModel
from evaluation.schemas import BaselineId


class ProviderOutcome(StrEnum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EscalationCandidate(FrozenFableModel):
    provider_id: str = Field(min_length=1)
    stage: int = Field(ge=0)
    estimated_cost: float = Field(ge=0)
    estimated_latency_ms: int = Field(ge=0)
    quality_score: float = Field(ge=0, le=1)
    transferred_bytes: int = Field(default=0, ge=0)
    cloud: bool = False


class EscalationContext(FrozenFableModel):
    candidates: tuple[EscalationCandidate, ...]
    previous_provider_ids: tuple[str, ...] = ()
    previous_outcome: ProviderOutcome | None = None
    deadline_slack_ms: int | None = None
    cloud_available: bool = True


class EscalationDecision(FrozenFableModel):
    baseline_id: BaselineId
    selected_provider_id: str | None = None
    escalated: bool = False
    resolved: bool = False
    reason: str


class EscalationPolicy:
    def __init__(self, baseline_id: BaselineId) -> None:
        if baseline_id not in {
            BaselineId.C0_CHEAP_ONLY,
            BaselineId.C1_STRONG_ONLY,
            BaselineId.C2_FIXED_CASCADE,
            BaselineId.C3_FABLE_ESCALATION,
            BaselineId.C4_FABLE_NO_ESCALATION,
        }:
            raise ValueError(f"not an escalation baseline: {baseline_id}")
        self.baseline_id = baseline_id

    def choose(self, context: EscalationContext) -> EscalationDecision:
        feasible = tuple(
            item
            for item in context.candidates
            if context.cloud_available or not item.cloud
        )
        if not feasible:
            return EscalationDecision(
                baseline_id=self.baseline_id,
                reason="no feasible escalation provider",
            )
        if context.previous_outcome in {
            ProviderOutcome.MATCH,
            ProviderOutcome.NO_MATCH,
        }:
            return EscalationDecision(
                baseline_id=self.baseline_id,
                resolved=True,
                reason="the previous provider returned a terminal outcome",
            )
        used = set(context.previous_provider_ids)
        remaining = tuple(item for item in feasible if item.provider_id not in used)
        if not remaining:
            return EscalationDecision(
                baseline_id=self.baseline_id,
                reason="the eligible cascade is exhausted",
            )
        if self.baseline_id == BaselineId.C0_CHEAP_ONLY:
            candidate = min(feasible, key=_cheap_key)
            if used:
                return EscalationDecision(
                    baseline_id=self.baseline_id,
                    reason="cheap-only policy does not escalate",
                )
        elif self.baseline_id == BaselineId.C1_STRONG_ONLY:
            candidate = min(feasible, key=_strong_key)
            if used:
                return EscalationDecision(
                    baseline_id=self.baseline_id,
                    reason="strong-only policy invokes one provider",
                )
        elif self.baseline_id == BaselineId.C4_FABLE_NO_ESCALATION:
            if used:
                return EscalationDecision(
                    baseline_id=self.baseline_id,
                    reason="FABLE no-escalation stops after the first provider",
                )
            candidate = min(feasible, key=_fable_initial_key)
        elif self.baseline_id == BaselineId.C2_FIXED_CASCADE:
            candidate = min(remaining, key=lambda item: (item.stage, item.provider_id))
        else:
            if not used:
                candidate = min(feasible, key=_fable_initial_key)
                return EscalationDecision(
                    baseline_id=self.baseline_id,
                    selected_provider_id=candidate.provider_id,
                    escalated=False,
                    reason="selected FABLE's initial provider",
                )
            within_deadline = tuple(
                item
                for item in remaining
                if context.deadline_slack_ms is None
                or item.estimated_latency_ms <= context.deadline_slack_ms
            )
            candidate = min(
                within_deadline or remaining,
                key=_fable_escalation_key,
            )
        return EscalationDecision(
            baseline_id=self.baseline_id,
            selected_provider_id=candidate.provider_id,
            escalated=bool(used),
            reason="selected the next provider allowed by the controlled policy",
        )


def _cheap_key(item: EscalationCandidate) -> tuple[object, ...]:
    return (item.estimated_cost, item.estimated_latency_ms, item.provider_id)


def _strong_key(item: EscalationCandidate) -> tuple[object, ...]:
    # "Strong only" denotes the strongest cascade tier, not the largest raw
    # quality number. Quality scores from task-specific providers are not
    # necessarily calibrated against cheap providers across different tasks.
    return (-item.stage, -item.quality_score, item.estimated_latency_ms, item.provider_id)


def _fable_initial_key(item: EscalationCandidate) -> tuple[object, ...]:
    return (
        item.estimated_cost,
        item.estimated_latency_ms,
        -item.quality_score,
        item.provider_id,
    )


def _fable_escalation_key(item: EscalationCandidate) -> tuple[object, ...]:
    return (
        -item.quality_score,
        item.estimated_latency_ms,
        item.estimated_cost,
        item.transferred_bytes,
        item.provider_id,
    )
