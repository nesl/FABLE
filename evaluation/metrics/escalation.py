"""Provider-escalation metrics shared by every E4 policy."""

from __future__ import annotations

from pydantic import Field

from fable.common.base import FrozenFableModel
from evaluation.schemas import BaselineId


class EscalationSample(FrozenFableModel):
    baseline_id: BaselineId
    predicate_id: str
    provider_id: str
    stage: int = Field(ge=0)
    correct: bool
    resolved: bool
    escalated: bool
    latency_ms: float = Field(ge=0)
    cost: float = Field(ge=0)
    transferred_bytes: int = Field(ge=0)


class EscalationMetrics(FrozenFableModel):
    attempts: int = Field(ge=0)
    resolved: int = Field(ge=0)
    correct: int = Field(ge=0)
    escalation_frequency: float = Field(ge=0, le=1)
    resolution_rate: float = Field(ge=0, le=1)
    accuracy_on_resolved: float | None = Field(default=None, ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)
    total_cost: float = Field(ge=0)
    cost_per_correct_binding: float | None = Field(default=None, ge=0)
    transferred_bytes: int = Field(ge=0)
    resolved_by_stage: dict[int, int]


def summarize_escalation(
    samples: tuple[EscalationSample, ...],
) -> EscalationMetrics:
    attempts = len(samples)
    resolved = sum(item.resolved for item in samples)
    correct = sum(item.resolved and item.correct for item in samples)
    total_cost = sum(item.cost for item in samples)
    by_stage: dict[int, int] = {}
    for item in samples:
        if item.resolved:
            by_stage[item.stage] = by_stage.get(item.stage, 0) + 1
    return EscalationMetrics(
        attempts=attempts,
        resolved=resolved,
        correct=correct,
        escalation_frequency=(
            sum(item.escalated for item in samples) / attempts if attempts else 0
        ),
        resolution_rate=resolved / attempts if attempts else 0,
        accuracy_on_resolved=(correct / resolved if resolved else None),
        mean_latency_ms=(
            sum(item.latency_ms for item in samples) / attempts if attempts else 0
        ),
        total_cost=total_cost,
        cost_per_correct_binding=(total_cost / correct if correct else None),
        transferred_bytes=sum(item.transferred_bytes for item in samples),
        resolved_by_stage=dict(sorted(by_stage.items())),
    )
