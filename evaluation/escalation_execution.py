"""Deterministic PROFILED_VLM_REPLAY execution for bounded E4 runs."""

from __future__ import annotations

from pydantic import Field

from evaluation.baselines.escalation import (
    EscalationCandidate,
    EscalationContext,
    EscalationPolicy,
    ProviderOutcome,
)
from evaluation.experiments.matrix import PlannedRun
from evaluation.metrics.escalation import EscalationMetrics, EscalationSample, summarize_escalation
from fable.common.base import FrozenFableModel
from fable.common.ids import deterministic_id


class EscalationStageProfile(FrozenFableModel):
    provider_id: str
    stage: int = Field(ge=0)
    quality_score: float = Field(ge=0, le=1)
    ambiguity_probability: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    cost: float = Field(ge=0)
    transferred_bytes: int = Field(ge=0)
    cloud: bool = False


class ProfiledEscalationResult(FrozenFableModel):
    schema_version: str = "fable.profiled_escalation_result.v1"
    run_id: str
    experiment_id: str
    baseline_id: str
    network_profile_id: str
    family: str
    execution_mode: str = "PROFILED_VLM_REPLAY"
    local_timing_profile_calibrated: bool = True
    task_outcome_profile_calibrated: bool = False
    hosted_vlm_latency_profile_calibrated: bool = True
    hosted_vlm_outcome_profile_calibrated: bool = False
    invoked_provider_ids: tuple[str, ...]
    resolved: bool
    correct: bool
    terminal_outcome: str
    metrics: EscalationMetrics


def execute_profiled_escalation_run(
    run: PlannedRun,
    *,
    family: str,
    stages: tuple[EscalationStageProfile, ...],
) -> tuple[ProfiledEscalationResult, tuple[EscalationSample, ...]]:
    policy = EscalationPolicy(run.baseline_id)
    candidates = tuple(
        EscalationCandidate(
            provider_id=item.provider_id,
            stage=item.stage,
            estimated_cost=item.cost,
            estimated_latency_ms=round(item.latency_ms),
            quality_score=item.quality_score,
            transferred_bytes=item.transferred_bytes,
            cloud=item.cloud,
        )
        for item in stages
    )
    by_provider = {item.provider_id: item for item in stages}
    used: list[str] = []
    samples: list[EscalationSample] = []
    previous_outcome: ProviderOutcome | None = None
    terminal = ProviderOutcome.INSUFFICIENT_EVIDENCE
    correct = False
    for _ in range(len(stages) + 1):
        decision = policy.choose(
            EscalationContext(
                candidates=candidates,
                previous_provider_ids=tuple(used),
                previous_outcome=previous_outcome,
                deadline_slack_ms=3000,
            )
        )
        if decision.resolved:
            terminal = previous_outcome or ProviderOutcome.INSUFFICIENT_EVIDENCE
            break
        if decision.selected_provider_id is None:
            break
        stage = by_provider[decision.selected_provider_id]
        used.append(stage.provider_id)
        # Every policy/network condition sees the same latent provider outcome
        # for a trace and repetition.  This preserves the paired counterfactual
        # design; only policy choice and network transfer latency may differ.
        ambiguity_draw = _draw(run, stage.provider_id, "ambiguity")
        correctness_draw = _draw(run, stage.provider_id, "correctness")
        resolved = ambiguity_draw >= stage.ambiguity_probability
        correct_now = resolved and correctness_draw <= stage.quality_score
        previous_outcome = (
            ProviderOutcome.MATCH
            if resolved and correct_now
            else ProviderOutcome.NO_MATCH
            if resolved
            else ProviderOutcome.AMBIGUOUS
        )
        latency = stage.latency_ms + _network_latency_ms(run, stage)
        samples.append(
            EscalationSample(
                baseline_id=run.baseline_id,
                predicate_id="SAME_ENTITY_OR_INTERACTION",
                provider_id=stage.provider_id,
                stage=stage.stage,
                correct=correct_now,
                resolved=resolved,
                escalated=len(used) > 1,
                latency_ms=latency,
                cost=stage.cost,
                transferred_bytes=stage.transferred_bytes,
            )
        )
        if resolved:
            terminal = previous_outcome
            correct = correct_now
            break
    metrics = summarize_escalation(tuple(samples))
    return (
        ProfiledEscalationResult(
            run_id=run.run_id,
            experiment_id=run.experiment_id,
            baseline_id=run.baseline_id.value,
            network_profile_id=run.network_profile_id,
            family=family,
            invoked_provider_ids=tuple(used),
            resolved=terminal in {ProviderOutcome.MATCH, ProviderOutcome.NO_MATCH},
            correct=correct,
            terminal_outcome=terminal.value,
            metrics=metrics,
        ),
        tuple(samples),
    )


def _draw(run: PlannedRun, provider_id: str, purpose: str) -> float:
    token = deterministic_id(
        "e4_profile_draw",
        (run.experiment_id, str(run.random_seed), provider_id, purpose),
    )
    return int(token[-8:], 16) / 0xFFFFFFFF


def _network_latency_ms(run: PlannedRun, stage: EscalationStageProfile) -> float:
    if not stage.cloud:
        return 0.0
    if run.network_profile_id == "good_network":
        rtt_ms, bandwidth_mbps = 50.0, 100.0
    elif run.network_profile_id == "constrained_bandwidth":
        rtt_ms, bandwidth_mbps = 190.0, 8.0
    else:
        raise ValueError(f"unsupported E4 network profile: {run.network_profile_id}")
    return rtt_ms + stage.transferred_bytes * 8 / (bandwidth_mbps * 1_000_000) * 1000
