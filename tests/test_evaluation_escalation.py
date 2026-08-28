from pathlib import Path

from evaluation.baselines.escalation import (
    EscalationCandidate,
    EscalationContext,
    EscalationPolicy,
    ProviderOutcome,
)
from evaluation.catalog import ExperimentCatalog
from evaluation.experiments.e0_calibration import CalibrationTarget, build as build_e0
from evaluation.experiments.e1_characterization import build as build_e1
from evaluation.experiments.e4_escalation import build as build_e4
from evaluation.escalation_execution import (
    EscalationStageProfile,
    execute_profiled_escalation_run,
)
from evaluation.experiments.matrix import PlannedRun
from evaluation.experiments.specs import ExperimentQuestion
from evaluation.metrics.escalation import EscalationSample, summarize_escalation
from evaluation.schemas import BaselineId, EvaluationMode

ROOT = Path(__file__).resolve().parents[1]


def _candidates():
    return (
        EscalationCandidate(
            provider_id="geometry",
            stage=0,
            estimated_cost=1,
            estimated_latency_ms=10,
            quality_score=0.6,
        ),
        EscalationCandidate(
            provider_id="reid",
            stage=1,
            estimated_cost=4,
            estimated_latency_ms=30,
            quality_score=0.85,
        ),
        EscalationCandidate(
            provider_id="vlm",
            stage=2,
            estimated_cost=10,
            estimated_latency_ms=100,
            quality_score=0.95,
            cloud=True,
        ),
    )


def test_controlled_escalation_policies_are_distinct() -> None:
    context = EscalationContext(candidates=_candidates())
    cheap = EscalationPolicy(BaselineId.C0_CHEAP_ONLY).choose(context)
    strong = EscalationPolicy(BaselineId.C1_STRONG_ONLY).choose(context)
    fable_initial = EscalationPolicy(BaselineId.C3_FABLE_ESCALATION).choose(
        context
    )
    no_escalation_initial = EscalationPolicy(
        BaselineId.C4_FABLE_NO_ESCALATION
    ).choose(context)
    assert cheap.selected_provider_id == "geometry"
    assert strong.selected_provider_id == "vlm"
    assert fable_initial.selected_provider_id == no_escalation_initial.selected_provider_id

    ambiguous = EscalationContext(
        candidates=_candidates(),
        previous_provider_ids=("geometry",),
        previous_outcome=ProviderOutcome.AMBIGUOUS,
        deadline_slack_ms=50,
    )
    fixed = EscalationPolicy(BaselineId.C2_FIXED_CASCADE).choose(ambiguous)
    fable = EscalationPolicy(BaselineId.C3_FABLE_ESCALATION).choose(ambiguous)
    stopped = EscalationPolicy(BaselineId.C4_FABLE_NO_ESCALATION).choose(
        ambiguous
    )
    assert fixed.selected_provider_id == "reid"
    assert fable.selected_provider_id == "reid"
    assert stopped.selected_provider_id is None


def test_e0_e1_e4_builders_are_bounded_and_deterministic() -> None:
    targets = (
        CalibrationTarget(
            target_id="yolo-device",
            provider_id="yolo",
            tier="device",
            input_class="720p-frame",
        ),
    )
    e0 = build_e0(targets, warm_repetitions=2, cold_repetitions=1, seed=7)
    assert len(e0) == 3
    assert e0 == build_e0(
        targets, warm_repetitions=2, cold_repetitions=1, seed=7
    )

    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv"
    )
    assert len(build_e1(catalog)) == len(catalog.recommended())
    e4 = build_e4(catalog, network_profiles=("good_network",))
    assert e4
    assert {item.baseline_id for item in e4} == {
        BaselineId.C0_CHEAP_ONLY,
        BaselineId.C1_STRONG_ONLY,
        BaselineId.C2_FIXED_CASCADE,
        BaselineId.C3_FABLE_ESCALATION,
        BaselineId.C4_FABLE_NO_ESCALATION,
    }


def test_escalation_metrics_report_cost_per_correct_binding() -> None:
    metrics = summarize_escalation(
        (
            EscalationSample(
                baseline_id=BaselineId.C3_FABLE_ESCALATION,
                predicate_id="IDENTITY_MATCH",
                provider_id="reid",
                stage=1,
                correct=True,
                resolved=True,
                escalated=True,
                latency_ms=20,
                cost=4,
                transferred_bytes=100,
            ),
        )
    )
    assert metrics.accuracy_on_resolved == 1
    assert metrics.cost_per_correct_binding == 4


def test_profiled_escalation_is_paired_across_policy_and_network() -> None:
    stages = (
        EscalationStageProfile(
            provider_id="cheap", stage=0, quality_score=0.7,
            ambiguity_probability=0.4, latency_ms=1, cost=1,
            transferred_bytes=100,
        ),
        EscalationStageProfile(
            provider_id="vlm", stage=2, quality_score=0.9,
            ambiguity_probability=0.1, latency_ms=100, cost=10,
            transferred_bytes=100_000, cloud=True,
        ),
    )

    def run(baseline: BaselineId, network: str):
        planned = PlannedRun(
            run_id=f"{baseline.value}-{network}",
            question=ExperimentQuestion.RQ_PROVIDER_ESCALATION,
            experiment_id="paired-trace",
            baseline_id=baseline,
            mode=EvaluationMode.COMMON_PERCEPTION,
            network_profile_id=network,
            random_seed=7,
        )
        return execute_profiled_escalation_run(
            planned, family="convoy", stages=stages
        )

    cheap_good, cheap_good_samples = run(
        BaselineId.C0_CHEAP_ONLY, "good_network"
    )
    cheap_bad, cheap_bad_samples = run(
        BaselineId.C4_FABLE_NO_ESCALATION, "constrained_bandwidth"
    )
    assert cheap_good.terminal_outcome == cheap_bad.terminal_outcome
    assert cheap_good.correct == cheap_bad.correct
    assert cheap_good_samples[0].resolved == cheap_bad_samples[0].resolved

    strong_good, strong_good_samples = run(
        BaselineId.C1_STRONG_ONLY, "good_network"
    )
    strong_bad, strong_bad_samples = run(
        BaselineId.C1_STRONG_ONLY, "constrained_bandwidth"
    )
    assert strong_good.terminal_outcome == strong_bad.terminal_outcome
    assert strong_good.correct == strong_bad.correct
    assert strong_bad_samples[0].latency_ms > strong_good_samples[0].latency_ms
