#!/usr/bin/env python3
"""Prepare the full recommended-trace RQ1 matrix for all six policies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluation.catalog import ExperimentCatalog
from evaluation.experiments.matrix import PlannedRun, write_planned_runs
from evaluation.experiments.specs import ExperimentQuestion
from evaluation.schemas import BaselineId, EvaluationMode
from fable.common.ids import deterministic_id


ROOT = Path(__file__).resolve().parents[1]
ALL_RQ1_POLICIES = (
    BaselineId.B0_PRODUCE_ALL,
    BaselineId.B1_STATIC_WHOLE_EVENT,
    BaselineId.B2_FRONTIER_FIXED_REALIZATION,
    BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
    BaselineId.B4_GREEDY_FRONTIER,
    BaselineId.FABLE,
)

# These labels remain marked recommended in the source CSV, but the E3 replay
# audit established that they are not executable positive cells. Keep the
# exclusion explicit and auditable until the source catalog is corrected.
E1_REPLAY_EXCLUSIONS = {
    "20241009-two-vehicle-chase-3-r005": "no overlapping replay scenario",
    "20241009-two-vehicle-chase-13-r016": (
        "first matching PASSES observation occurs after the labeled CE window"
    ),
    "20241009-two-vehicle-chase-14-r017": "no overlapping replay scenario",
}


def planned_runs(catalog: ExperimentCatalog, *, seed: int) -> tuple[PlannedRun, ...]:
    """Build the executable, full-stack E1 matrix in CE round-robin order."""

    experiments_by_variant: dict[str, list[object]] = {}
    for experiment in catalog.recommended():
        if experiment.experiment_id in E1_REPLAY_EXCLUSIONS:
            continue
        experiments_by_variant.setdefault(experiment.ce_variant, []).append(experiment)
    for experiments in experiments_by_variant.values():
        experiments.sort(key=lambda item: item.experiment_id)

    ordered_experiments = []
    for trace_index in range(max(map(len, experiments_by_variant.values()))):
        for variant in sorted(experiments_by_variant):
            experiments = experiments_by_variant[variant]
            if trace_index < len(experiments):
                ordered_experiments.append(experiments[trace_index])

    rows = []
    for experiment in ordered_experiments:
        for baseline in ALL_RQ1_POLICIES:
            identity = {
                "question": ExperimentQuestion.RQ1_END_TO_END.value,
                "experiment_id": experiment.experiment_id,
                "baseline_id": baseline.value,
                "mode": EvaluationMode.FULL_STACK.value,
                "network_profile_id": "good_network",
                "repetition": 1,
                "random_seed": seed,
                "playback_mode": "realtime",
                "provider_profile_version": "lease-controlled-current-v3",
            }
            rows.append(
                PlannedRun(
                    run_id=deterministic_id("eval_run", identity, length=32),
                    question=ExperimentQuestion.RQ1_END_TO_END,
                    experiment_id=experiment.experiment_id,
                    baseline_id=baseline,
                    mode=EvaluationMode.FULL_STACK,
                    network_profile_id="good_network",
                    network_profile_path=(
                        "netwaggle/configs/profiles/good_network.json"
                    ),
                    repetition=1,
                    random_seed=seed,
                    playback_mode="realtime",
                    provider_profile_version="lease-controlled-current-v3",
                    campaign_year=experiment.campaign_year,
                    replay_supported_sensor_ids=experiment.replay_supported_sensor_ids,
                    unavailable_mobile_sensor_ids=experiment.unavailable_mobile_sensor_ids,
                    topology_deployment_ids=experiment.topology_deployment_ids,
                    warnings=experiment.spatial_notes,
                )
            )
    return tuple(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv",
        transition_model_path=(
            ROOT / "evaluation/labels/site_sensor_transition_model_2024_2025.json"
        ),
    )
    runs = planned_runs(catalog, seed=args.seed)

    selected_experiments = tuple(
        item
        for item in catalog.recommended()
        if item.experiment_id not in E1_REPLAY_EXCLUSIONS
    )
    selected_trace_count = len(selected_experiments)
    expected = selected_trace_count * len(ALL_RQ1_POLICIES)
    if len(runs) != expected:
        raise RuntimeError(f"expected {expected} runs, generated {len(runs)}")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    write_planned_runs(runs, output)
    excluded = [
        {
            "experiment_id": item.experiment_id,
            "quality_status": item.quality_status,
            "recommended_for_use": item.recommended_for_use,
        }
        for item in catalog.experiments
        if not item.recommended_for_use
    ]
    summary = {
        "schema_version": "fable.full_e1_plan.v2",
        "manifest": str(output.resolve()),
        "planned_runs": len(runs),
        "recommended_traces": len(catalog.recommended()),
        "selected_executable_traces": selected_trace_count,
        "catalog_traces": len(catalog.experiments),
        "execution_order": (
            "ce-round-robin: trace N from each CE; nominal FABLE calibrates "
            "the exact frozen B1 placement before B1, then remaining policies"
        ),
        "playback_mode": "realtime",
        "evaluation_mode": "FULL_STACK",
        "provider_profile_version": "lease-controlled-current-v3",
        "secondary_common_observation_control": "not included in executable manifest",
        "repetitions": 1,
        "baselines": dict(Counter(run.baseline_id.value for run in runs)),
        "traces_by_year": dict(Counter(item.campaign_year for item in selected_experiments)),
        "traces_by_variant": dict(Counter(item.ce_variant for item in selected_experiments)),
        "excluded_nonrecommended_traces": excluded,
        "excluded_replay_invalid_traces": dict(sorted(E1_REPLAY_EXCLUSIONS.items())),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
