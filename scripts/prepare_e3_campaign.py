#!/usr/bin/env python3
"""Prepare the matched E3 spatial and retrospective campaign manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluation.catalog import ExperimentCatalog
from evaluation.experiments.matrix import PlannedRun, write_planned_runs
from evaluation.experiments.specs import ExperimentQuestion
from evaluation.schemas import BaselineId, EvaluationMode
from fable.common.ids import deterministic_id

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPATIAL = (
    ROOT / "evaluation/results/rq3_all_campaigns_20260731/manifests/rq3b.jsonl"
)
# The historical RQ3B manifest contains three chase labels that are not valid
# replay cells in the current catalog.  r005/r017 have no overlapping replay
# scenario, while r016's first PASSES observations occur after the labeled CE
# window.  Use the two chase labels with observed replay overlap and omit the
# third slot rather than silently measuring coverage/label failures as spatial
# policy failures.
SPATIAL_TRACE_REPLACEMENTS = {
    "20241009-two-vehicle-chase-3-r005": "20241009-two-vehicle-chase-17-r020",
    "20241009-two-vehicle-chase-13-r016": "20241009-two-vehicle-chase-18-r021",
}
SPATIAL_TRACE_EXCLUSIONS = {
    "20241009-two-vehicle-chase-14-r017",
}
RETROSPECTIVE_TRACES = (
    # Robbery-with-alarm uses the intentionally non-retrospective
    # ``alarm_departure`` graph and therefore cannot discriminate R0/R1/R2.
    # E3 continuation uses only traces compiled to the cross-sensor robbery
    # graph, whose prior-entry and departure demands are retrospective.
    "20260414-cross-sensor-robbery-robbery-31-r031",
    "20260414-cross-sensor-robbery-robbery-32-r032",
    "20260415-cross-sensor-robbery-robbery-12-r012",
    "20260415-cross-sensor-robbery-robbery-13-r013",
)
RETROSPECTIVE_POLICIES = (
    (BaselineId.B1_HANDWRITTEN_STATIC, "R0_NO_REPLAY"),
    (BaselineId.B2_FRONTIER_FIXED_REALIZATION, "R1_RAW_REPLAY"),
    (BaselineId.FABLE, "R2_FABLE_TYPED_REPLAY"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evaluation/manifests/spatial/e3_prepared_20260827",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv",
        transition_model_path=(
            ROOT / "evaluation/labels/site_sensor_transition_model_2024_2025.json"
        ),
    )
    missing = sorted(set(RETROSPECTIVE_TRACES) - set(catalog.by_id))
    if missing:
        raise SystemExit(f"retrospective traces missing from catalog: {missing}")

    spatial_rows: list[PlannedRun] = []
    for line in SOURCE_SPATIAL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = PlannedRun.model_validate_json(line)
        if run.experiment_id in SPATIAL_TRACE_EXCLUSIONS:
            continue
        replacement = SPATIAL_TRACE_REPLACEMENTS.get(run.experiment_id)
        if replacement is not None:
            if replacement not in catalog.by_id:
                raise SystemExit(f"spatial replacement missing from catalog: {replacement}")
            identity = {
                "question": run.question.value,
                "experiment_id": replacement,
                "baseline_id": run.baseline_id.value,
                "repetition": run.repetition,
                "random_seed": run.random_seed,
            }
            run = run.model_copy(
                update={
                    "run_id": deterministic_id("eval_run", identity, length=32),
                    "experiment_id": replacement,
                }
            )
        spatial_rows.append(run)
    spatial = write_planned_runs(spatial_rows, output / "rq3b.jsonl")

    rows: list[PlannedRun] = []
    # Trace-major, policy-minor ordering makes partial results immediately paired.
    for experiment_id in RETROSPECTIVE_TRACES:
        experiment = catalog.by_id[experiment_id]
        for baseline, retrospective_policy in RETROSPECTIVE_POLICIES:
            identity = {
                "experiment_id": experiment_id,
                "baseline_id": baseline.value,
                "retrospective_policy_id": retrospective_policy,
                "seed": 33001,
            }
            rows.append(
                PlannedRun(
                    run_id=deterministic_id("eval_run", identity, length=32),
                    question=ExperimentQuestion.RQ3_CONTINUATION,
                    experiment_id=experiment_id,
                    baseline_id=baseline,
                    mode=EvaluationMode.FULL_STACK,
                    network_profile_id="good_network",
                    network_profile_path=(
                        "netwaggle/configs/profiles/good_network.json"
                    ),
                    playback_mode="realtime",
                    provider_profile_version="e3-retrospective-v1",
                    provider_execution_mode="real",
                    vlm_mode="replayed_response",
                    retrospective_policy_id=retrospective_policy,
                    campaign_year=experiment.campaign_year,
                    replay_supported_sensor_ids=experiment.replay_supported_sensor_ids,
                    unavailable_mobile_sensor_ids=experiment.unavailable_mobile_sensor_ids,
                    topology_deployment_ids=experiment.topology_deployment_ids,
                    random_seed=33001,
                    warnings=experiment.spatial_notes,
                )
            )
    retrospective = write_planned_runs(rows, output / "rq3c.jsonl")
    summary = {
        "schema_version": "fable.e3_prepared_campaign.v1",
        "execution_authorized": False,
        "execution_order": "trace_major_policy_minor",
        "playback_mode": "realtime",
        "network_profile": "good_network",
        "spatial": {
            "manifest": str(spatial),
            "sha256": digest(spatial),
            "rows": len(spatial_rows),
            "traces": len({run.experiment_id for run in spatial_rows}),
            "excluded_unreplayable_traces": sorted(SPATIAL_TRACE_EXCLUSIONS),
            "replaced_traces": dict(sorted(SPATIAL_TRACE_REPLACEMENTS.items())),
            "oracle_deferred": True,
        },
        "retrospective": {
            "manifest": str(retrospective),
            "sha256": digest(retrospective),
            "rows": len(rows),
            "traces": len(RETROSPECTIVE_TRACES),
            "policies": [value for _, value in RETROSPECTIVE_POLICIES],
            "eligibility": (
                "cross-sensor robbery graph with authored retrospective demands"
            ),
        },
    }
    (output / "campaign.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
