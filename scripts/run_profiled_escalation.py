#!/usr/bin/env python3
"""Execute the bounded E4 PROFILED_VLM_REPLAY manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.escalation_execution import (  # noqa: E402
    EscalationStageProfile,
    execute_profiled_escalation_run,
)
from evaluation.experiments.matrix import PlannedRun  # noqa: E402
from evaluation.metrics.escalation import summarize_escalation  # noqa: E402


def family_for(experiment_id: str) -> str:
    if "convoy" in experiment_id:
        return "convoy"
    if "stalking" in experiment_id:
        return "repeated_visit"
    return "rendezvous"


def stages_for(family: str) -> tuple[EscalationStageProfile, ...]:
    definitions = {
        "convoy": (
            ("geometry_color", 0, 0.78, 0.35, 0.8, 1.0, 8_192, False),
            ("task_vehicle_reid", 1, 0.89, 0.15, 15.0, 4.0, 65_536, False),
        ),
        "repeated_visit": (
            ("geometry_color", 0, 0.68, 0.65, 0.8, 1.0, 8_192, False),
            ("task_person_reid", 1, 0.87, 0.25, 40.0, 4.0, 65_536, False),
        ),
        "rendezvous": (
            ("person_proximity", 0, 0.72, 0.55, 0.5, 1.0, 8_192, False),
            ("conversation_action", 1, 0.88, 0.20, 0.5, 4.0, 65_536, False),
        ),
    }
    # The hosted-provider base excludes the 91.943 ms good-network transfer
    # component from the 3096.034 ms LIVE_VLM calibration mean.
    values = (*definitions[family], ("hosted_vlm_identity", 2, 0.82, 0.18, 3004.091, 10.0, 524_288, True))
    return tuple(EscalationStageProfile(
        provider_id=value[0], stage=value[1], quality_score=value[2],
        ambiguity_probability=value[3], latency_ms=value[4], cost=value[5],
        transferred_bytes=value[6], cloud=value[7],
    ) for value in values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = tuple(
        PlannedRun.model_validate_json(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    samples_by_group = defaultdict(list)
    for run in runs:
        family = family_for(run.experiment_id)
        result, samples = execute_profiled_escalation_run(
            run, family=family, stages=stages_for(family)
        )
        results.append(result)
        samples_by_group[(result.baseline_id, result.network_profile_id)].extend(samples)
        run_dir = args.output / "runs" / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        with (run_dir / "escalation_samples.jsonl").open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(sample.model_dump_json() + "\n")
    fields = tuple(results[0].model_dump(mode="json")) if results else ()
    with (args.output / "campaign_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(item.model_dump(mode="json") for item in results)
    reports = {
        f"{baseline}|{profile}": summarize_escalation(tuple(samples)).model_dump(mode="json")
        for (baseline, profile), samples in sorted(samples_by_group.items())
    }
    summary = {
        "schema_version": "fable.profiled_escalation_campaign.v1",
        "planned_runs": len(runs), "completed_runs": len(results),
        "correct_resolved_runs": sum(item.correct and item.resolved for item in results),
        "unresolved_runs": sum(not item.resolved for item in results),
        "execution_mode": "PROFILED_VLM_REPLAY",
        "local_timing_profile_calibrated": True,
        "task_outcome_profile_calibrated": False,
        "hosted_vlm_latency_profile_calibrated": True,
        "hosted_vlm_latency_calibration": (
            "evaluation/results/e4_bounded_20260801/live_validation_retry/"
            "live_vlm_calibration.json"
        ),
        "hosted_vlm_outcome_profile_calibrated": False,
        "publishable": False,
        "validity_note": (
            "Policy execution and local timing are measured/implemented, but task "
            "ambiguity, correctness, and hosted-VLM outcomes are deterministic "
            "profiles rather than live observations."
        ),
        "reports": reports,
    }
    (args.output / "campaign_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
