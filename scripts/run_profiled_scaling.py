#!/usr/bin/env python3
"""Execute a bounded E8 manifest using a versioned calibration profile."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.experiments.e8_scaling import PlannedScalingRun  # noqa: E402
from evaluation.metrics.statistics import LoadSample  # noqa: E402
from evaluation.report import generate_scaling_report  # noqa: E402
from evaluation.scaling_execution import (  # noqa: E402
    ScalingExecutionProfile,
    execute_profiled_scaling_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic profile-driven E8 scaling."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-runs", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument(
        "--allow-unmeasured-profile",
        action="store_true",
        help="Permit an explicitly uncalibrated implementation-validation profile.",
    )
    args = parser.parse_args()
    if args.maximum_runs is not None and args.maximum_runs < 1:
        parser.error("--maximum-runs must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    profile = ScalingExecutionProfile.model_validate_json(
        args.profile.read_text(encoding="utf-8")
    )
    runs = tuple(
        PlannedScalingRun.model_validate_json(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if args.maximum_runs is not None:
        runs = runs[: args.maximum_runs]
    args.output.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    results = []
    for run in runs:
        if monotonic() - started >= args.timeout_seconds:
            raise TimeoutError(
                f"profiled scaling exceeded {args.timeout_seconds} seconds "
                f"after {len(results)} runs"
            )
        results.append(
            execute_profiled_scaling_run(
                run,
                profile,
                args.output / "runs",
                allow_unmeasured_profile=args.allow_unmeasured_profile,
            )
        )

    fields = tuple(results[0].model_dump(mode="json")) if results else ()
    with (args.output / "campaign_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(item.model_dump(mode="json") for item in results)

    reports = {}
    report_groups = sorted(
        {(item.baseline_id, item.network_profile_id) for item in results}
    )
    for baseline, network_profile_id in report_groups:
        selected = tuple(
            item
            for item in results
            if item.baseline_id == baseline
            and item.network_profile_id == network_profile_id
        )
        target = min(item.target_timely_recall for item in selected)
        latency_bound = min(
            item.maximum_p95_control_latency_ms for item in selected
        )
        report_id = f"{baseline}|{network_profile_id}"
        reports[report_id] = generate_scaling_report(
            tuple(
                LoadSample(
                    workload=item.generated_labels,
                    timely_recall=item.timely_recall,
                    p95_latency_ms=item.p95_control_latency_ms,
                    completed=item.completed,
                )
                for item in selected
            ),
            args.output / "reports" / baseline / network_profile_id,
            target_timely_recall=target,
            maximum_p95_latency_ms=latency_bound,
        )
    summary = {
        "schema_version": "fable.profiled_scaling_campaign.v1",
        "profile_id": profile.profile_id,
        "calibrated": profile.calibrated,
        "planned_runs": len(runs),
        "completed_runs": len(results),
        "slo_satisfied_runs": sum(item.slo_satisfied for item in results),
        "elapsed_seconds": round(monotonic() - started, 6),
        "reports": reports,
    }
    (args.output / "campaign_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
