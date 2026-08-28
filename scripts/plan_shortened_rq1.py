#!/usr/bin/env python3
"""Generate the lease-controlled 9-variant x 5-policy RQ1 manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.catalog import ExperimentCatalog
from evaluation.experiments.matrix import build_run_matrix, write_planned_runs
from evaluation.experiments.specs import ExperimentQuestion


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
    by_variant: dict[str, list] = {}
    for experiment in catalog.recommended():
        by_variant.setdefault(experiment.ce_variant, []).append(experiment)

    rng = random.Random(args.seed)
    representatives = []
    for variant, candidates in sorted(by_variant.items()):
        # Prefer fully usable traces, then select reproducibly within the best
        # available quality tier.  This preserves authored CE distinctions
        # that the normalized-family sampler intentionally collapses.
        best_quality = min(
            (0 if item.quality_status == "usable" else 1)
            for item in candidates
        )
        eligible = sorted(
            (
                item
                for item in candidates
                if (0 if item.quality_status == "usable" else 1) == best_quality
            ),
            key=lambda item: item.experiment_id,
        )
        representatives.append(rng.choice(eligible))

    representative_catalog = ExperimentCatalog(representatives)
    runs = build_run_matrix(
        representative_catalog,
        ExperimentQuestion.RQ1_END_TO_END,
        repetitions=1,
        seed=args.seed,
        playback_mode="realtime",
        provider_profile_version="lease-controlled-v1",
    )
    expected = len(representatives) * 5
    if len(representatives) != 9 or len(runs) != expected:
        raise RuntimeError(
            f"expected 9 variants and 45 runs; got {len(representatives)} and {len(runs)}"
        )
    output = write_planned_runs(runs, args.output)
    summary = {
        "schema_version": "fable.rq1_lease_controlled_plan.v1",
        "planned_runs": len(runs),
        "unique_experiments": len(representatives),
        "selection_unit": "authored_ce_variant",
        "provider_profile_version": "lease-controlled-v1",
        "playback_mode": "realtime",
        "random_seed": args.seed,
        "baselines": dict(
            sorted(Counter(run.baseline_id.value for run in runs).items())
        ),
        "representatives": [
            {
                "ce_variant": item.ce_variant,
                "campaign_year": item.campaign_year,
                "experiment_id": item.experiment_id,
                "quality_status": item.quality_status,
            }
            for item in sorted(representatives, key=lambda item: item.ce_variant)
        ],
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
