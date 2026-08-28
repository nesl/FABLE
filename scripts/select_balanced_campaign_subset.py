#!/usr/bin/env python3
"""Select a deterministic, paired number of traces per CE family."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluation.catalog import ExperimentCatalog


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces-per-ce", type=int, default=3)
    parser.add_argument(
        "--condition",
        choices=("all", "disturbed", "nominal"),
        default="all",
        help="Retain both conditions, only disturbed cells, or only nominal cells.",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        help=(
            "Exclude every experiment represented in this manifest. This is "
            "used to create a continuation without repeating an earlier "
            "balanced campaign."
        ),
    )
    args = parser.parse_args()

    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv"
    )
    family_by_experiment = {
        item.experiment_id: item.ce_variant for item in catalog.recommended()
    }
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    experiments_by_family: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        experiment_id = str(row["experiment_id"])
        family = family_by_experiment[experiment_id]
        if experiment_id not in experiments_by_family[family]:
            experiments_by_family[family].append(experiment_id)

    selected = {
        experiment_id
        for family in sorted(experiments_by_family)
        for experiment_id in experiments_by_family[family][
            : args.traces_per_ce
        ]
    }
    excluded: set[str] = set()
    if args.exclude_manifest is not None:
        excluded = {
            str(row["experiment_id"])
            for row in (
                json.loads(line)
                for line in args.exclude_manifest.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
        }
    selected_rows = [
        row
        for row in rows
        if str(row["experiment_id"]) in selected
        and str(row["experiment_id"]) not in excluded
        and (
            args.condition == "all"
            or (
                args.condition == "disturbed"
                and bool(row.get("condition_trace_path"))
            )
            or (
                args.condition == "nominal"
                and not bool(row.get("condition_trace_path"))
            )
        )
    ]
    continuation = selected - excluded
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected_rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "fable.balanced_campaign_subset.v1",
        "source_manifest": str(args.manifest.resolve()),
        "manifest": str(args.output.resolve()),
        "traces_per_ce": args.traces_per_ce,
        "condition": args.condition,
        "target_trace_count": len(selected),
        "excluded_trace_count": len(selected & excluded),
        "trace_count": len(continuation),
        "cell_count": len(selected_rows),
        "selected_by_ce": {
            family: experiments[: args.traces_per_ce]
            for family, experiments in sorted(experiments_by_family.items())
        },
        "continuation_by_ce": {
            family: [
                experiment
                for experiment in experiments[: args.traces_per_ce]
                if experiment not in excluded
            ]
            for family, experiments in sorted(experiments_by_family.items())
        },
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
