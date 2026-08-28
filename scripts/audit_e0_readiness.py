#!/usr/bin/env python3
"""Write the exact runtime/fixture boundary for E0 targets without mutation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.calibration_coverage import (  # noqa: E402
    classify_calibration_targets,
    readiness_summary,
    worker_coverage_summary,
)
from evaluation.experiments.e0_calibration import targets_from_inventory  # noqa: E402
from fable.distributed.config import (  # noqa: E402
    ProviderRuntimeResolver,
    load_deployment_graph,
)
from fable.planning.provider_registry import ProviderRegistry  # noqa: E402
from providers.calibration_worker import worker_capabilities  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit executable E0 coverage.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "providers/registry/catalog.yaml",
    )
    parser.add_argument(
        "--data-types",
        type=Path,
        default=ROOT / "providers/registry/data_types.yaml",
    )
    parser.add_argument(
        "--deployment",
        type=Path,
        default=ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml",
    )
    parser.add_argument(
        "--runtimes",
        type=Path,
        default=ROOT
        / "iobt-minimal-ce-replay/config/fable_provider_runtimes.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = ProviderRegistry.from_files(
        catalog_path=args.catalog,
        data_types_path=args.data_types,
    )
    deployment = load_deployment_graph(args.deployment)
    runtimes = ProviderRuntimeResolver.from_yaml(args.runtimes)
    rows = classify_calibration_targets(
        targets_from_inventory(registry, deployment),
        registry=registry,
        deployment=deployment,
        runtimes=runtimes,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = readiness_summary(rows)
    worker_coverage = worker_coverage_summary(
        rows, worker_capabilities()["operations"]
    )
    (args.output / "readiness.json").write_text(
        json.dumps(
            {
                **summary,
                "worker_coverage": worker_coverage,
                "targets": [item.model_dump(mode="json") for item in rows],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output / "readiness.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "provider_id",
            "tier",
            "input_class",
            "status",
            "candidate_node_ids",
            "runtime_node_ids",
            "requires_replay_fixture",
            "hosted_external",
            "reason",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow(
                {
                    "provider_id": item.target.provider_id,
                    "tier": item.target.tier,
                    "input_class": item.target.input_class,
                    "status": item.status,
                    "candidate_node_ids": ",".join(item.candidate_node_ids),
                    "runtime_node_ids": ",".join(item.runtime_node_ids),
                    "requires_replay_fixture": item.requires_replay_fixture,
                    "hosted_external": item.hosted_external,
                    "reason": item.reason,
                }
            )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
