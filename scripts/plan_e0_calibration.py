#!/usr/bin/env python3
"""Generate immutable E0 runs from the deployed provider inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.experiments.e0_calibration import (  # noqa: E402
    build,
    targets_from_inventory,
    write_manifest,
)
from fable.distributed.config import load_deployment_graph  # noqa: E402
from fable.planning.provider_registry import ProviderRegistry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan E0 provider/tier calibration from checked-in inventory."
    )
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
        "--output",
        type=Path,
        default=ROOT / "evaluation/manifests/providers/e0_calibration_runs.json",
    )
    parser.add_argument("--network-profile", action="append")
    parser.add_argument("--warm-repetitions", type=int, default=30)
    parser.add_argument("--cold-repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    registry = ProviderRegistry.from_files(
        catalog_path=args.catalog,
        data_types_path=args.data_types,
    )
    deployment = load_deployment_graph(args.deployment)
    targets = targets_from_inventory(registry, deployment)
    profiles = tuple(args.network_profile or ("good_network",))
    runs = build(
        targets,
        network_profiles=profiles,
        warm_repetitions=args.warm_repetitions,
        cold_repetitions=args.cold_repetitions,
        seed=args.seed,
    )
    destination = write_manifest(runs, args.output)
    print(
        json.dumps(
            {
                "schema_version": "fable.calibration_plan_summary.v1",
                "target_count": len(targets),
                "run_count": len(runs),
                "network_profiles": profiles,
                "output": str(destination.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
