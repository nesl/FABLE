#!/usr/bin/env python3
"""Generate a matched canonical deployment, NetWaggle topology, and profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.deployment_topology import (
    build_network_profile,
    build_site_local_deployment,
    validate_unique_network_identities,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-count", type=int, default=20)
    parser.add_argument("--first-device-number", type=int, default=11)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--topology-out", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument(
        "--condition-map-out",
        type=Path,
        help="Defaults to CONDITION_MAP.json inside --profiles-dir.",
    )
    args = parser.parse_args()

    deployment = build_site_local_deployment(
        args.device_count,
        first_device_number=args.first_device_number,
    )
    validate_unique_network_identities(deployment)

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.topology_out.parent.mkdir(parents=True, exist_ok=True)
    args.profiles_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        yaml.safe_dump(deployment, sort_keys=False),
        encoding="utf-8",
    )
    args.topology_out.write_text(
        json.dumps(deployment, indent=2) + "\n",
        encoding="utf-8",
    )
    conditions = ("N0", "W1", "W2", "L1")
    for condition in conditions:
        path = args.profiles_dir / f"{condition}.json"
        path.write_text(
            json.dumps(build_network_profile(deployment, condition), indent=2)
            + "\n",
            encoding="utf-8",
        )
    condition_map_out = args.condition_map_out or (
        args.profiles_dir / "CONDITION_MAP.json"
    )
    condition_map_out.parent.mkdir(parents=True, exist_ok=True)
    condition_map_out.write_text(
        json.dumps(
            {
                "schema_version": "netwaggle.condition_map.v1",
                "conditions": {
                    condition: os.path.relpath(
                        (args.profiles_dir / f"{condition}.json").resolve(),
                        condition_map_out.resolve().parent,
                    )
                    for condition in conditions
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "device_count": deployment["device_count"],
                "logical_node_count": len(deployment["logical_nodes"]),
                "manifest": str(args.manifest_out),
                "topology": str(args.topology_out),
                "profiles": [
                    str(args.profiles_dir / f"{condition}.json")
                    for condition in conditions
                ],
                "condition_map": str(condition_map_out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
