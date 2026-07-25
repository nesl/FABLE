#!/usr/bin/env python3
"""Validate the internal FABLE evaluation harness and all prior runtime phases."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def validate_yaml() -> None:
    paths = (
        ROOT / "evaluation/manifests/baselines/static_pipelines.yaml",
        ROOT / "evaluation/manifests/deployments/spatial_scope.yaml",
        ROOT / "evaluation/manifests/network_profiles/controlled_profiles.yaml",
        ROOT / "iobt-minimal-ce-replay/compose.fable.evaluation.yaml",
    )
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            yaml.safe_load(handle)


def print_catalog_summary() -> None:
    path = ROOT / "evaluation/manifests/workloads/catalog_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    print("Evaluation catalog summary:")
    print(json.dumps(summary, indent=2))


def main() -> int:
    run(sys.executable, "scripts/build_evaluation_manifests.py")
    run(sys.executable, "scripts/export_evaluation_schemas.py")
    validate_yaml()
    run(sys.executable, "-m", "compileall", "-q", "evaluation", "scripts")
    run(sys.executable, "-m", "providers.tools.chain_validator")
    run(sys.executable, "-m", "pytest", "-q")
    print_catalog_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
