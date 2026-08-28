#!/usr/bin/env python3
"""Replay one immutable live E2 frontier across beam widths."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.baselines.factory import build_baseline_policy
from evaluation.e2_snapshots import (
    load_checkpoint_snapshot,
    load_checkpoint_snapshot_artifacts,
)
from evaluation.schemas import BaselineId
from fable.distributed.config import load_deployment_graph
from fable.planning import BoundedLabelPlanner
from fable.planning.beam_search import BeamSearchConfig
from fable.planning.provider_registry import ProviderRegistry


def signature(decision) -> dict[str, object]:
    return {
        "feasible": bool(decision.selected_alternative_ids),
        "chain_ids": sorted(decision.selected_chain_ids),
        "node_ids": sorted(decision.selected_node_ids),
        "completion_ms": decision.predicted_completion_ms,
        "transfer_bytes": decision.predicted_transfer_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--beam-widths", default="1,2,4,8,16,32")
    args = parser.parse_args()

    raw = args.snapshot.read_bytes()
    case = load_checkpoint_snapshot(args.snapshot)
    deployment = load_deployment_graph(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    providers = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
        profiles_path=ROOT / "evaluation/manifests/providers/calibrated_desktop_profiles.json",
    )
    artifacts = load_checkpoint_snapshot_artifacts(args.snapshot)
    if not artifacts.artifacts:
        raise SystemExit("snapshot is not self-contained: deployment_artifacts is empty")
    rows = []
    for width in tuple(int(x) for x in args.beam_widths.split(",")):
        for repetition in range(1, args.repetitions + 1):
            planner = BoundedLabelPlanner(
                provider_registry=providers,
                artifact_catalog=artifacts,
                deployment=deployment,
                config=BeamSearchConfig(beam_width=width, run_oracle=False),
            )
            policy = build_baseline_policy(BaselineId.FABLE, planner=planner)
            started = time.perf_counter_ns()
            decision = policy.plan(case)
            rows.append(
                {
                    "beam_width": width,
                    "repetition": repetition,
                    "wall_ms": (time.perf_counter_ns() - started) / 1_000_000,
                    "signature": signature(decision),
                }
            )
    deterministic = {
        str(width): len(
            {
                json.dumps(row["signature"], sort_keys=True)
                for row in rows
                if row["beam_width"] == width
            }
        ) == 1
        for width in sorted({row["beam_width"] for row in rows})
    }
    output = {
        "schema_version": "fable.e2_snapshot_beam_sweep.v1",
        "snapshot": str(args.snapshot.resolve()),
        "snapshot_file_sha256": hashlib.sha256(raw).hexdigest(),
        "trace_id": case.trace_id,
        "event_family": case.event_family,
        "demand_count": len(case.frontier_demands),
        "alternative_count": len(case.frontier_graph.alternatives),
        "deterministic_by_beam_width": deterministic,
        "all_deterministic": all(deterministic.values()),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2))
    return 0 if output["all_deterministic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
