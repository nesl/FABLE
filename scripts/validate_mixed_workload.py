#!/usr/bin/env python3
"""Fail-closed validation and concise description of an RQ3a workload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.condition_trace import ConditionTrace  # noqa: E402
from evaluation.mixed_workload import MixedRequestWorkload  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    workload = MixedRequestWorkload.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    trace_path = (ROOT / workload.condition_trace_path).resolve()
    if not trace_path.is_relative_to(ROOT):
        raise ValueError("condition trace escapes repository root")
    trace = ConditionTrace.model_validate_json(trace_path.read_text(encoding="utf-8"))
    if trace.duration_s != workload.duration_s:
        raise ValueError("workload and condition-trace durations differ")
    catalog_ids = {
        json.loads(line)["experiment_id"]
        for line in (ROOT / "evaluation/manifests/workloads/ground_truth.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    missing = sorted(
        {episode.experiment_id for episode in workload.episodes} - catalog_ids
    )
    if missing:
        raise ValueError(f"unknown experiment IDs: {missing}")
    overlaps = sum(
        left.request_offset_s < right.request_offset_s < left.end_offset_s
        for index, left in enumerate(workload.episodes)
        for right in workload.episodes[index + 1 :]
    )
    print(json.dumps({
        "workload_id": workload.workload_id,
        "duration_s": workload.duration_s,
        "episodes": len(workload.episodes),
        "overlapping_episode_pairs": overlaps,
        "condition_trace_id": trace.trace_id,
        "systems": ["B2_FRONTIER_FIXED_REALIZATION", "B3_TASK_RESOURCE_ADAPTIVE", "FABLE"],
        "long_executions": 3,
        "validated": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
