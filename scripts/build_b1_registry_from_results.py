#!/usr/bin/env python3
"""Build an exact B1 registry from previously validated nominal executions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.derive_b1_trace_placement import install


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--base-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    targets = {
        str(row["experiment_id"])
        for row in (
            json.loads(line)
            for line in args.target_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    candidates: dict[str, list[tuple[int, Path, dict[str, object]]]] = defaultdict(list)
    preference = {"B1_STATIC_WHOLE_EVENT": 0, "FABLE": 1}
    for path in args.result_root.rglob("*.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        experiment_id = str(result.get("experiment_id") or "")
        baseline = str(result.get("baseline") or "")
        if (
            experiment_id not in targets
            or baseline not in preference
            or result.get("classification") != "TRUE_POSITIVE"
            or not bool((result.get("execution_conformance") or {}).get("valid"))
            or (result.get("condition_trace") or {}).get("transitions")
        ):
            continue
        candidates[experiment_id].append((preference[baseline], path, result))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.base_registry, args.output)
    installed: dict[str, dict[str, object]] = {}
    failures: dict[str, list[str]] = {}
    for experiment_id in sorted(targets):
        attempts: list[str] = []
        for _rank, path, result in sorted(
            candidates.get(experiment_id, ()), key=lambda item: (item[0], str(item[1]))
        ):
            try:
                placement = install(path, args.output)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                attempts.append(
                    f"{result.get('baseline')}:{path}: {type(exc).__name__}: {exc}"
                )
                continue
            installed[experiment_id] = {
                "source_baseline": result.get("baseline"),
                "source_result": str(path.resolve()),
                "trace_id": placement["trace_id"],
                "allowed_chain_ids": placement["allowed_chain_ids"],
                "allowed_node_ids": placement["allowed_node_ids"],
            }
            break
        if experiment_id not in installed:
            failures[experiment_id] = attempts or ["no conformant RQ1 true positive"]

    summary = {
        "schema_version": "fable.rq1_derived_b1_registry.v1",
        "target_manifest": str(args.target_manifest.resolve()),
        "result_root": str(args.result_root.resolve()),
        "registry": str(args.output.resolve()),
        "target_trace_count": len(targets),
        "installed_trace_count": len(installed),
        "unavailable_trace_count": len(failures),
        "installed": installed,
        "unavailable": failures,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
