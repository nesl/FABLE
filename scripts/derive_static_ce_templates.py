#!/usr/bin/env python3
"""Derive reusable B0/B1 CE templates from successful FABLE executions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    ROOT / "evaluation/results/rq1_9case_all_baselines_20260805/FABLE/repetition-01"
)


def ordered_union(target: list[str], values) -> None:
    for value in values or ():
        value = str(value)
        if value not in target:
            target.append(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "evaluation/manifests/baselines/fable_derived_ce_templates.yaml"
        ),
    )
    args = parser.parse_args()

    templates = {}
    for result_path in sorted(args.results.glob("*.json")):
        if result_path.name in {"plan.json", "report.json"}:
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("classification") != "TRUE_POSITIVE":
            continue
        decisions = result_path.with_suffix(".records") / "plan_decision.jsonl"
        if not decisions.is_file():
            raise RuntimeError(f"missing plan decisions for {result_path}")
        chains: list[str] = []
        providers: list[str] = []
        nodes: list[str] = []
        sources: list[str] = []
        for line in decisions.read_text(encoding="utf-8").splitlines():
            decision = json.loads(line)
            ordered_union(chains, decision.get("selected_chain_ids"))
            ordered_union(nodes, decision.get("selected_node_ids"))
            ordered_union(sources, decision.get("selected_source_ids"))
            ordered_union(
                providers,
                (
                    str(item).split("@", 1)[0]
                    for item in decision.get("activated_provider_keys") or ()
                ),
            )
        variant = str(result["variant"])
        templates[variant] = {
            "schema_version": "fable.calibrated_static_ce_template.v1",
            "exemplar_experiment_id": str(result["experiment_id"]),
            "exemplar_trace_id": str(result["scenario"]),
            "campaign_year": int(result["campaign_year"]),
            "calibration_outcome": "TRUE_POSITIVE",
            "placement_resolution": "intersection_with_trace_available_nodes",
            "allowed_chain_ids": chains,
            "allowed_provider_ids": providers,
            "allowed_node_ids": nodes,
            "allowed_source_ids": sources,
        }
    if len(templates) != 9:
        raise RuntimeError(f"expected nine successful CE templates, found {len(templates)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(
            {
                "schema_version": "fable.calibrated_static_ce_registry.v1",
                "calibration_results": str(args.results.resolve()),
                "templates": templates,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"templates": len(templates), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
