#!/usr/bin/env python3
"""Freeze disconnect targets to sensors required by validated B1 placements."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _source_node(source_id: str) -> str | None:
    match = re.match(r"^(orin\d+|mobile\d+)_", source_id.strip().lower())
    if not match:
        return None
    prefix = match.group(1)
    if prefix.startswith("orin"):
        return f"dvpg_gq_orin_{prefix.removeprefix('orin')}"
    return f"mobile_{prefix.removeprefix('mobile')}"


def _switch(node_id: str) -> str:
    if node_id.startswith("dvpg_gq_orin_"):
        return f"s_orin{node_id.rsplit('_', 1)[-1]}"
    if node_id.startswith("mobile_"):
        return f"s_mobile{node_id.rsplit('_', 1)[-1]}"
    raise ValueError(f"unsupported B1 sensor node: {node_id}")


def _target_node(placement: dict[str, object]) -> tuple[str, str]:
    sources = [str(item) for item in placement.get("allowed_source_ids") or ()]
    allowed_nodes = {str(item) for item in placement.get("allowed_node_ids") or ()}

    # Prefer the first authored camera source: visual evidence is the payload
    # actually severed by a sensor-uplink outage. Preserve registry order.
    ordered = [item for item in sources if item.endswith("_camera")]
    ordered.extend(item for item in sources if item not in ordered)
    for source in ordered:
        node = _source_node(source)
        if node and node in allowed_nodes:
            return node, source

    for nodes in (placement.get("allowed_chain_node_ids") or {}).values():
        for node in nodes:
            node = str(node)
            if node in allowed_nodes and node != "x86server" and node != "cloud1":
                return node, "allowed_chain_node_ids"
    raise ValueError("B1 placement has no disconnectable sensor source")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, required=True)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument(
        "--require-placement",
        action="store_true",
        help="Omit every trace without an exact validated B1 placement.",
    )
    args = parser.parse_args()

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8")) or {}
    by_experiment = {
        str(value["experiment_id"]): value
        for value in (registry.get("trace_placements") or {}).values()
    }
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.condition_root.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, dict[str, object]] = {}
    output: list[dict[str, object]] = []
    for row in rows:
        if args.baseline and str(row["baseline_id"]) not in set(args.baseline):
            continue
        experiment_id = str(row["experiment_id"])
        placement = by_experiment.get(experiment_id)
        if placement is None:
            # B1 is unavailable for this trace; retain the pre-generated cut
            # for the remaining policies and record that it was not retargeted.
            if not args.require_placement:
                output.append(row)
            continue
        node, source = _target_node(placement)
        target = f"link:{_switch(node)}:s_edge"
        if row.get("condition_trace_path"):
            source_path = Path(str(row["condition_trace_path"]))
            trace = json.loads(source_path.read_text(encoding="utf-8"))
            for transition in trace.get("transitions") or ():
                if transition.get("action") in {"FAIL_LINK", "RESTORE_LINK"}:
                    transition["target_id"] = target
            destination = args.condition_root / f"{experiment_id}_{row['baseline_id']}.json"
            destination.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
            row["condition_trace_path"] = str(destination.resolve())
        output.append(row)
        provenance[experiment_id] = {
            "b1_trace_id": placement.get("trace_id"),
            "b1_source_id": source,
            "b1_source_node_id": node,
            "disconnect_target": target,
            "same_target_for_all_baselines": True,
            "fanout_allowed": False,
        }

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )
    provenance_path = args.output_manifest.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "rows": len(output),
        "retargeted_experiments": len(provenance),
        "manifest": str(args.output_manifest.resolve()),
        "provenance": str(provenance_path.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
