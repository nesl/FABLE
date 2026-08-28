#!/usr/bin/env python3
"""Fail-closed static preflight for the updated RQ3a network matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "evaluation/manifests/adaptation/rq3a_updated"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(manifest_root: Path, topology_path: Path) -> dict[str, object]:
    summary = load_json(manifest_root / "rq3a_network_updated_55.summary.json")
    rows = [
        json.loads(line)
        for line in (manifest_root / "rq3a_network_updated_55.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    topology = load_json(topology_path)
    switch_names = {str(item) for item in topology.get("switches", ())}
    failures: list[str] = []
    policies = Counter(str(row["baseline_id"]) for row in rows)
    conditions = Counter(str(row["disturbance_profile_id"]) for row in rows)
    expected_policies = {
        "B1_STATIC_WHOLE_EVENT",
        "B2_FRONTIER_FIXED_REALIZATION",
        "B3_TASK_RESOURCE_ADAPTIVE",
        "B4_GREEDY_FRONTIER",
        "FABLE",
    }
    if len(rows) != 55:
        failures.append(f"expected 55 cells, found {len(rows)}")
    if set(policies) != expected_policies or set(policies.values()) != {11}:
        failures.append(f"invalid policy balance: {dict(policies)}")
    if conditions != Counter({"N0": 25, "N1": 15, "N2": 15}):
        failures.append(f"invalid condition balance: {dict(conditions)}")
    if any(row.get("playback_mode") != "realtime" for row in rows):
        failures.append("all cells must use realtime playback")
    required_nodes = {
        *(f"s_orin{index}" for index in (1, 4, 7, *range(11, 31))),
        *(f"s_mob{index}" for index in range(1, 7)),
        "s_site",
        "s_cloud",
    }
    missing_nodes = sorted(required_nodes - switch_names)
    if missing_nodes:
        failures.append(f"topology missing nodes: {missing_nodes}")
    trace_reports = []
    for case in summary["cases"]:
        trace = load_json(Path(case["condition_trace_path"]))
        condition = case["condition"]
        expected_target = case["target"]
        transitions = trace.get("transitions", [])
        valid = trace.get("anchor") == "TRACE_START"
        if condition == "N0":
            valid = valid and not transitions
        else:
            targets = [item.get("target_id") for item in transitions]
            offsets = [item.get("offset_s") for item in transitions]
            expected_offsets = [
                float(case["disturbance_start_seconds"]),
                float(case["disturbance_end_seconds"]),
            ]
            target = (
                f"sensor_uplink:{expected_target}"
                if condition == "N1"
                else "site_backbone"
            )
            # The updated experiment deliberately scales the disturbance to
            # each recording (50% through 95%).  Do not retain the obsolete
            # fixed 30--75 second contract here: it rejects every trace whose
            # duration is not 90 seconds while the manifest and trace agree.
            valid = valid and offsets == expected_offsets and targets == [target, target]
            target_switch = (
                f"s_mob{expected_target.rsplit('_', 1)[1]}"
                if str(expected_target).startswith("s_mobile_archive_")
                else expected_target
            )
            if condition == "N1" and target_switch not in switch_names:
                valid = False
        if not valid:
            failures.append(f"invalid trace contract: {case['case_id']}")
        trace_reports.append({"case_id": case["case_id"], "valid": valid})
    return {
        "schema_version": "fable.rq3a_updated_static_preflight.v1",
        "valid": not failures,
        "failures": failures,
        "planned_cells": len(rows),
        "policy_counts": dict(policies),
        "condition_counts": dict(conditions),
        "topology": str(topology_path),
        "trace_contracts": trace_reports,
        "runtime_requirements": {
            "namespace_binding_validation": True,
            "unshaped_docker_fallback_forbidden": True,
            "disturbed_cell_live_evidence_preflight": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--topology",
        type=Path,
        default=ROOT / "netwaggle/configs/site_evaluation_29node.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.manifest_root.resolve(), args.topology.resolve())
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
