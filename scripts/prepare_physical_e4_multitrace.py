#!/usr/bin/env python3
"""Freeze the bounded multi-trace physical E4 campaign without executing it."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.catalog import ExperimentCatalog
from scripts.derive_b1_trace_placement import derive

FAMILIES = (
    "Route convoy",
    "Three-visit stalking",
    "Vehicle convergence",
    "Cross-sensor robbery",
)
RESULT_ROOTS = (
    ROOT / "evaluation/results",
    Path("/media/brianw/Extreme SSD2/fable_results"),
)
OUTPUT = ROOT / "evaluation/manifests/adaptation/physical_e4_multitrace.json"
CONDITION_ROOT = ROOT / "evaluation/manifests/adaptation/physical_e4_multitrace_conditions"


def result_index() -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = defaultdict(list)
    for root in RESULT_ROOTS:
        for directory, _, files in os.walk(root):
            for filename in files:
                if filename.endswith(".json"):
                    indexed[filename[:-5]].append(Path(directory) / filename)
    return indexed


def calibration(experiment_id: str, indexed: dict[str, list[Path]]) -> tuple[Path | None, dict | None]:
    candidates = []
    for path in indexed.get(experiment_id, ()): 
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("baseline") != "FABLE" or result.get("classification") != "TRUE_POSITIVE":
                continue
            placement = derive(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        candidates.append((path.stat().st_mtime, path, placement, result))
    if not candidates:
        return None, None
    _, path, placement, result = max(candidates, key=lambda row: row[0])
    return path, {"placement": placement, "result": result}


def replay_sources(experiment, evidence: dict | None) -> tuple[list[dict], str]:
    nodes = list((((evidence or {}).get("result") or {}).get("suite") or {}).get("replay_nodes") or ())
    if not nodes:
        nodes = (
            ["mobile_archive_1", "mobile_archive_2", "mobile_archive_3", "orin1", "orin4", "orin7"]
            if experiment.campaign_year == 2024
            else ["orin11", "orin13", "orin14", "orin15", "orin16"]
        )
        if experiment.date == "2026-04-15":
            nodes.insert(1, "orin12")
    allowed = list((((evidence or {}).get("placement") or {}).get("allowed_node_ids") or ()))
    def replay_name(node: str) -> str:
        value = node.removeprefix("dvpg_gq_")
        return "orin" + value.removeprefix("orin_") if value.startswith("orin_") else value

    normalized = [replay_name(node) for node in allowed if replay_name(node) in nodes]
    fallback = next((node for node in nodes if node.startswith("orin")), nodes[0])
    physical = normalized[0] if normalized else fallback
    sources = [
        {"logical_replay_node": node, "execution": "physical_pi" if node == physical else "desktop"}
        for node in nodes
    ]
    planner_node = (
        physical if physical.startswith("mobile_archive_")
        else "dvpg_gq_orin_" + physical.removeprefix("orin")
    )
    return sources, planner_node


def main() -> int:
    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv"
    )
    indexed = result_index()
    cases = []
    CONDITION_ROOT.mkdir(parents=True, exist_ok=True)
    counts = {}
    calibration_required = []
    recommended = tuple(catalog.recommended())
    for family in FAMILIES:
        selected = [item for item in recommended if item.ce_variant == family][:10]
        counts[family] = len(selected)
        for experiment in selected:
            calibration_path, evidence = calibration(experiment.experiment_id, indexed)
            sources, planner_node = replay_sources(experiment, evidence)
            needs_calibration = calibration_path is None
            if needs_calibration:
                calibration_required.append(experiment.experiment_id)
            duration = float(experiment.duration_seconds)
            apply_at = round(max(5.0, duration * 0.5 - 22.5), 3)
            restore_at = round(min(duration - 5.0, apply_at + 45.0), 3)
            if restore_at <= apply_at:
                restore_at = round(apply_at + 10.0, 3)
            condition_paths = {}
            actions = {
                "compute_contention": ("APPLY_COMPUTE_CONTENTION", "CLEAR_COMPUTE_CONTENTION", "physical_jetson", "JETSON_E1", "N0"),
                "network_degradation": ("APPLY_NETWORK_PROFILE", "RESTORE_NETWORK_PROFILE", "physical_link:rpi_to_jetson", "P1_JETSON_PATH_DEGRADED", "N0"),
                "network_disconnect": ("FAIL_LINK", "RESTORE_LINK", "physical_link:rpi_to_jetson", None, None),
            }
            for condition, (apply_action, restore_action, target, apply_profile, restore_profile) in actions.items():
                transitions = [
                    {"transition_id": f"{experiment.experiment_id}:{condition}:apply", "offset_s": apply_at, "action": apply_action, "target_id": target},
                    {"transition_id": f"{experiment.experiment_id}:{condition}:restore", "offset_s": restore_at, "action": restore_action, "target_id": target},
                ]
                if apply_profile:
                    transitions[0]["profile_id"] = apply_profile
                if restore_profile:
                    transitions[1]["profile_id"] = restore_profile
                trace_path = CONDITION_ROOT / f"{experiment.experiment_id}.{condition}.json"
                trace_path.write_text(json.dumps({
                    "schema_version": "fable.condition_trace.v1",
                    "trace_id": f"physical-e4:{experiment.experiment_id}:{condition}",
                    "initial_network_profile": "N0", "initial_compute_profile": "N0",
                    "anchor": "TRACE_START", "transitions": transitions,
                    "duration_s": round(duration + 60.0, 3), "random_seed": 34010,
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                condition_paths[condition] = str(trace_path.relative_to(ROOT))
            cases.append({
                "experiment_id": experiment.experiment_id,
                "ce_variant": family,
                "campaign_year": experiment.campaign_year,
                "duration_seconds": experiment.duration_seconds,
                "max_seconds": int(duration + 120),
                "hard_cell_timeout_seconds": int(duration + 240),
                "logical_physical_node_id": planner_node,
                "replay_sources": sources,
                "b1_calibration_result": str(calibration_path.resolve()) if calibration_path else None,
                "nominal_fable_calibration_required": needs_calibration,
                "conditions": ["nominal", "compute_contention", "network_degradation", "network_disconnect"],
                "condition_trace_paths": condition_paths,
                "disturbance_window_seconds": [apply_at, restore_at],
                "baselines": ["B1_STATIC_WHOLE_EVENT", "B3_TASK_RESOURCE_ADAPTIVE", "FABLE"],
                "playback_mode": "realtime",
                "playback_speed": 1.0,
            })
    document = {
        "schema_version": "fable.physical_e4_multitrace.v1",
        "campaign_id": "physical-e4-multitrace-four-ce-v1",
        "selection_cap_per_ce": 10,
        "available_trace_counts": counts,
        "trace_count": len(cases),
        "matrix_cell_count": len(cases) * 4 * 3,
        "calibration_prepass_cell_count": len(calibration_required),
        "total_maximum_execution_cells": len(cases) * 12 + len(calibration_required),
        "condition_schedule": "trace-relative centered 45-second window",
        "network_disconnect_semantics": {
            "path": "physical Pi raw-video TCP service to Jetson/NetWaggle",
            "implementation": "fixed root helper rejects TCP port 8090 while preserving SSH control",
            "apply_offset_s": 10,
            "restore_offset_s": 55,
            "automatic_safety_restore_seconds": 180
        },
        "calibration_required_experiment_ids": calibration_required,
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: document[key] for key in (
        "trace_count", "available_trace_counts", "matrix_cell_count",
        "calibration_prepass_cell_count", "total_maximum_execution_cells",
    )}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
