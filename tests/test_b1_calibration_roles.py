from __future__ import annotations

import json
from pathlib import Path

from scripts.derive_b1_trace_placement import derive


def _write_calibration(
    root: Path,
    *,
    predictions: list[dict],
    decisions: list[dict],
) -> Path:
    result = root / "calibration.json"
    result.write_text(
        json.dumps(
            {
                "baseline": "FABLE",
                "classification": "TRUE_POSITIVE",
                "condition_trace": None,
                "experiment_id": "trace-event",
                "scenario": "trace",
                "predictions": predictions,
            }
        ),
        encoding="utf-8",
    )
    records = result.with_suffix(".records")
    records.mkdir()
    (records / "predicate_observation.jsonl").write_text("", encoding="utf-8")
    (records / "plan_decision.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in decisions),
        encoding="utf-8",
    )
    return result


def test_calibration_projects_each_convergence_chain_onto_its_causal_roles(
    tmp_path: Path,
) -> None:
    node1 = "dvpg_gq_orin_1"
    node7 = "dvpg_gq_orin_7"
    prediction = {
        "accepted": True,
        "hypothesis_id": "terminal",
        "bindings": {
            "seed_vehicle": f"{node1}:run:0",
            "vehicle_a": f"{node7}:run:1",
            "vehicle_b": f"{node7}:run:2",
            "departing_vehicle_a": f"{node1}:run:0",
            "departing_vehicle_b": f"{node7}:run:2",
        },
    }
    decisions = [
        {
            "selected_chain_ids": [chain],
            "selected_node_ids": [node1, node7],
            "selected_source_ids": ["orin1_camera", "orin7_camera"],
            "activated_provider_keys": [
                f"{provider}@{node1}", f"{provider}@{node7}"
            ],
        }
        for chain, provider in (
            ("passes_live_vehicle", "pass_reference_evaluator"),
            ("pairwise_distance_live_vehicle", "pairwise_distance_evaluator"),
            ("track_lifecycle_exit_live_vehicle", "track_lifecycle_exit_evaluator"),
        )
    ]
    placement = derive(
        _write_calibration(tmp_path, predictions=[prediction], decisions=decisions)
    )

    assert placement["allowed_chain_node_ids"] == {
        "passes_live_vehicle": [node1],
        "pairwise_distance_live_vehicle": [node7],
        "track_lifecycle_exit_live_vehicle": [node1, node7],
    }
    assert placement["allowed_source_ids"] == ["orin1_camera", "orin7_camera"]


def test_calibration_normalizes_obsolete_raw_offload_to_fixed_causal_camera(
    tmp_path: Path,
) -> None:
    node4 = "dvpg_gq_orin_4"
    placement = derive(
        _write_calibration(
            tmp_path,
            predictions=[
                {
                    "accepted": True,
                    "bindings": {
                        "leader": f"{node4}:run:1",
                        "follower": f"{node4}:run:2",
                    },
                }
            ],
            decisions=[
                {
                    "selected_chain_ids": ["passes_live_vehicle"],
                    "selected_node_ids": [node4, "x86server"],
                    "selected_source_ids": ["orin4_camera"],
                    "activated_provider_keys": [
                        f"camera_projection@{node4}",
                        f"multi_object_tracker@{node4}",
                        f"pass_reference_evaluator@{node4}",
                        "yolo_vehicle_fast_640@x86server",
                    ],
                }
            ],
        )
    )

    providers = placement["allowed_chain_provider_node_ids"]["passes_live_vehicle"]
    assert providers["yolo_vehicle_fast_640"] == [node4]
    assert placement["allowed_chain_node_ids"]["passes_live_vehicle"] == [node4]
