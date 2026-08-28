from __future__ import annotations

import json
from pathlib import Path

import yaml

from evaluation.schemas import BaselineId
from scripts.run_full_ce_suite import (
    condition_recovery_budget_seconds,
    pin_authored_static_provider_containers,
    require_zed_calibrations,
    resolve_static_chain_id,
)


def test_legacy_static_chain_ids_resolve_to_current_catalog_ids() -> None:
    assert resolve_static_chain_id("same_entity_cross_camera_reid") == (
        "follows_cross_camera_reid"
    )
    assert resolve_static_chain_id("recover_vehicle_from_local_segments") == (
        "recover_vehicle_before_audio_event"
    )
    assert resolve_static_chain_id("passes_live_vehicle") == "passes_live_vehicle"


def test_post_eof_restore_reserves_bounded_recovery_budget(tmp_path) -> None:
    trace = tmp_path / "post-eof.json"
    trace.write_text(
        json.dumps(
            {
                "duration_s": 150,
                "transitions": [
                    {"action": "FAIL_LINK", "offset_s": 0},
                    {"action": "RESTORE_LINK", "offset_s": 90},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert condition_recovery_budget_seconds(trace) == 180.0


def test_trace_without_restore_does_not_extend_runner_budget(tmp_path) -> None:
    trace = tmp_path / "fail-only.json"
    trace.write_text(
        json.dumps({"transitions": [{"action": "FAIL_LINK", "offset_s": 10}]}),
        encoding="utf-8",
    )

    assert condition_recovery_budget_seconds(trace) is None


def test_zed_calibration_preflight_is_limited_to_west_point(monkeypatch) -> None:
    # Historical campaigns do not use the West Point ZED serial inventory.
    require_zed_calibrations(2025)


def test_b1_pins_only_providers_for_each_authored_chain_node(
    tmp_path, monkeypatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "fable_provider_runtimes.yaml").write_text(
        yaml.safe_dump(
            {
                "nodes": {
                    node: {
                        "providers": {
                            "audio_event_classifier": {
                                "mode": "ADOPT_EXISTING",
                                "stop_adopted_when_idle": True,
                                "container_name": f"audio-{node}",
                            },
                            "yolo_vehicle_fast_640": {
                                "mode": "ADOPT_EXISTING",
                                "stop_adopted_when_idle": True,
                                "container_name": f"yolo-{node}",
                            },
                        }
                    }
                    for node in ("audio-node", "video-node")
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "static.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "trace_placements": {
                    "trace-1": {
                        "experiment_id": "experiment-1",
                        "allowed_chain_ids": [
                            "detect_audio_event",
                            "track_lifecycle_exit_live_vehicle",
                        ],
                        "allowed_provider_ids": [
                            "audio_event_classifier",
                            "yolo_vehicle_fast_640",
                        ],
                        "allowed_node_ids": ["audio-node", "video-node"],
                        "allowed_chain_node_ids": {
                            "detect_audio_event": ["audio-node"],
                            "track_lifecycle_exit_live_vehicle": ["video-node"],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FABLE_STATIC_PIPELINE_REGISTRY", str(registry))
    pinned = pin_authored_static_provider_containers(
        bundle,
        baseline_id=BaselineId.B1_STATIC_WHOLE_EVENT.value,
        placement_id="event",
        trace_id="trace-1",
    )

    assert pinned == ["audio-audio-node", "yolo-video-node"]
    runtime = yaml.safe_load(
        (bundle / "fable_provider_runtimes.yaml").read_text(encoding="utf-8")
    )
    assert not runtime["nodes"]["audio-node"]["providers"][
        "audio_event_classifier"
    ]["stop_adopted_when_idle"]
    assert runtime["nodes"]["audio-node"]["providers"][
        "yolo_vehicle_fast_640"
    ]["stop_adopted_when_idle"]
    assert runtime["nodes"]["video-node"]["providers"][
        "audio_event_classifier"
    ]["stop_adopted_when_idle"]
    assert not runtime["nodes"]["video-node"]["providers"][
        "yolo_vehicle_fast_640"
    ]["stop_adopted_when_idle"]


def test_b0_derives_ce_provider_union_without_trace_calibration(
    tmp_path, monkeypatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    provider_ids = (
        "yolo_vehicle_fast_640",
        "multi_object_tracker",
        "camera_projection",
        "pass_reference_evaluator",
    )
    (bundle / "fable_provider_runtimes.yaml").write_text(
        yaml.safe_dump(
            {
                "nodes": {
                    node_id: {
                        "providers": {
                            provider_id: {
                                "mode": "ADOPT_EXISTING",
                                "stop_adopted_when_idle": True,
                                "container_name": f"{provider_id}-{node_id}",
                            }
                            for provider_id in provider_ids
                        }
                    }
                    for node_id in ("orin11", "orin14", "x86server")
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "static.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "pipelines": {
                    "pass_follow_clear_convoy": {
                        "preferred_chain_ids": ["passes_live_vehicle"],
                        "fixed_sensor_policy": "all_replay_supported_orin",
                        "fixed_representation_policy": "tracks",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FABLE_STATIC_PIPELINE_REGISTRY", str(registry))

    pinned = pin_authored_static_provider_containers(
        bundle,
        baseline_id=BaselineId.B0_PRODUCE_ALL.value,
        placement_id="Pass-follow-clear convoy",
        trace_id="uncalibrated-trace",
    )

    assert pinned == sorted(
        f"{provider_id}-{node_id}"
        for node_id in ("orin11", "orin14")
        for provider_id in provider_ids
    )
    assert not any(name.endswith("-x86server") for name in pinned)
