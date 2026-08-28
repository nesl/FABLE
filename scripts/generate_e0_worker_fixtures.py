#!/usr/bin/env python3
"""Generate deterministic, typed E0 fixtures for implemented worker targets."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

from providers.vehicle.models import scoped_track_identity


BASE = datetime(2026, 1, 1, tzinfo=UTC)


def interval(start: float, end: float) -> dict:
    return {
        "start": (BASE + timedelta(seconds=start)).isoformat(),
        "end": (BASE + timedelta(seconds=end)).isoformat(),
    }


def track(local_id: int, x: float, second: float, class_name: str = "car") -> dict:
    return {
        "local_track_id": local_id,
        "scoped_track_id": scoped_track_identity("camera", "session", local_id),
        "source_id": "camera",
        "tracker_session_id": "session",
        "class_name": class_name,
        "confidence": 0.95,
        "bbox": {"x1": x, "y1": 0, "x2": x + 2, "y2": 4},
        "event_time": (BASE + timedelta(seconds=second)).isoformat(),
        "world_point": {
            "x": x,
            "y": 0,
            "coordinate_frame_id": "world",
        },
    }


def track_set(second: float, tracks: list[dict]) -> dict:
    return {
        "source_id": "camera",
        "tracker_family": "fixture",
        "tracker_version": "1",
        "tracker_session_id": "session",
        "event_time": (BASE + timedelta(seconds=second)).isoformat(),
        "tracks": tracks,
    }


def target(provider: str, input_class: str) -> dict:
    return {
        "target_id": f"desktop-{provider}-{input_class}".replace("+", "-"),
        "provider_id": provider,
        "tier": "sensor",
        "input_class": input_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-path", default="/calibration/frame.jpg")
    parser.add_argument("--small-model-path", default="/calibration/yolov8s.pt")
    parser.add_argument("--nano-model-path", default="/calibration/yolov8n.pt")
    args = parser.parse_args()

    route = {
        "route_id": "route-a",
        "coordinate_frame_id": "world",
        "points": [
            {"x": 0, "y": 0, "coordinate_frame_id": "world"},
            {"x": 20, "y": 0, "coordinate_frame_id": "world"},
        ],
    }
    zone = {
        "zone_id": "door",
        "coordinate_frame_id": "world",
        "polygon": [
            {"x": 0, "y": -2, "coordinate_frame_id": "world"},
            {"x": 5, "y": -2, "coordinate_frame_id": "world"},
            {"x": 5, "y": 5, "coordinate_frame_id": "world"},
            {"x": 0, "y": 5, "coordinate_frame_id": "world"},
        ],
    }
    leader = scoped_track_identity("camera", "session", 1)
    follower = scoped_track_identity("camera", "session", 2)
    follows_frames = [
        track_set(
            second,
            [
                {**track(1, 20 + 4 * second, second), "world_point": None},
                {**track(2, 10 + 4 * second, second), "world_point": None},
            ],
        )
        for second in range(4)
    ]
    people = [
        track_set(second, [track(3, 0, second, "person"), track(4, 1, second, "person")])
        for second in (0, 1)
    ]
    speaker_turns = {
        "source_id": "microphone",
        "event_time_interval": interval(0, 2),
        "turns": [
            {
                "turn_id": "turn-a",
                "speaker_id": "speaker-a",
                "event_time_interval": interval(0, 1),
                "speech_probability": 0.95,
            },
            {
                "turn_id": "turn-b",
                "speaker_id": "speaker-b",
                "event_time_interval": interval(1, 2),
                "speech_probability": 0.95,
            },
        ],
        "speaker_count": 2,
        "diarization_model_id": "fixture",
        "diarization_model_version": "1",
    }
    waveform_left = [0.0] * 800
    waveform_right = [0.0] * 800
    waveform_left[100] = 1.0
    waveform_right[101] = 1.0
    audio = {
        "source_id": "microphone",
        "event_time_interval": interval(0, 0.1),
        "sample_rate_hz": 8000,
        "channel_ids": ["left", "right"],
        "waveform": [waveform_left, waveform_right],
    }
    detection_frames = [
        {
            "source_id": "camera",
            "event_time": (BASE + timedelta(seconds=second)).isoformat(),
            "frame_id": f"frame-{second}",
            "image_width": 640,
            "image_height": 480,
            "detector_id": "fixture",
            "detector_version": "1",
            "detections": [
                {
                    "detection_id": f"car-{second}",
                    "class_name": "car",
                    "confidence": 0.95,
                    "bbox": {"x1": 100 + second, "y1": 100, "x2": 220 + second, "y2": 220},
                }
            ],
        }
        for second in (0, 1, 2)
    ]
    fixtures: dict[tuple[str, str], dict] = {
        ("pairwise_distance_evaluator", "projected_track_set.v1"): {
            "left": track(1, 0, 0),
            "right": track(2, 3, 0),
            "maximum_distance_m": 5,
            "expected_truth": True,
        },
        ("motion_state_evaluator", "projected_track_set.v1"): {
            "track_sets": [track_set(0, [track(1, 0, 0)]), track_set(2, [track(1, 4, 2)])],
            "expected_predicate_id": "MOVING",
        },
        ("route_map_matcher", "projected_track_set.v1+route_graph.v1"): {
            "track": track(1, 3, 0),
            "routes": [route],
            "expected_route_id": "route-a",
        },
        ("zone_membership_evaluator", "projected_track_set.v1+route_graph.v1"): {
            "track": track(1, 2, 0),
            "zone": zone,
            "expected_truth": True,
        },
        ("follows_local_geometry", "projected_track_set.v1"): {
            "track_sets": follows_frames,
            "leader_id": leader,
            "follower_id": follower,
            "minimum_duration_s": 2,
            "expected_truth": True,
        },
        ("follows_local_geometry", "projected_track_set.v1+pair_trajectory.v1"): {
            "track_sets": follows_frames,
            "leader_id": leader,
            "follower_id": follower,
            "minimum_duration_s": 2,
            "expected_truth": True,
        },
        ("zone_transition_evaluator", "projected_track_set.v1+route_graph.v1"): {
            "zone": zone,
            "tracks": [track(1, -3, 0), track(1, 2, 1)],
            "expected_predicate_id": "ENTERS",
        },
        ("pass_reference_evaluator", "projected_track_set.v1+route_graph.v1"): {
            "reference": {
                "reference_id": "gate",
                "start": {"x": 0, "y": -5, "coordinate_frame_id": "world"},
                "end": {"x": 0, "y": 5, "coordinate_frame_id": "world"},
            },
            "tracks": [track(1, -1, 0), track(1, 1, 1)],
            "expected_truth": True,
        },
        ("person_proximity_provider", "track_set.v1"): {
            "track_sets": people,
            "minimum_duration_seconds": 1,
            "expected_truth": True,
        },
        ("conversation_provider", "projected_track_set.v1+speaker_turn_set.v1"): {
            "track_sets": people,
            "speaker_turn_set": speaker_turns,
            "minimum_duration_seconds": 0,
            "expected_truth": True,
        },
        ("conversation_provider", "projected_track_set.v1+speaker_turn_set.v1+transcript_event_set.v1"): {
            "track_sets": people,
            "speaker_turn_set": speaker_turns,
            "minimum_duration_seconds": 0,
            "required_terms": [],
            "expected_truth": True,
        },
        ("gcc_phat_audio_localizer", "audio_segment.v1+microphone_array_geometry.v1"): {
            "window": audio,
            "geometry": {
                "array_id": "array",
                "coordinate_frame_id": "array",
                "microphones": [
                    {"microphone_id": "left", "x_m": -0.03, "y_m": 0},
                    {"microphone_id": "right", "x_m": 0.03, "y_m": 0},
                ],
                "reference_microphone_id": "left",
            },
            "expected_azimuth_deg": 0,
            "azimuth_tolerance_deg": 180,
        },
        ("audio_visual_association", "audio_event_set.v1+audio_localization.v1+visual_bearing_set.v1"): {
            "event": {
                "occurrence_id": "audio-1",
                "label": "speech",
                "confidence": 0.9,
                "event_time_interval": interval(0, 1),
                "source_id": "microphone",
                "provider_id": "fixture",
                "provider_version": "1",
            },
            "localization": {
                "localization_id": "loc-1",
                "source_id": "microphone",
                "array_id": "array",
                "event_time_interval": interval(0, 1),
                "azimuth_deg": 10,
                "confidence": 0.9,
            },
            "candidates": [
                {
                    "local_entity_id": "person-1",
                    "entity_type": "person",
                    "source_id": "camera",
                    "event_time_interval": interval(0, 1),
                    "azimuth_deg": 10,
                    "confidence": 0.9,
                }
            ],
            "expected_local_entity_id": "person-1",
        },
        ("historical_vehicle_interval_matcher", "no_external_input"): {
            "track_sets": [track_set(0, [track(1, 2, 0)])],
            "expected_scoped_track_id": leader,
        },
        ("historical_vehicle_interval_matcher", "track_set.v1"): {
            "track_sets": [track_set(0, [track(1, 2, 0)])],
            "expected_scoped_track_id": leader,
        },
        ("person_vehicle_relation_provider", "projected_track_set.v1"): {
            "track_sets": [
                track_set(0, [track(1, 0, 0)]),
                track_set(1, [track(1, 0, 1), track(3, 1, 1, "person")]),
                track_set(2, [track(1, 0, 2), track(3, 7, 2, "person")]),
            ],
            "minimum_stop_seconds": 0,
            "expected_predicate_id": "DISEMBARKS",
        },
        ("speaker_diarization_provider", "speaker_embedding_set.v1+speech_segment_set.v1"): {
            "segments": [
                {
                    "segment_id": "a",
                    "source_id": "microphone",
                    "event_time_interval": interval(0, 1),
                    "speech_probability": 0.9,
                    "embedding": [1.0, 0.0],
                },
                {
                    "segment_id": "b",
                    "source_id": "microphone",
                    "event_time_interval": interval(1, 2),
                    "speech_probability": 0.9,
                    "embedding": [0.0, 1.0],
                },
            ],
            "expected_speaker_count": 2,
        },
        ("multi_object_tracker", "detection_set.v1"): {
            "detection_frames": detection_frames,
            "minimum_track_count": 1,
        },
        ("yolo_full_context_960", "raw_video_frames.v1"): {
            "image_path": args.image_path,
            "model_path": args.small_model_path,
            "device": "cpu",
            "minimum_detection_count": 1,
        },
        ("yolo_vehicle_fast_640", "raw_video_frames.v1"): {
            "image_path": args.image_path,
            "model_path": args.nano_model_path,
            "device": "cpu",
            "minimum_detection_count": 1,
            "required_labels": ["car"],
        },
        ("package_detector", "raw_video_frames.v1"): {
            "image_path": args.image_path,
            "model_path": args.small_model_path,
            "device": "cpu",
            "minimum_package_count": 0,
        },
    }
    readiness = json.loads(args.readiness.read_text(encoding="utf-8"))
    exact_targets = {
        (row["target"]["provider_id"], row["target"]["input_class"]): row["target"]
        for row in readiness["targets"]
        if row["status"] == "READY_CONTAINER"
        and row["target"]["tier"] == "sensor"
    }
    args.output.mkdir(parents=True, exist_ok=True)
    emitted = []
    for key, payload in sorted(fixtures.items()):
        if key not in exact_targets:
            continue
        stem = f"{key[0]}__{key[1].replace('+', '__')}"
        target_path = args.output / f"{stem}.target.json"
        fixture_path = args.output / f"{stem}.fixture.json"
        target_path.write_text(
            json.dumps(exact_targets[key], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fixture_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        emitted.append(
            {
                "provider_id": key[0],
                "input_class": key[1],
                "target": str(target_path),
                "fixture": str(fixture_path),
            }
        )
    manifest = {
        "schema_version": "fable.e0_worker_fixture_set.v1",
        "fixture_count": len(emitted),
        "fixtures": emitted,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"fixture_count": len(emitted), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
