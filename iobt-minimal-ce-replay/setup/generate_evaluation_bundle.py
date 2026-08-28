#!/usr/bin/env python3
"""Generate one internally consistent multi-node FABLE evaluation bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

REPLAY_ROOT = Path(__file__).resolve().parents[1]
FABLE_ROOT = REPLAY_ROOT.parent
sys.path.insert(0, str(FABLE_ROOT))

from evaluation.mobile_recordings import load_alias_map, resolve_mobile_recordings
from evaluation.gpu_partition import (
    GpuPartition,
    pin_compose_service,
    resolve_gpu_partition,
    validate_evaluation_bundle,
)

VEHICLE_PROVIDERS = (
    "yolo_vehicle_fast_640",
    "yolo_full_context_960",
    "multi_object_tracker",
    "camera_projection",
    "route_map_matcher",
    "pass_reference_evaluator",
    "zone_membership_evaluator",
    "zone_transition_evaluator",
    "track_lifecycle_exit_evaluator",
    "pairwise_distance_evaluator",
    "motion_state_evaluator",
    "follows_local_geometry",
)
MULTIMODAL_PROVIDERS = (
    "audio_event_classifier",
    "gcc_phat_audio_localizer",
    "audio_visual_association",
    "voice_activity_detector",
    "speaker_embedding_provider",
    "speaker_diarization_provider",
    "keyword_or_asr_provider",
    "conversation_provider",
    "person_proximity_provider",
    "person_vehicle_relation_provider",
    "package_detector",
    "interaction_evidence_analyzer",
    "object_transfer_reasoner",
)
RUNTIME_TEMPLATE = REPLAY_ROOT / "config/fable_provider_runtimes.yaml"
REID_MANIFEST = REPLAY_ROOT / "models/reid/models.json"
E0_DESKTOP_PROFILES = (
    FABLE_ROOT / "evaluation/manifests/providers/calibrated_desktop_profiles.json"
)


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def rq3a_provider_profiles(output: Path) -> None:
    """Write explicit logical-tier timings without altering E0 measurements.

    E0 was measured on one x86 host, so its sensor and server rows are
    physically identical. RQ3a runs the same images on that host but models
    the intended embedded sensor versus site-edge processing tiers. Keep the
    assumptions in the generated bundle so they are auditable per run.
    """

    document = json.loads(E0_DESKTOP_PROFILES.read_text(encoding="utf-8"))
    # TrackLifecycleExitEvaluator is an uncalibrated, lower-compute variant of
    # the measured zone-transition evaluator.  Preserve every measured tier
    # row and make the derivation explicit instead of silently falling back to
    # an unmeasured generic profile (profile files are authoritative when
    # supplied to the orchestrator).
    lifecycle_rows = []
    for source in document.get("profiles", []):
        if source.get("provider_id") != "zone_transition_evaluator":
            continue
        derived = json.loads(json.dumps(source))
        derived["provider_id"] = "track_lifecycle_exit_evaluator"
        derived.setdefault("metadata", {}).update(
            {
                "profile_derivation": "zone_transition_evaluator",
                "profile_derivation_reason": (
                    "same typed transition logic without projection or route lookup"
                ),
            }
        )
        lifecycle_rows.append(derived)
    document.setdefault("profiles", []).extend(lifecycle_rows)
    detector_ids = {
        "yolo_vehicle_fast_640",
        "yolo_vehicle_balanced_960",
        "yolo_full_context_960",
    }
    for row in document.get("profiles", []):
        if row.get("provider_id") not in detector_ids:
            continue
        node_class = row.get("node_class")
        factor = 6.0 if node_class == "sensor" else 0.5 if node_class == "server" else 1.0
        startup_factor = 2.0 if node_class == "sensor" else 0.5 if node_class == "server" else 1.0
        row["cold_start_samples_ms"] = [
            value * startup_factor for value in row.get("cold_start_samples_ms", [])
        ]
        row["warm_execution_samples_ms"] = [
            value * factor for value in row.get("warm_execution_samples_ms", [])
        ]
        row.setdefault("metadata", {}).update(
            {
                "rq3a_logical_tier_assumption": True,
                "rq3a_execution_multiplier": factor,
                "rq3a_startup_multiplier": startup_factor,
                "rq3a_source_profile_sha256": sha256(E0_DESKTOP_PROFILES),
            }
        )
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def node_id(folder: str) -> str:
    if folder.startswith("orin"):
        return f"dvpg_gq_orin_{int(folder.removeprefix('orin'))}"
    return folder


def utc_text(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        # Fixed-site replay filenames/catalog timestamps are local EST.  The
        # labels and FABLE event-time contracts are UTC.
        parsed = parsed.replace(tzinfo=timezone(timedelta(hours=-5)))
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def selected_nodes(
    catalog: Path,
    scenario: str,
    maximum: int | None,
    requested: list[str] | None = None,
) -> list[str]:
    row = selected_scenario(catalog, scenario)
    available = sorted({str(item) for item in row.get("nodes", ())})
    if requested:
        invalid = sorted(set(requested) - set(available))
        if invalid:
            raise ValueError(
                f"scenario {scenario} does not contain requested nodes: "
                + ", ".join(invalid)
            )
        nodes = list(dict.fromkeys(requested))
    else:
        nodes = available
    if maximum is not None:
        nodes = nodes[:maximum]
    if not nodes:
        raise ValueError(f"scenario {scenario} has no selected nodes")
    return nodes


def selected_scenario(catalog: Path, scenario: str) -> dict[str, object]:
    document = json.loads(catalog.read_text(encoding="utf-8"))
    for row in document.get("scenarios", ()):
        if row.get("scenario_id") == scenario:
            return row
    raise ValueError(f"unknown scenario: {scenario}")


def adopted(
    *,
    container: str,
    readiness_topic: str,
    output_topics: list[str],
    adapter: str,
) -> dict[str, object]:
    return {
        "mode": "ADOPT_EXISTING",
        "container_name": container,
        # Evaluation providers are demand driven.  Compose creates their
        # containers, but the node agent owns their running lifetime: an
        # activation lease starts the container and the last lease release
        # stops it.  This prevents unselected alternatives from consuming the
        # replay stream and contaminating resource measurements.
        "stop_adopted_when_idle": True,
        "readiness": {
            "mqtt_topic": readiness_topic,
            "ready_field": "ready",
            "ready_value": True,
            "timeout_ms": 60_000,
        },
        "output_topics": output_topics,
        "output_adapter": adapter,
    }


def provider_overlay(
    nodes: list[str],
    output: Path,
    *,
    audio_backend: str = "spectral-rule",
    enable_reid: bool = False,
    vision_only_nodes: set[str] | None = None,
    gpu_partition: GpuPartition | None = None,
) -> None:
    vision_only_nodes = vision_only_nodes or set()
    gpu_partition = gpu_partition or resolve_gpu_partition()
    reid_models = (
        json.loads(REID_MANIFEST.read_text(encoding="utf-8"))["models"]
        if enable_reid
        else {}
    )
    services: dict[str, object] = {
        "mqtt": {
            "volumes": [
                f"{REPLAY_ROOT / 'server/mosquitto.fable.conf'}:/mosquitto/config/mosquitto.conf:ro",
                "fable-mosquitto-data:/mosquitto/data",
            ],
            "restart": "unless-stopped",
        },
        "fable-mongo": {
            "image": "mongo:7",
            "container_name": "fable-mongo",
            "command": ["mongod", "--bind_ip_all", "--wiredTigerCacheSizeGB", "1"],
            "volumes": ["fable-mongo-data:/data/db"],
            "healthcheck": {
                "test": ["CMD", "mongosh", "--quiet", "--eval", "db.runCommand({ping:1}).ok"],
                "interval": "5s",
                "timeout": "3s",
                "retries": 20,
            },
            "restart": "unless-stopped",
        },
        "fable-orchestrator": {
            "build": {
                "context": str(FABLE_ROOT),
                "dockerfile": "iobt-minimal-ce-replay/services/orchestration/fable_orchestrator/Dockerfile",
            },
            "image": "fable/orchestrator:phase6",
            "container_name": "fable-orchestrator",
            "environment": {
                "MQTT_HOST_IP": "mqtt",
                "MQTT_PORT": 1883,
                "MONGODB_URI": "mongodb://fable-mongo:27017",
                "MONGODB_DATABASE": "fable",
                "FABLE_ORCHESTRATOR_ID": "orchestrator",
                "FABLE_DEPLOYMENT_CONFIG": "/workspace/bundle/fable_deployment.yaml",
                "FABLE_RUNTIME_CONFIG": "/workspace/bundle/fable_provider_runtimes.yaml",
                "FABLE_ARTIFACT_CONFIG": "/workspace/bundle/fable_deployment_artifacts.yaml",
                "FABLE_PROVIDER_PROFILES": "/workspace/bundle/rq3a_provider_profiles.json",
                "FABLE_PROVIDER_STARTUP_OVERRIDES_JSON": "${FABLE_PROVIDER_STARTUP_OVERRIDES_JSON:-{}}",
                "FABLE_EXECUTION_PROFILE": "${FABLE_EXECUTION_PROFILE:-real}",
                "FABLE_ALLOW_REMOTE_REFERENCE": "false",
                # Deployment artifacts are copied into the generated bundle so
                # the orchestrator can verify them without depending on a host
                # checkout being mounted into the trusted runtime.
                "FABLE_REPOSITORY_ROOT": "/workspace/bundle",
                "FABLE_CAPTURE_E2_SNAPSHOTS": "${FABLE_CAPTURE_E2_SNAPSHOTS:-false}",
                "FABLE_BEAM_WIDTH": "${FABLE_BEAM_WIDTH:-8}",
                "FABLE_RUN_ONLINE_ORACLE": "${FABLE_RUN_ONLINE_ORACLE:-false}",
                "FABLE_JOINT_RESOURCE_EPOCH_PLANNING": "${FABLE_JOINT_RESOURCE_EPOCH_PLANNING:-0}",
                "FABLE_RETROSPECTIVE_POLICY": "${FABLE_RETROSPECTIVE_POLICY:-R2_FABLE_TYPED_REPLAY}",
                "FABLE_STATE_DIR": "/var/lib/fable/orchestrator",
                "FABLE_MONITOR_INTERVAL_SEC": 1.0,
                # Video inference can temporarily delay best-effort MQTT
                # heartbeats without implying node loss. Explicit evaluation
                # resource/fault notifications remain immediate; this only
                # prevents false liveness transitions under sustained load.
                "FABLE_HEARTBEAT_SUSPECT_MISSES": "10",
                "FABLE_HEARTBEAT_UNAVAILABLE_MISSES": "30",
                "FABLE_FANOUT_BATCH_SIZE": "${FABLE_FANOUT_BATCH_SIZE:-0}",
                "FABLE_FANOUT_BATCH_INTERVAL_SECONDS": "${FABLE_FANOUT_BATCH_INTERVAL_SECONDS:-12}",
                "FABLE_REQUIRE_HEARTBEAT_ON_RESTART": "false",
            },
            "volumes": [
                f"{output.parent}:/workspace/bundle:ro",
                (
                    f"{FABLE_ROOT / 'evaluation/manifests/baselines/static_pipelines.yaml'}:"
                    "/workspace/FABLE/evaluation/manifests/baselines/static_pipelines.yaml:ro"
                ),
                "fable-orchestrator-state:/var/lib/fable/orchestrator",
            ],
            "depends_on": {
                "mqtt": {"condition": "service_started"},
                "fable-mongo": {"condition": "service_healthy"},
            },
            "restart": "unless-stopped",
        },
        "fable-agent-x86server": {
            "build": {
                "context": str(FABLE_ROOT),
                "dockerfile": "iobt-minimal-ce-replay/services/agent/fable_node_agent/Dockerfile",
            },
            "image": "fable/node-agent:phase6",
            "container_name": "fable-agent-x86server",
            "environment": {
                "MQTT_HOST_IP": "mqtt",
                "MQTT_PORT": 1883,
                "FABLE_NODE_ID": "x86server",
                "FABLE_MQTT_CLIENT_ID": "fable-agent-x86server",
                "FABLE_STATE_DIR": "/var/lib/fable/x86server",
                "FABLE_CONTAINER_RUNTIME": "docker",
                "FABLE_ALLOW_FAULT_INJECTION": "true",
                "FABLE_CLOSE_LIVE_EVIDENCE_AT_REPLAY_END": "${FABLE_CLOSE_LIVE_EVIDENCE_AT_REPLAY_END:-false}",
            },
            "volumes": [
                "/var/run/docker.sock:/var/run/docker.sock",
                "fable-agent-x86server-state:/var/lib/fable/x86server",
            ],
            "depends_on": {
                "mqtt": {"condition": "service_started"},
                "fable-orchestrator": {"condition": "service_started"},
            },
            "restart": "unless-stopped",
        },
        "fable-identity-x86server": {
            "build": {
                "context": str(FABLE_ROOT),
                "dockerfile": "iobt-minimal-ce-replay/services/analytics/fable_vehicle_stack/Dockerfile",
            },
            "image": "fable/vehicle-stack:phase7",
            "container_name": "fable-identity-x86server",
            "command": ["python", "-u", "-m", "providers.vehicle.identity_service"],
            "environment": {
                "MQTT_HOST": "mqtt",
                "MQTT_PORT": 1883,
                "FABLE_REID_MAXIMUM_EVENT_TIME_GAP_SEC": "${FABLE_REID_MAXIMUM_EVENT_TIME_GAP_SEC:-300}",
                # Evaluation identity demands use a strictly bounded fallback
                # (at most ten calls per replay) after local ReID cannot decide.
                "FABLE_VLM_REID_ENABLED": "${FABLE_VLM_REID_ENABLED:-true}",
                "FABLE_VLM_REID_MODEL": "${FABLE_VLM_REID_MODEL:-gpt-4o-mini-2024-07-18}",
                "FABLE_VLM_REID_MAX_CALLS": "${FABLE_VLM_REID_MAX_CALLS:-10}",
                "FABLE_VLM_REID_MIN_CONFIDENCE": "${FABLE_VLM_REID_MIN_CONFIDENCE:-0.50}",
                "FABLE_IDENTITY_ESCALATION_POLICY": "${FABLE_IDENTITY_ESCALATION_POLICY:-}",
                "FABLE_RETROSPECTIVE_POLICY": "${FABLE_RETROSPECTIVE_POLICY:-R2_FABLE_TYPED_REPLAY}",
                "FABLE_RAW_RETROSPECTIVE_STAGGER_SECONDS": "${FABLE_RAW_RETROSPECTIVE_STAGGER_SECONDS:-0}",
                "OPENAI_API_KEY": "${OPENAI_API_KEY:-}",
                "FABLE_VLM_PROXY_URL": "unix:///run/fable-vlm/proxy.sock",
            },
            "volumes": ["fable-vlm-runtime:/run/fable-vlm"],
            "depends_on": {
                "mqtt": {"condition": "service_started"},
                "fable-vlm-cloud1": {"condition": "service_healthy"},
            },
            "restart": "unless-stopped",
        },
        "fable-reid-x86server": {
            "build": {
                "context": str(FABLE_ROOT),
                "dockerfile": "iobt-minimal-ce-replay/services/analytics/site_reid/Dockerfile",
            },
            "image": "fable/site-reid:phase1",
            "container_name": "fable-reid-x86server",
            "command": ["python", "-u", "-m", "providers.vehicle.descriptor_service"],
            "environment": {
                "MQTT_HOST": "mqtt",
                "MQTT_PORT": 1883,
                "LOG_LEVEL": "INFO",
                "VEHICLE_REID_DEVICE": "${FABLE_SITE_REID_DEVICE:-cuda:0}",
                "VEHICLE_REID_MODEL_PATH": "/models/reid/vehicle.pth",
                "VEHICLE_REID_CONFIG": "/app/reid/fastreid_veri_sbs_r50_ibn.yaml",
                "VEHICLE_REID_MODEL_ID": "fastreid:sbs_R50_ibn:vehicle",
                "VEHICLE_REID_MODEL_VERSION": "${VEHICLE_REID_MODEL_VERSION:-veri-wild-sbs-r50-ibn-v1}",
                "VEHICLE_REID_PREPROCESSING_ID": "fastreid-veri-256x256-rgb",
            },
            "volumes": [f"{REPLAY_ROOT / 'models/reid'}:/models/reid:ro"],
            "depends_on": {"mqtt": {"condition": "service_started"}},
            "restart": "unless-stopped",
        },
        "fable-vlm-cloud1": {
            "image": "fable/vehicle-stack:phase7",
            "container_name": "fable-vlm-cloud1",
            "command": [
                "python", "-u", "-m", "providers.vehicle.vlm_proxy",
                "--unix-socket", "/run/fable-vlm/proxy.sock",
            ],
            "environment": {
                "FABLE_VLM_REID_MODEL": "${FABLE_VLM_REID_MODEL:-gpt-4o-mini-2024-07-18}",
                "FABLE_VLM_REID_MAX_CALLS": "${FABLE_VLM_REID_MAX_CALLS:-10}",
                "FABLE_VLM_DEBUG_DIR": "/var/lib/fable/vlm-debug",
                "OPENAI_API_KEY": "${OPENAI_API_KEY:-}",
            },
            "volumes": [
                "${FABLE_VLM_DEBUG_HOST_DIR:-../debug/vlm_requests}:/var/lib/fable/vlm-debug",
                "fable-vlm-runtime:/run/fable-vlm",
            ],
            "healthcheck": {
                "test": [
                    "CMD",
                    "python",
                    "-c",
                    (
                        "import socket; "
                        "s=socket.socket(socket.AF_UNIX); s.settimeout(2); "
                        "s.connect('/run/fable-vlm/proxy.sock'); "
                        "s.sendall(b'GET /healthz HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n'); "
                        "assert b' 200 ' in s.recv(64)"
                    ),
                ],
                "interval": "5s",
                "timeout": "3s",
                "retries": 20,
            },
            "restart": "unless-stopped",
        },
        "fable-agent-cloud1": {
            "image": "fable/node-agent:phase6",
            "container_name": "fable-agent-cloud1",
            "environment": {
                "MQTT_HOST_IP": "mqtt",
                "MQTT_PORT": 1883,
                "FABLE_NODE_ID": "cloud1",
                "FABLE_MQTT_CLIENT_ID": "fable-agent-cloud1",
                "FABLE_STATE_DIR": "/var/lib/fable/cloud1",
                "FABLE_CONTAINER_RUNTIME": "docker",
            },
            "volumes": [
                "/var/run/docker.sock:/var/run/docker.sock",
                "fable-agent-cloud1-state:/var/lib/fable/cloud1",
            ],
            "depends_on": {
                "mqtt": {"condition": "service_started"},
                "fable-orchestrator": {"condition": "service_started"},
                "fable-vlm-cloud1": {"condition": "service_healthy"},
            },
            "restart": "unless-stopped",
        },
    }
    volumes: dict[str, object] = {
        "fable-mosquitto-data": {},
        "fable-mongo-data": {},
        "fable-orchestrator-state": {},
        "fable-agent-x86server-state": {},
        "fable-agent-cloud1-state": {},
        "fable-vlm-runtime": {},
    }
    for folder in nodes:
        nid = node_id(folder)
        mobile_node = folder.startswith("mobile_archive_")
        tmp = f"/tmp/iobt-{folder}"
        vehicle = f"fable-vehicle-{folder}"
        multimodal = f"fable-multimodal-{folder}"
        agent = f"fable-agent-{folder}"
        services[vehicle] = {
            "build": {
                "context": str(FABLE_ROOT),
                "dockerfile": "iobt-minimal-ce-replay/services/analytics/fable_vehicle_stack/Dockerfile",
            },
            "image": "fable/vehicle-stack:phase7",
            "container_name": vehicle,
            "environment": {
                "MQTT_HOST": "mqtt",
                "MQTT_PORT": 1883,
                "MQTT_CLIENT_ID": f"fable-vehicle-{folder}",
                "SOURCE_ID": nid,
                "YOLO_TOPIC": f"/{nid}/analytics/yolo/bbox",
                "VEHICLE_TRACK_TOPIC": f"/{nid}/fable/vehicle/tracks",
                "TRACK_TOPIC": f"/{nid}/fable/vehicle/tracks",
                "PREDICATE_TOPIC": f"/{nid}/fable/vehicle/predicates",
                "READINESS_TOPIC": f"/readiness/{nid}/fable_vehicle",
                "VEHICLE_GEOMETRY_CONFIG": "/workspace/replay/config/fable_vehicle_geometry.json",
                "TRACKER_ALGORITHM": "bytetrack",
                "TRACKER_FRAME_RATE": "${TRACKER_FRAME_RATE:-5.0}",
                "VEHICLE_TRACK_ACTIVATION_THRESHOLD": "${VEHICLE_TRACK_ACTIVATION_THRESHOLD:-0.25}",
                "VEHICLE_HIGH_CONFIDENCE_THRESHOLD": "${VEHICLE_HIGH_CONFIDENCE_THRESHOLD:-0.30}",
                "VEHICLE_MINIMUM_CONSECUTIVE_FRAMES": "${VEHICLE_MINIMUM_CONSECUTIVE_FRAMES:-2}",
                "VEHICLE_CLASS_AGNOSTIC_NMS_IOU_THRESHOLD": "${VEHICLE_CLASS_AGNOSTIC_NMS_IOU_THRESHOLD:-0.85}",
            },
            "volumes": [
                f"{REPLAY_ROOT / 'config'}:/workspace/replay/config:ro",
                (
                    f"{output.parent / 'fable_vehicle_geometry.json'}:"
                    "/workspace/replay/config/fable_vehicle_geometry.json:ro"
                ),
            ],
            "depends_on": {
                "mqtt": {"condition": "service_started"},
                f"yolo-{folder}": {"condition": "service_started"},
            },
            "restart": "unless-stopped",
        }
        if mobile_node:
            # Mobile replay inference is intentionally sparse (roughly
            # 0.4 Hz under the shared desktop GPU). Requiring two consecutive
            # detector samples means a valid vehicle may leave the view before
            # ByteTrack ever exposes it. A detector-backed first sample is a
            # valid camera-local candidate; later identity/temporal predicates
            # still impose their own continuity requirements.
            services[vehicle]["environment"][
                "VEHICLE_MINIMUM_CONSECUTIVE_FRAMES"
            ] = "1"
        if enable_reid:
            services[f"yolo-{folder}"] = {
                "environment": {
                    "REID_ENABLED": "true",
                    "PERSON_REID_BACKEND": "torchreid",
                    "PERSON_REID_MODEL_NAME": "osnet_ain_x1_0",
                    "PERSON_REID_MODEL_PATH": "/models/reid/person.pth",
                    "PERSON_REID_MODEL_VERSION": reid_models["person"]["version"],
                    "PERSON_REID_PREPROCESSING_ID": reid_models["person"][
                        "preprocessing_id"
                    ],
                    "VEHICLE_REID_MODEL_PATH": "/models/reid/vehicle.pth",
                    "VEHICLE_REID_BACKEND": "fastreid",
                    "VEHICLE_REID_MODEL_NAME": "sbs_R50_ibn",
                    "VEHICLE_REID_CONFIG": (
                        "/app/reid/fastreid_veri_sbs_r50_ibn.yaml"
                    ),
                    "VEHICLE_REID_MODEL_VERSION": reid_models["vehicle"]["version"],
                    "VEHICLE_REID_PREPROCESSING_ID": reid_models["vehicle"][
                        "preprocessing_id"
                    ],
                },
                "volumes": [
                    f"{REPLAY_ROOT / 'models/reid'}:/models/reid:ro",
                ],
            }
        if folder not in vision_only_nodes:
            services[multimodal] = {
            "build": {
                "context": str(FABLE_ROOT),
                "dockerfile": "iobt-minimal-ce-replay/services/analytics/fable_multimodal_stack/Dockerfile",
                "args": {"FABLE_MODEL_EXTRAS": "${FABLE_MODEL_EXTRAS:-0}"},
            },
            "image": "fable/multimodal-stack:phase8",
            "container_name": multimodal,
            "environment": {
                "TEST_CONTROL": "False",
                "MCP_NODE_NAME": nid,
                "MQTT_HOST_IP": "mqtt",
                "MQTT_PORT": 1883,
                "MCP_CONTAINER_OUTPUT_DIR": "/tmp",
                "SERIALIZER": "msgpack",
                "SOURCE_ID": nid,
                "AUDIO_SOURCE_ID": f"{folder}_microphone",
                "YOLO_TOPIC": f"/{nid}/analytics/yolo/bbox",
                "VEHICLE_TRACK_TOPIC": f"/{nid}/fable/vehicle/tracks",
                "AUDIO_EVENT_TOPIC": f"/{nid}/fable/audio/events",
                "AUDIO_LOCALIZATION_TOPIC": f"/{nid}/fable/audio/localizations",
                "SPEECH_TURN_TOPIC": f"/{nid}/fable/audio/speaker_turns",
                "CONTEXT_TRACK_TOPIC": f"/{nid}/fable/context/tracks",
                "INTERACTION_TOPIC": f"/{nid}/fable/interactions/predicates",
                "CUSTODY_TOPIC": f"/{nid}/fable/interactions/custody",
                "READINESS_TOPIC": f"/readiness/{nid}/fable_multimodal",
                "FABLE_AUDIO_GEOMETRY": "/workspace/replay/config/fable_multimodal_geometry.json",
                "AUDIO_CHANNEL_INDICES": "1,2,3,4",
                "TRACKER_ALGORITHM": "bytetrack",
                "FABLE_VISUAL_IMAGE_WIDTH_PX": "${FABLE_VISUAL_IMAGE_WIDTH_PX:-1280}",
                "FABLE_CAMERA_HORIZONTAL_FOV_DEG": "${FABLE_CAMERA_HORIZONTAL_FOV_DEG:-90}",
                "FABLE_VISUAL_BEARING_OFFSET_DEG": "${FABLE_VISUAL_BEARING_OFFSET_DEG:-0}",
                "FABLE_VISUAL_ZONE_ID": "${FABLE_VISUAL_ZONE_ID:-front}",
                "FABLE_AV_TIME_TOLERANCE_SECONDS": "${FABLE_AV_TIME_TOLERANCE_SECONDS:-0.5}",
                "FABLE_AUDIO_BACKEND": audio_backend,
                "FABLE_YAMNET_MODEL_HANDLE": "${FABLE_YAMNET_MODEL_HANDLE:-/models/yamnet}",
                "FABLE_YAMNET_CLASS_MAP": "${FABLE_YAMNET_CLASS_MAP:-/models/yamnet_class_map.csv}",
                "FABLE_YAMNET_MODEL_VERSION": "${FABLE_YAMNET_MODEL_VERSION:-}",
                "FABLE_YAMNET_TEMPORAL_POOLING": "${FABLE_YAMNET_TEMPORAL_POOLING:-max}",
            },
            "volumes": [
                f"{REPLAY_ROOT / 'config'}:/workspace/replay/config:ro",
                f"{tmp}:/tmp",
                f"{REPLAY_ROOT / 'models'}:/models:ro",
            ],
            "depends_on": {
                f"yolo-{folder}": {"condition": "service_started"},
                (
                    f"mobile-{folder}"
                    if mobile_node
                    else f"respeaker-{folder}"
                ): {"condition": "service_started"},
            },
            "restart": "unless-stopped",
            }
            if mobile_node:
                services[multimodal]["environment"].update(
                    {
                        "AUDIO_CHANNEL_INDICES": "0",
                        "AUDIO_SAMPLE_RATE_HZ": "16000",
                        "FABLE_AUDIO_GEOMETRY": "",
                        # Mobile YOLO runs at roughly 0.4 Hz when all replay
                        # analytics share the GPU. Preserve a sustained pair
                        # across that measured cadence without changing the
                        # spatial proximity threshold or required duration.
                        "FABLE_PERSON_PROXIMITY_MAXIMUM_FRAME_GAP_SECONDS": "5.0",
                    }
                )
        volume_name = f"fable-agent-{folder}-state"
        volumes[volume_name] = {}
        # Every agent may host an adopted evaluator that consumes typed
        # evidence from another deployed sensor. Give each agent the complete
        # deployment alias table so it can validate global MQTT evidence
        # against PredicateDemand.eligible_source_ids without trusting an
        # arbitrary topic name.
        aliases = {}
        for source_folder in nodes:
            source_node_id = node_id(source_folder)
            aliases.update(
                {
                    f"zed:{source_node_id}": f"{source_folder}_camera",
                    f"camera:{source_node_id}": f"{source_folder}_camera",
                }
            )
            if source_folder not in vision_only_nodes:
                aliases.update(
                    {
                        f"respeaker:{source_node_id}": (
                            f"{source_folder}_microphone"
                        ),
                        f"audio:{source_node_id}": f"{source_folder}_microphone",
                    }
                )
        services[agent] = {
            "build": {
                "context": str(FABLE_ROOT),
                "dockerfile": "iobt-minimal-ce-replay/services/agent/fable_node_agent/Dockerfile",
            },
            "image": "fable/node-agent:phase6",
            "container_name": agent,
            "environment": {
                "MQTT_HOST_IP": "mqtt",
                "MQTT_PORT": 1883,
                "FABLE_NODE_ID": nid,
                "FABLE_MQTT_CLIENT_ID": f"fable-agent-{folder}",
                "FABLE_STATE_DIR": f"/var/lib/fable/{nid}",
                "FABLE_CONTAINER_RUNTIME": "docker",
                "FABLE_ALLOW_FAULT_INJECTION": "true",
                "FABLE_CLOSE_LIVE_EVIDENCE_AT_REPLAY_END": "${FABLE_CLOSE_LIVE_EVIDENCE_AT_REPLAY_END:-false}",
                "FABLE_SOURCE_ALIASES_JSON": json.dumps(aliases, separators=(",", ":")),
            },
            "volumes": [
                "/var/run/docker.sock:/var/run/docker.sock",
                f"{volume_name}:/var/lib/fable/{nid}",
            ],
            "depends_on": {
                "mqtt": {"condition": "service_started"},
                "fable-orchestrator": {"condition": "service_started"},
            },
            "restart": "unless-stopped",
        }
    # One trusted site-edge worker follows the primary selected camera. A
    # provider runtime is node-scoped, so multi-camera requests transfer compact
    # evidence to this worker rather than creating ambiguous duplicate runtime
    # identities on x86server.
    primary_folder = next(
        (folder for folder in nodes if folder.startswith("orin")), nodes[0]
    )
    primary_node_id = node_id(primary_folder)
    services[f"yolo-edge-{primary_folder}"] = {
        "build": {
            "context": str(REPLAY_ROOT),
            "dockerfile": "services/analytics/yolo_detector/Dockerfile",
        },
        "image": "iobt-minimal/yolo-detector:latest",
        "container_name": f"yolo-edge-{primary_folder}",
        "working_dir": "/app",
        "command": "python3 -u /app/app.py",
        "gpus": "all",
        "environment": {
            "TEST_CONTROL": "False",
            "MCP_NODE_NAME": "x86server",
            "MQTT_HOST_IP": "mqtt",
            "MQTT_PORT": 1883,
            "MCP_CONTAINER_OUTPUT_DIR": "/tmp",
            "SERIALIZER": "msgpack",
            "SOURCE": primary_node_id,
            "LOAD_MODEL": "true",
            "YOLO_MODEL": "/app/yolov8s.pt",
            "YOLO_DEVICE": "${YOLO_DEVICE:-auto}",
            "YOLO_OUTPUT_TOPIC": (
                f"/x86server/from/{primary_node_id}/analytics/yolo/bbox"
            ),
            "YOLO_PUBLISH_STATUS": "true",
            "YOLO_STATUS_TOPIC": "/debug/x86server/analytics/yolo/status",
        },
        "depends_on": {
            "mqtt": {"condition": "service_started"},
            f"zed-{primary_folder}": {"condition": "service_started"},
        },
        "restart": "unless-stopped",
    }
    # Site-local detector alternatives and identity inference share GPU 1.
    for service_name in (
        f"yolo-edge-{primary_folder}",
        "fable-identity-x86server",
        "fable-reid-x86server",
        "fable-agent-x86server",
    ):
        pin_compose_service(
            services[service_name],
            gpu_uuid=gpu_partition.site_gpu_uuid,
            tier="site_local",
        )
    # Sensor-local YOLO, tracking/ReID facets, multimodal inference, and node
    # telemetry share GPU 0. Replay decoding is pinned separately below.
    for folder in nodes:
        for service_name in (
            f"fable-vehicle-{folder}",
            f"fable-multimodal-{folder}", f"fable-agent-{folder}",
        ):
            if service_name in services:
                pin_compose_service(
                    services[service_name],
                    gpu_uuid=gpu_partition.device_gpu_uuid,
                    tier="device",
                )
    services[f"fable-vehicle-x86server-{primary_folder}"] = {
        "image": "fable/vehicle-stack:phase7",
        "container_name": f"fable-vehicle-x86server-{primary_folder}",
        "environment": {
            "MQTT_HOST": "mqtt",
            "MQTT_PORT": 1883,
            "MQTT_CLIENT_ID": f"fable-vehicle-x86server-{primary_folder}",
            "SOURCE_ID": primary_node_id,
            "YOLO_TOPIC": (
                f"/x86server/from/{primary_node_id}/analytics/yolo/bbox"
            ),
            "TRACK_TOPIC": (
                f"/x86server/from/{primary_node_id}/fable/vehicle/tracks"
            ),
            "PREDICATE_TOPIC": (
                f"/x86server/from/{primary_node_id}/fable/vehicle/predicates"
            ),
            "READINESS_TOPIC": (
                f"/readiness/x86server/fable_vehicle_{primary_folder}"
            ),
            "TRACKER_ALGORITHM": "bytetrack",
            "TRACKER_FRAME_RATE": "${TRACKER_FRAME_RATE:-5.0}",
            "VEHICLE_GEOMETRY_CONFIG": (
                "/workspace/replay/config/fable_vehicle_geometry.json"
            ),
        },
        "volumes": [
            f"{REPLAY_ROOT / 'config'}:/workspace/replay/config:ro",
            (
                f"{output.parent / 'fable_vehicle_geometry.json'}:"
                "/workspace/replay/config/fable_vehicle_geometry.json:ro"
            ),
        ],
        "depends_on": {
            "mqtt": {"condition": "service_started"},
            f"yolo-edge-{primary_folder}": {"condition": "service_started"},
        },
        "restart": "unless-stopped",
    }
    # This service is created after the general site-local pinning loop above.
    # Keep its Compose assignment identical to the x86server runtime contract;
    # otherwise an adopted tracker is repeatedly rejected by the node agent.
    pin_compose_service(
        services[f"fable-vehicle-x86server-{primary_folder}"],
        gpu_uuid=gpu_partition.site_gpu_uuid,
        tier="site_local",
    )
    output.write_text(
        yaml.safe_dump({"services": services, "volumes": volumes}, sort_keys=False),
        encoding="utf-8",
    )


def vehicle_geometry(nodes: list[str], output: Path) -> None:
    """Write camera-frame geometry valid for every selected replay node.

    The fixed Orin replay archives and physical Pi-to-Jetson path preserve the
    native 1920x1080 image coordinates. Mobile archives are normalized to the
    portrait 720x1280 contract. Each node needs its own coordinate-frame
    identifier; a geometry artifact scoped only to orin11 silently disables
    ENTERS/EXITS everywhere else.
    """

    zones = []
    for folder in nodes:
        frame = f"image:{node_id(folder)}"
        width, height = (1920.0, 1080.0) if folder.startswith("orin") else (720.0, 1280.0)
        zones.append(
            {
                "zone_id": f"{folder}_camera_fov",
                "coordinate_frame_id": frame,
                "polygon": [
                    {"x": 0.0, "y": 0.0, "coordinate_frame_id": frame},
                    {"x": width, "y": 0.0, "coordinate_frame_id": frame},
                    {"x": width, "y": height, "coordinate_frame_id": frame},
                    {"x": 0.0, "y": height, "coordinate_frame_id": frame},
                ],
            }
        )
    output.write_text(
        json.dumps(
            {
                "schema_version": "fable.vehicle_geometry.v1",
                "description": (
                    "Generated replay camera-FOV zones. Uncalibrated PASSES is "
                    "derived from detector-backed track lifecycle traversal; "
                    "no surveyed line geometry is claimed."
                ),
                "references": [],
                "zones": zones,
                "routes": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def deployment(
    nodes: list[str],
    output: Path,
    *,
    event_start: str | None = None,
    event_end: str | None = None,
    vision_only_nodes: set[str] | None = None,
) -> None:
    vision_only_nodes = vision_only_nodes or set()
    node_rows: dict[str, object] = {}
    sources: dict[str, object] = {}
    links = []
    for folder in nodes:
        nid = node_id(folder)
        node_rows[nid] = {
            "resource_pool_id": "device_gpu0",
            "node_class": "sensor",
            "region": "scenario_replay",
            "capacity": {"cpu_cores": 8, "memory_mb": 16384, "gpu_memory_mb": 8192},
            "capabilities": (
                ["vision", "gpu", "replay"]
                if folder in vision_only_nodes
                else ["vision", "audio", "gpu", "replay"]
            ),
            "policy_tags": ["raw-local"],
        }
        sources[f"{folder}_camera"] = {
            "node_id": nid,
            "region": "scenario_replay",
            "modalities": ["vision"],
            "live_data_types": ["raw_video_frames.v1"],
            "coverage_regions": ["scenario_replay"],
        }
        if folder not in vision_only_nodes:
            sources[f"{folder}_microphone"] = {
                "node_id": nid,
                "region": "scenario_replay",
                "modalities": ["audio"],
                "live_data_types": ["audio_segment.v1"],
                "coverage_regions": ["scenario_replay"],
            }
        if event_start and event_end:
            interval = {"start": event_start, "end": event_end}
            sources[f"{folder}_camera"]["raw_buffer_interval"] = interval
            if folder not in vision_only_nodes:
                sources[f"{folder}_microphone"]["raw_buffer_interval"] = interval
        links.append(
            {
                "source_node_id": nid,
                "target_node_id": "x86server",
                "latency_ms": 2,
                "bandwidth_mbps": 1000,
                "bidirectional": True,
            }
        )
    node_rows["x86server"] = {
        "resource_pool_id": "site_gpu1",
        "node_class": "server",
        "region": "server",
        "capacity": {"cpu_cores": 16, "memory_mb": 65536, "gpu_memory_mb": 24576},
        "capabilities": [
            "vision",
            "audio",
            "gpu",
            "orchestration",
            "replay",
            "site_identity",
        ],
    }
    node_rows["cloud1"] = {
        "resource_pool_id": "desktop_cpu",
        "node_class": "server",
        "region": "cloud",
        "capacity": {"cpu_cores": 16, "memory_mb": 65536, "gpu_memory_mb": 0},
        "capabilities": ["vision", "audio", "orchestration", "replay", "hosted_vlm"],
    }
    sources["replay_gps"] = {
        "node_id": "x86server",
        "region": "scenario_replay",
        "modalities": ["location"],
        "live_data_types": ["gps_fix.v1"],
    }
    links.append(
        {
            "source_node_id": "x86server",
            "target_node_id": "cloud1",
            "latency_ms": 25,
            "bandwidth_mbps": 200,
            "bidirectional": True,
        }
    )
    output.write_text(
        yaml.safe_dump(
            {
                "schema_version": "fable.deployment.v1",
                "resource_pools": {
                    "device_gpu0": {
                        "capacity": {
                            "cpu_cores": 16,
                            "memory_mb": 32768,
                            "gpu_memory_mb": 24576,
                        }
                    },
                    "site_gpu1": {
                        "capacity": {
                            "cpu_cores": 16,
                            "memory_mb": 32768,
                            "gpu_memory_mb": 24576,
                        }
                    },
                    "desktop_cpu": {
                        "capacity": {
                            "cpu_cores": 16,
                            "memory_mb": 65536,
                            "gpu_memory_mb": 0,
                        }
                    }
                },
                "nodes": node_rows,
                "sources": sources,
                "links": links,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _replace_template_strings(value: object, old: str, new: str) -> object:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_template_strings(item, old, new) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_template_strings(item, old, new)
            for key, item in value.items()
        }
    return value


def runtimes(
    nodes: list[str],
    output: Path,
    *,
    enable_reid: bool = False,
    vision_only_nodes: set[str] | None = None,
    gpu_partition: GpuPartition | None = None,
) -> None:
    vision_only_nodes = vision_only_nodes or set()
    gpu_partition = gpu_partition or resolve_gpu_partition()
    template = yaml.safe_load(RUNTIME_TEMPLATE.read_text(encoding="utf-8"))
    source = template["nodes"]["dvpg_gq_orin_11"]
    rows = {}
    for folder in nodes:
        nid = node_id(folder)
        rendered = _replace_template_strings(source, "dvpg_gq_orin_11", nid)
        rendered = _replace_template_strings(rendered, "orin11", folder)
        if folder in vision_only_nodes:
            for provider_id in MULTIMODAL_PROVIDERS:
                rendered["providers"].pop(provider_id, None)
        # The replay image implements the detector quality variants through
        # one leaseable YOLO worker. Advertise each typed provider identity so
        # chains may select a quality profile without requiring duplicate
        # always-on containers.
        detector = rendered["providers"].get("yolo_vehicle_fast_640")
        if detector is not None:
            detection_topic = f"/{nid}/analytics/yolo/bbox"
            detector["artifact_topic_outputs"] = {
                "detection_set.v1": detection_topic
            }
            for provider_id in ("yolo_vehicle_balanced_960", "yolo_full_context_960"):
                rendered["providers"].setdefault(
                    provider_id, copy.deepcopy(detector)
                )
            tracker = rendered["providers"].get("multi_object_tracker")
            if tracker is not None:
                tracker["artifact_topic_inputs"] = {
                    "detection_set.v1": detection_topic
                }
                if enable_reid:
                    # The identity chain executes crop extraction and descriptor
                    # inference inside the same lease-controlled vehicle worker.
                    # Advertising only the site association runtime leaves a
                    # logically valid SAME_ENTITY frontier with zero executable
                    # physical alternatives.
                    crop_runtime = copy.deepcopy(tracker)
                    crop_runtime.pop("artifact_topic_inputs", None)
                    crop_runtime["output_topics"] = [
                        f"/{nid}/fable/identity/bounded-crops"
                    ]
                    crop_runtime["artifact_topic_outputs"] = {
                        "image_crop_set.v1": f"/{nid}/fable/identity/bounded-crops"
                    }
                    crop_runtime["artifact_broker_scope_id"] = "evaluation-mqtt"
                    crop_runtime["output_adapter"] = "NONE"
                    rendered["providers"]["track_crop_extractor"] = crop_runtime
            # Historical recovery executes over the same replayed track stream
            # as the tracker.  Do not advertise the calibration-only
            # multimodal stub as an executable runtime.
            matcher = rendered["providers"].get("historical_vehicle_interval_matcher")
            vehicle_runtime = rendered["providers"].get("multi_object_tracker")
            if matcher is not None and vehicle_runtime is not None:
                matcher.update(copy.deepcopy(vehicle_runtime))
                matcher["output_topics"] = [f"/{nid}/fable/vehicle/predicates"]
                matcher["output_adapter"] = "VEHICLE_PREDICATE"
        rows[nid] = rendered
    if "x86server" in template["nodes"]:
        primary_folder = next(
            (folder for folder in nodes if folder.startswith("orin")), nodes[0]
        )
        primary_node_id = node_id(primary_folder)
        rows["x86server"] = _replace_template_strings(
            template["nodes"]["x86server"], "dvpg_gq_orin_11", primary_node_id
        )
        rows["x86server"] = _replace_template_strings(
            rows["x86server"], "orin11", primary_folder
        )
        # The site-edge detector is a real raw-offload alternative, not a
        # calibration stub.  During physical contention the Jetson publishes
        # typed orin camera frames and stops local inference; these runtimes
        # preserve that source while moving the detector/tracker chain to
        # x86server.  Leaving the template REFERENCE runtimes in place made the
        # planner unable to select the already-provisioned edge workers.
        edge_detection_topic = (
            f"/x86server/from/{primary_node_id}/analytics/yolo/bbox"
        )
        edge_predicate_topic = (
            f"/x86server/from/{primary_node_id}/fable/vehicle/predicates"
        )
        edge_yolo = adopted(
            container=f"yolo-edge-{primary_folder}",
            readiness_topic="/readiness/x86server/yolo",
            output_topics=[edge_detection_topic],
            adapter="YOLO_OBJECT_PRESENT",
        )
        edge_yolo["artifact_topic_outputs"] = {
            "detection_set.v1": edge_detection_topic
        }
        edge_yolo["artifact_broker_scope_id"] = "evaluation-mqtt"
        rows["x86server"]["providers"]["yolo_vehicle_fast_640"] = edge_yolo

        edge_vehicle = adopted(
            container=f"fable-vehicle-x86server-{primary_folder}",
            readiness_topic=(
                f"/readiness/x86server/fable_vehicle_{primary_folder}"
            ),
            output_topics=[edge_predicate_topic],
            adapter="VEHICLE_PREDICATE",
        )
        edge_vehicle["worker_id"] = f"edge-vehicle-worker-{primary_folder}"
        edge_vehicle["artifact_topic_inputs"] = {
            "detection_set.v1": edge_detection_topic
        }
        edge_vehicle["artifact_broker_scope_id"] = "evaluation-mqtt"
        for provider_id in (
            "camera_projection",
            "multi_object_tracker",
            "pass_reference_evaluator",
        ):
            runtime = copy.deepcopy(edge_vehicle)
            if provider_id != "pass_reference_evaluator":
                runtime.pop("output_topics", None)
                runtime["output_adapter"] = "NONE"
            rows["x86server"]["providers"][provider_id] = runtime
        if enable_reid:
            rows["x86server"]["providers"]["vehicle_reid_descriptor"] = adopted(
                container="fable-reid-x86server",
                readiness_topic="/readiness/x86server/fable_reid_descriptor",
                output_topics=["/+/fable/identity/descriptors"],
                adapter="NONE",
            )
            rows["x86server"]["providers"]["vehicle_reid_descriptor"][
                "artifact_topic_inputs"
            ] = {"image_crop_set.v1": "/+/fable/identity/bounded-crops"}
            rows["x86server"]["providers"]["vehicle_reid_descriptor"][
                "artifact_topic_outputs"
            ] = {"vehicle_reid_embedding_set.v1": "/+/fable/identity/descriptors"}
            rows["x86server"]["providers"]["vehicle_reid_descriptor"][
                "artifact_broker_scope_id"
            ] = "evaluation-mqtt"
            rows["x86server"]["providers"]["cross_sensor_identity_association"] = adopted(
                container="fable-identity-x86server",
                readiness_topic="/readiness/x86server/fable_identity",
                output_topics=["/fable/identity/associations"],
                adapter="IDENTITY_ASSOCIATION",
            )
            rows["x86server"]["providers"]["cross_sensor_identity_association"][
                "artifact_topic_inputs"
            ] = {"vehicle_reid_embedding_set.v1": "/+/fable/identity/descriptors"}
            rows["x86server"]["providers"]["cross_sensor_identity_association"][
                "artifact_broker_scope_id"
            ] = "evaluation-mqtt"
        else:
            rows["x86server"]["providers"].pop(
                "cross_sensor_identity_association", None
            )
        detector = rows["x86server"]["providers"].get("yolo_vehicle_fast_640")
        if detector is not None:
            for provider_id in ("yolo_vehicle_balanced_960", "yolo_full_context_960"):
                rows["x86server"]["providers"].setdefault(
                    provider_id, copy.deepcopy(detector)
                )
    if "cloud1" in template["nodes"]:
        rows["cloud1"] = template["nodes"]["cloud1"]
    for node in rows.values():
        for provider_id, runtime in node.get("providers", {}).items():
            if runtime.get("mode") == "ADOPT_EXISTING":
                # FastReID model loading is materially longer than the late
                # SAME_ENTITY frontier's remaining deadline. Keep the worker
                # model-resident, but work-idle: it cannot process anything
                # until the identity service publishes a bounded crop request.
                # The hosted-VLM proxy is likewise an idle control endpoint:
                # keeping its HTTP/Unix-socket server resident does not invoke
                # the external model.  Stopping it before the downstream
                # identity worker requests a fallback breaks typed replay.
                # Every other inference provider remains lease-controlled.
                runtime["stop_adopted_when_idle"] = (
                    provider_id
                    not in {
                        "vehicle_reid_descriptor",
                        "hosted_vlm_identity_comparator",
                    }
                )
    for node_id_value, node in rows.items():
        assigned_uuid = (
            gpu_partition.site_gpu_uuid
            if node_id_value == "x86server"
            else gpu_partition.device_gpu_uuid
            if node_id_value.startswith("dvpg_gq_orin_")
            or node_id_value.startswith("mobile_")
            else None
        )
        if assigned_uuid:
            for runtime in node.get("providers", {}).values():
                if runtime.get("mode") != "REFERENCE":
                    runtime["gpu_device_ids"] = [assigned_uuid]
    output.write_text(
        yaml.safe_dump(
            {"schema_version": "fable.provider_runtimes.v1", "nodes": rows},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def artifacts(nodes: list[str], output: Path) -> None:
    rows = []
    for folder in nodes:
        nid = node_id(folder)
        rows.extend(
            (
                {
                    "artifact_type": "camera_calibration.v1",
                    "path": "fable_vehicle_geometry.json",
                    "runtime_uri": "file:///workspace/replay/config/fable_vehicle_geometry.json",
                    "node_id": nid,
                    "bindings": {
                        "source_id": f"{folder}_camera",
                        "coordinate_frame_id": "world",
                    },
                    # Deployment metadata is copied into every generated
                    # provider bundle. Model that replicated availability so
                    # trusted-edge offload can consume the originating
                    # camera's geometry without treating calibration as raw
                    # sensor data tied to the camera host.
                    "access_modes": ["LOCAL", "REMOTE_REFERENCE", "TRANSFERRED"],
                },
                {
                    "artifact_type": "route_graph.v1",
                    "path": "fable_vehicle_geometry.json",
                    "runtime_uri": "file:///workspace/replay/config/fable_vehicle_geometry.json",
                    "node_id": nid,
                    "bindings": {
                        "deployment_id": "scenario_replay",
                        # Route/reference geometry is camera-specific.  Without
                        # this binding the bounded Cartesian input search can
                        # pair video from one camera with another camera's
                        # local route graph, producing only unplaceable plans.
                        "source_id": f"{folder}_camera",
                    },
                    "access_modes": ["LOCAL", "REMOTE_REFERENCE", "TRANSFERRED"],
                },
            )
        )
    output.write_text(
        yaml.safe_dump(
            {"schema_version": "fable.deployment_artifacts.v1", "artifacts": rows},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def add_mobile_replay_services(
    compose_path: Path,
    recordings,
    *,
    scenario: str,
) -> list[str]:
    """Add MP4 replay and standard YOLO services for selected mobile archives."""

    document = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
    services = document.setdefault("services", {})
    folders: list[str] = []
    for recording in recordings:
        folder = (
            recording.logical_id
            if recording.logical_id.startswith("mobile_")
            else f"mobile_{recording.logical_id}"
        )
        if folder in folders:
            raise ValueError(f"duplicate resolved mobile logical identity: {folder}")
        folders.append(folder)
        nid = node_id(folder)
        tmp = f"/tmp/iobt-{folder}"
        replay_service = f"mobile-{folder}"
        segments = recording.segments or ()
        if len(segments) > 1:
            manifest_path = compose_path.parent / f"{folder}-segments.json"
            timeline_start = recording.timeline_start or recording.recording_start
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "fable.mobile_replay_segments.v1",
                        "timeline_start_epoch": timeline_start.timestamp(),
                        "segments": [
                            {
                                "video": f"/recording/segment-{index}.mp4",
                                "recording_start_epoch": item.recording_start.timestamp(),
                                "trim_start_seconds": item.trim_start_seconds,
                                "trim_end_seconds": item.trim_end_seconds,
                            }
                            for index, item in enumerate(segments)
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            replay_command = [
                "python",
                "-u",
                "/app/app.py",
                "--segment-manifest",
                "/recording/segments.json",
                "--scenario",
                scenario,
            ]
            replay_volumes = [
                f"{manifest_path}:/recording/segments.json:ro",
                *[
                    f"{item.path}:/recording/segment-{index}.mp4:ro"
                    for index, item in enumerate(segments)
                ],
                f"{tmp}:/tmp",
            ]
        else:
            timeline_start = recording.timeline_start or recording.recording_start
            replay_command = [
                "python",
                "-u",
                "/app/app.py",
                "--video",
                "/recording/input.mp4",
                "--scenario",
                scenario,
                "--start",
                str(round(recording.trim_start_seconds, 6)),
                "--end",
                str(round(recording.trim_end_seconds, 6)),
                "--recording-start-epoch",
                str(recording.recording_start.timestamp()),
                "--timeline-start-epoch",
                str(timeline_start.timestamp()),
            ]
            replay_volumes = [
                f"{recording.path}:/recording/input.mp4:ro",
                f"{tmp}:/tmp",
            ]
        services[replay_service] = {
            "build": {
                "context": str(REPLAY_ROOT),
                "dockerfile": "services/replay/mobile/Dockerfile",
            },
            "image": "iobt-minimal/mobile-replay:latest",
            "container_name": f"mobile-replay-{folder}",
            "command": replay_command,
            "environment": {
                "TEST_CONTROL": "False",
                "MCP_NODE_NAME": nid,
                "MQTT_HOST_IP": "mqtt",
                "MQTT_PORT": 1883,
                "MCP_CONTAINER_OUTPUT_DIR": "/tmp",
                "SERIALIZER": "msgpack",
            },
            "volumes": replay_volumes,
            "depends_on": {"mqtt": {"condition": "service_started"}},
        }
        services[f"yolo-{folder}"] = {
            "build": {
                "context": str(REPLAY_ROOT),
                "dockerfile": "services/analytics/yolo_detector/Dockerfile",
            },
            "image": "iobt-minimal/yolo-detector:latest",
            "container_name": f"yolo-detector-{folder}",
            # Mobile services are appended after the base replay Compose is
            # loaded, so they must request the NVIDIA runtime explicitly.
            # Without this field torch correctly falls back to CPU.
            "gpus": "all",
            "environment": {
                "TEST_CONTROL": "False",
                "MCP_NODE_NAME": nid,
                "MQTT_HOST_IP": "mqtt",
                "MQTT_PORT": 1883,
                "MCP_CONTAINER_OUTPUT_DIR": "/tmp",
                "SERIALIZER": "msgpack",
                "SOURCE": "local",
                "LOAD_MODEL": "true",
                "YOLO_MODEL": "/app/yolov8s.pt",
                "YOLO_DEVICE": "${YOLO_DEVICE:-auto}",
                "YOLO_PUBLISH_STATUS": "true",
                "YOLO_STATUS_TOPIC": f"/debug/{nid}/analytics/yolo/status",
            },
            "volumes": [f"{tmp}:/tmp"],
            "depends_on": {replay_service: {"condition": "service_started"}},
        }
    compose_path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return folders


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help=(
            "Optional resource cap. By default every available replay node is "
            "included so multi-camera event evidence is not silently omitted."
        ),
    )
    parser.add_argument(
        "--node",
        action="append",
        dest="nodes",
        help="Select an exact scenario node; repeat for multiple cameras.",
    )
    parser.add_argument(
        "--catalog", type=Path, default=REPLAY_ROOT / "generated/scenario_catalog.json"
    )
    parser.add_argument(
        "--enable-reid",
        action="store_true",
        help=(
            "Enable calibrated person and vehicle ReID inference. Requires "
            "the pinned person and vehicle checkpoints provisioned by "
            "setup/provision_reid_models.py."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--mobile-recording-prefix",
        help=(
            "Enable mobile MP4 replay for an exact archive prefix, for example "
            "spatial_ce1_1 or temporal_ce2_all."
        ),
    )
    parser.add_argument(
        "--mobile-timestamp-only",
        action="store_true",
        help=(
            "Select Android files named recording__<epoch_ms>.mp4. This is "
            "the archive format used by the 2025 mobile 4–6 recordings."
        ),
    )
    parser.add_argument(
        "--mobile-prefixed-any",
        action="store_true",
        help=(
            "Match legacy prefixed Android recordings by epoch interval, "
            "without assuming that the filename's experiment label maps "
            "one-to-one to the authored scenario."
        ),
    )
    parser.add_argument(
        "--mobile-optional",
        action="store_true",
        help=(
            "Continue with fixed sensors when no mobile file covers the selected "
            "scenario. Intended for mixed-coverage campaign suites."
        ),
    )
    parser.add_argument(
        "--mobile-root",
        type=Path,
        default=Path("/media/brianw/Extreme SSD3"),
    )
    parser.add_argument(
        "--mobile-alias-map",
        type=Path,
        help=(
            "Optional run-specific JSON mapping mobile_archive_N to a topology "
            "alias. Without it, stable archive identities are used."
        ),
    )
    parser.add_argument(
        "--mobile-event-start",
        help=(
            "Optional site-local ISO timestamp for mobile matching. Use the "
            "ground-truth event interval when the replay scenario envelope is broader."
        ),
    )
    parser.add_argument(
        "--mobile-event-end",
        help="Site-local ISO end timestamp paired with --mobile-event-start.",
    )
    parser.add_argument(
        "--audio-backend",
        choices=("spectral-rule", "yamnet"),
        default="spectral-rule",
        help=(
            "Audio classifier. spectral-rule is dependency-light smoke-test mode; "
            "yamnet requires models/yamnet and models/yamnet_class_map.csv."
        ),
    )
    args = parser.parse_args()
    gpu_partition = resolve_gpu_partition()
    scenario = selected_scenario(args.catalog, args.scenario)
    if args.audio_backend == "yamnet":
        missing = [
            path
            for path in (
                REPLAY_ROOT / "models/yamnet",
                REPLAY_ROOT / "models/yamnet_class_map.csv",
            )
            if not path.exists()
        ]
        if missing:
            parser.error(
                "YAMNet requested but required local model artifacts are missing: "
                + ", ".join(str(path) for path in missing)
            )
    if args.enable_reid:
        manifest = json.loads(REID_MANIFEST.read_text(encoding="utf-8"))
        invalid = []
        for entity_kind, model in manifest["models"].items():
            path = REID_MANIFEST.parent / model["filename"]
            if not path.is_file() or sha256(path) != model["sha256"]:
                invalid.append(f"{entity_kind}={path}")
        if invalid:
            parser.error(
                "ReID requested but pinned checkpoints are missing or invalid: "
                + ", ".join(invalid)
                + "; run setup/provision_reid_models.py"
            )
    nodes = selected_nodes(
        args.catalog,
        args.scenario,
        args.max_nodes,
        requested=args.nodes,
    )
    output = (
        args.output_dir
        or REPLAY_ROOT / "generated/evaluation_bundles" / args.scenario
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    replay_command = [
            sys.executable,
            str(REPLAY_ROOT / "setup/generate_replay_compose.py"),
            "--compose-out",
            str(output / "compose.replay.yaml"),
            # The executable site-edge YOLO alternative consumes the ZED
            # camera stream over MQTT.  The persistent replay stack disables
            # those (low-rate) RGB/depth topics by default, which makes a plan
            # placed on x86server look executable while it actually receives
            # no frames.  Evaluation bundles include that edge alternative,
            # so expose the corresponding 1 Hz transport stream here.
            "--debug-raw-mqtt",
        ]
    # generate_replay_compose now produces scenario-agnostic supervisors. The
    # bundle generator already resolved the exact scenario node set above, so
    # pass that set through the stable --device boundary instead of forwarding
    # the removed scenario-selection CLI.
    replay_command.extend(
        argument for node in nodes for argument in ("--device", node)
    )
    subprocess.run(
        replay_command,
        check=True,
    )
    replay_document = yaml.safe_load(
        (output / "compose.replay.yaml").read_text(encoding="utf-8")
    )
    for name, service in replay_document.get("services", {}).items():
        if name.startswith("zed-") or name.startswith("yolo-"):
            pin_compose_service(
                service,
                gpu_uuid=gpu_partition.device_gpu_uuid,
                tier="device",
            )
    (output / "compose.replay.yaml").write_text(
        yaml.safe_dump(replay_document, sort_keys=False), encoding="utf-8"
    )
    mobile_recordings = ()
    mobile_nodes: list[str] = []
    mobile_modes = sum(
        bool(item)
        for item in (
            args.mobile_recording_prefix,
            args.mobile_timestamp_only,
            args.mobile_prefixed_any,
        )
    )
    if mobile_modes > 1:
        parser.error(
            "mobile recording selection modes are mutually exclusive"
        )
    if args.mobile_recording_prefix or args.mobile_timestamp_only or args.mobile_prefixed_any:
        if bool(args.mobile_event_start) != bool(args.mobile_event_end):
            parser.error(
                "--mobile-event-start and --mobile-event-end must be supplied together"
            )
        start_text = str(
            args.mobile_event_start
            or scenario.get("observed_start_datetime")
            or scenario.get("start_datetime")
            or ""
        )
        end_text = str(
            args.mobile_event_end or scenario.get("observed_end_datetime") or ""
        )
        if not start_text or not end_text:
            parser.error("scenario lacks an observed interval for mobile matching")
        mobile_recordings = resolve_mobile_recordings(
            args.mobile_root,
            recording_prefix=args.mobile_recording_prefix,
            allow_any_prefix=args.mobile_prefixed_any,
            event_start=datetime.fromisoformat(start_text),
            event_end=datetime.fromisoformat(end_text),
            alias_map=load_alias_map(args.mobile_alias_map),
        )
        if not mobile_recordings and not args.mobile_optional:
            parser.error(
                "no mobile recording covers the scenario interval for "
                + (
                    f"prefix {args.mobile_recording_prefix}"
                    if args.mobile_recording_prefix
                    else "timestamp-only filenames"
                )
            )
        if mobile_recordings:
            mobile_nodes = add_mobile_replay_services(
                output / "compose.replay.yaml",
                mobile_recordings,
                scenario=args.scenario,
            )
            replay_document = yaml.safe_load(
                (output / "compose.replay.yaml").read_text(encoding="utf-8")
            )
            for folder in mobile_nodes:
                service = replay_document.get("services", {}).get(f"yolo-{folder}")
                if service is not None:
                    pin_compose_service(
                        service,
                        gpu_uuid=gpu_partition.device_gpu_uuid,
                        tier="device",
                    )
            (output / "compose.replay.yaml").write_text(
                yaml.safe_dump(replay_document, sort_keys=False), encoding="utf-8"
            )
    all_nodes = nodes + mobile_nodes
    canonical_event_start = utc_text(
        scenario.get("observed_start_datetime") or scenario.get("start_datetime")
    )
    if canonical_event_start:
        replay_document = yaml.safe_load(
            (output / "compose.replay.yaml").read_text(encoding="utf-8")
        )
        for folder in nodes:
            for service_name in (f"zed-{folder}", f"respeaker-{folder}"):
                service = replay_document.get("services", {}).get(service_name)
                if service is None:
                    continue
                environment = service.setdefault("environment", [])
                if isinstance(environment, list):
                    environment.append(f"FABLE_REPLAY_EVENT_START={canonical_event_start}")
                else:
                    environment["FABLE_REPLAY_EVENT_START"] = canonical_event_start
        (output / "compose.replay.yaml").write_text(
            yaml.safe_dump(replay_document, sort_keys=False), encoding="utf-8"
        )
    # Mobile MP4s contain synchronized audio tracks. Treat selected mobile
    # archives as audiovisual nodes so their multimodal providers are actually
    # generated and their person/audio predicates are deployable.
    vision_only: set[str] = set()
    provider_overlay(
        all_nodes,
        output / "compose.fable.providers.yaml",
        audio_backend=args.audio_backend,
        enable_reid=args.enable_reid,
        vision_only_nodes=vision_only,
        gpu_partition=gpu_partition,
    )
    vehicle_geometry(all_nodes, output / "fable_vehicle_geometry.json")
    deployment(
        all_nodes,
        output / "fable_deployment.yaml",
        event_start=canonical_event_start,
        event_end=utc_text(scenario.get("observed_end_datetime")),
        vision_only_nodes=vision_only,
    )
    runtimes(
        all_nodes,
        output / "fable_provider_runtimes.yaml",
        enable_reid=args.enable_reid,
        vision_only_nodes=vision_only,
        gpu_partition=gpu_partition,
    )
    artifacts(all_nodes, output / "fable_deployment_artifacts.yaml")
    rq3a_provider_profiles(output / "rq3a_provider_profiles.json")
    gpu_partition_validation = validate_evaluation_bundle(output, gpu_partition)
    (output / "bundle.json").write_text(
        json.dumps(
            {
                "scenario": args.scenario,
                "nodes": all_nodes,
                "fixed_nodes": nodes,
                "mobile_nodes": mobile_nodes,
                "mobile_recordings": [
                    {
                        "archive_id": item.archive_id,
                        "logical_id": item.logical_id,
                        "path": str(item.path),
                        "trim_start_seconds": item.trim_start_seconds,
                        "trim_end_seconds": item.trim_end_seconds,
                        "segments": [
                            {
                                "path": str(segment.path),
                                "recording_start": segment.recording_start.isoformat(),
                                "trim_start_seconds": segment.trim_start_seconds,
                                "trim_end_seconds": segment.trim_end_seconds,
                            }
                            for segment in item.segments
                        ],
                    }
                    for item in mobile_recordings
                ],
                "audio_backend": args.audio_backend,
                "evaluation_quality_audio": args.audio_backend == "yamnet",
                "reid_enabled": args.enable_reid,
                "gpu_partition": gpu_partition.as_dict(),
                "gpu_partition_validation": gpu_partition_validation,
                "files": sorted(
                    {item.name for item in output.iterdir()} | {"bundle.json"}
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), "scenario": args.scenario, "nodes": all_nodes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
