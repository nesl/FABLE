#!/usr/bin/env python3
"""Run a FABLE node agent.

The default ``runtime: dataflow`` mode executes recovered provider workers in
this node-agent process and connects them through the same-node StreamBus.
Terminal PredicateMatch/ReID results are sent asynchronously to the controller.

Example:

node:
  node_id: edge1
  node_type: edge
runtime: dataflow
controller_results:
  host: 10.0.0.10
  port: 8766
sources:
  camera3:
    type: video
    uri: rtsp://camera3/stream

For legacy/external provider commands, ``runtime: subprocess`` remains
available and uses the ``providers: <id>: {command: [...]}`` mapping.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import yaml
from replay import build_synchronized_replay, load_replay_manifest

from fable.execution import (
    DataflowProviderRuntime,
    GatedSourceAdapter,
    NodeAgent,
    NodeAgentTCPServer,
    OpenCVVideoSourceAdapter,
    SubprocessProviderRuntime,
    SystemResourceProbe,
    TcpResultTransport,
    WaveAudioSourceAdapter,
    DockerProviderRuntime,
    MqttResultTransport,
    MQTTVideoFrameRelay,
    MQTTAudioWindowRelay,
    MQTTVisionMetadataRelay,
    MQTTEventEvidenceRelay,
    NodeAgentMQTTService,
    PahoMQTTClient,
    HistoricalStore,
    StreamHistoryRecorder,
    parse_input_gates,
)
from fable.planning import NodeState


def _sources(raw: dict, gates: dict) -> tuple[dict, object | None]:
    output = {}
    for source_id, spec in raw.get("sources", {}).items():
        kind = str(spec.get("type", "video")).lower()
        if kind == "video":
            adapter = OpenCVVideoSourceAdapter(
                str(source_id), spec["uri"], realtime=bool(spec.get("realtime", True))
            )
        elif kind in {"wav", "audio"}:
            adapter = WaveAudioSourceAdapter(
                str(source_id), spec["path"],
                window_ms=int(spec.get("window_ms", 1000)),
                realtime=bool(spec.get("realtime", True)),
            )
        else:
            raise ValueError(f"unsupported source adapter type {kind!r}")
        output[str(source_id)] = (
            GatedSourceAdapter(adapter, gates[str(source_id)])
            if str(source_id) in gates else adapter
        )
    synchronized = None
    replay_manifest = raw.get("replay_manifest")
    if replay_manifest:
        synchronized = build_synchronized_replay(load_replay_manifest(str(replay_manifest)))
        duplicate = set(output).intersection(synchronized.adapters)
        if duplicate:
            raise ValueError(f"sources and replay manifest duplicate source IDs: {sorted(duplicate)}")
        for source_id, adapter in synchronized.adapters.items():
            output[source_id] = (
                GatedSourceAdapter(adapter, gates[source_id])
                if source_id in gates else adapter
            )
    return output, synchronized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--replay-manifest", type=Path, help="override config replay_manifest for this session")
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.replay_manifest is not None:
        raw["replay_manifest"] = str(args.replay_manifest)
    node_raw = raw["node"]
    node = NodeState(
        str(node_raw["node_id"]),
        str(node_raw.get("node_type", "unknown")),
        bool(node_raw.get("available", True)),
        float(node_raw.get("cpu_free", float("inf"))),
        float(node_raw.get("memory_mb_free", float("inf"))),
        float(node_raw.get("gpu_memory_mb_free", float("inf"))),
    )

    runtime_kind = str(raw.get("runtime", "dataflow")).lower()
    input_gates = parse_input_gates(raw.get("input_gates"))
    mqtt_raw = raw.get("mqtt", {})
    transport_kind = str(raw.get("transport", "tcp")).lower()
    result_client = None
    historical_store = None
    history_recorder = None
    synchronized_replay = None
    source_relay = None
    audio_relay = None
    vision_relay = None
    evidence_relay = None
    pinned_source_ids: tuple[str, ...] = ()
    prewarm_provider_ids: tuple[str, ...] = ()
    if runtime_kind in {"dataflow", "bridged_dataflow"}:
        if transport_kind == "mqtt":
            result_client = PahoMQTTClient(
                str(mqtt_raw.get("host", "localhost")), int(mqtt_raw.get("port", 1883)),
                f"fable-results-{node.node_id}",
            )
            result_transport = MqttResultTransport(result_client)
        else:
            result_raw = raw.get("controller_results")
            if not isinstance(result_raw, dict):
                raise ValueError("TCP dataflow agents require controller_results.host/port")
            result_transport = TcpResultTransport(
                str(result_raw["host"]), int(result_raw.get("port", 8766))
            )
        source_adapters, synchronized_replay = _sources(raw, input_gates)
        local_runtime = DataflowProviderRuntime(
            result_transport=result_transport,
            source_adapters=source_adapters,
        )
        prewarm_provider_ids = tuple(str(value) for value in raw.get("prewarm_providers", ()))
        vision_relay_raw = raw.get("mqtt_vision_metadata_relay", {})
        vision_relay_enabled = not (
            isinstance(vision_relay_raw, dict)
            and vision_relay_raw.get("enabled") is False
        )
        if transport_kind == "mqtt" and vision_relay_enabled:
            vision_relay = MQTTVisionMetadataRelay(
                result_client, local_runtime.bus, node_id=node.node_id,
            )
            vision_relay.start()
        evidence_raw = raw.get("mqtt_event_evidence")
        if transport_kind == "mqtt" and isinstance(evidence_raw, dict) and evidence_raw.get("enabled", False):
            evidence_relay = MQTTEventEvidenceRelay(
                result_client, local_runtime.bus, node_id=node.node_id,
                local_source_ids=tuple(source_adapters),
                jpeg_quality=int(evidence_raw.get("jpeg_quality", 80)),
                minimum_interval_seconds=float(evidence_raw.get("minimum_interval_seconds", 2.0)),
                metrics_path=os.environ.get("FABLE_EVIDENCE_METRICS_FILE"),
                include_track_observations=(
                    os.environ.get("FABLE_EVIDENCE_ALL_TRACKS", "0") == "1"
                ),
            )
            evidence_relay.start()
        relay_raw = raw.get("mqtt_video_relay")
        if transport_kind == "mqtt" and isinstance(relay_raw, dict) and relay_raw.get("enabled", False):
            source_relay = MQTTVideoFrameRelay(
                result_client, local_runtime.bus, node_id=node.node_id,
                local_source_ids=tuple(source_adapters),
                jpeg_quality=int(relay_raw.get("jpeg_quality", 85)),
                max_fps=(
                    float(relay_raw["max_fps"])
                    if relay_raw.get("max_fps") is not None else None
                ),
            )
            source_relay.start()
            audio_relay_raw = raw.get("mqtt_audio_relay", {})
            if isinstance(audio_relay_raw, dict) and audio_relay_raw.get("enabled", False):
                audio_relay = MQTTAudioWindowRelay(
                    result_client, local_runtime.bus, node_id=node.node_id,
                    local_source_ids=tuple(source_adapters),
                )
                audio_relay.start()
            pinned_source_ids = tuple(source_adapters)
            local_runtime.pin_sources(pinned_source_ids)
        history_raw = raw.get("historical_store")
        if isinstance(history_raw, dict) and history_raw.get("database"):
            historical_store = HistoricalStore(str(history_raw["database"]))
            history_recorder = StreamHistoryRecorder(
                historical_store, local_runtime.bus,
                tuple(str(value) for value in history_raw.get(
                    "evidence_types", ("detection_frame", "track_frame")
                )),
            )
        if runtime_kind == "dataflow":
            provider_runtime = local_runtime
        else:
            from deployment.external_provider import ExternalProviderBridgeRuntime

            bridge_raw = raw.get("external_provider_bridge")
            if not isinstance(bridge_raw, dict):
                raise ValueError("bridged_dataflow requires external_provider_bridge")
            commands = {
                str(provider_id): tuple(spec["command"])
                for provider_id, spec in bridge_raw.get("providers", {}).items()
                if isinstance(spec, dict) and isinstance(spec.get("command"), list)
            }
            if not commands:
                raise ValueError("external_provider_bridge requires provider commands")
            provider_runtime = ExternalProviderBridgeRuntime(
                local_runtime,
                commands,
                cwd=bridge_raw.get("cwd"),
                ready_timeout_seconds=float(bridge_raw.get("ready_timeout_seconds", 30)),
                input_gates=input_gates,
            )
    elif runtime_kind == "subprocess":
        commands = {
            str(provider_id): tuple(spec["command"])
            for provider_id, spec in raw.get("providers", {}).items()
            if isinstance(spec, dict) and "command" in spec
        }
        provider_runtime = SubprocessProviderRuntime(commands)
    elif runtime_kind == "docker":
        images = {
            str(provider_id): str(spec["image"])
            for provider_id, spec in raw.get("providers", {}).items()
            if isinstance(spec, dict) and "image" in spec
        }
        provider_runtime = DockerProviderRuntime(
            images,
            extra_args=tuple(str(value) for value in raw.get("docker", {}).get("extra_args", ())),
            extra_env=(
                {"FABLE_MQTT_HOST": str(mqtt_raw.get("host", "localhost")),
                 "FABLE_MQTT_PORT": str(mqtt_raw.get("port", 1883))}
                if transport_kind == "mqtt" else None
            ),
        )
    else:
        raise ValueError("runtime must be 'dataflow', 'bridged_dataflow', 'subprocess', or 'docker'")

    resource_probe = (
        SystemResourceProbe(node.node_id, node.node_type)
        if raw.get("system_probe", True)
        else None
    )
    agent = NodeAgent(node, provider_runtime, resource_probe=resource_probe)
    if result_client is not None:
        result_client.start()
    if transport_kind == "mqtt":
        command_client = PahoMQTTClient(
            str(mqtt_raw.get("host", "localhost")), int(mqtt_raw.get("port", 1883)),
            f"fable-agent-{node.node_id}",
        )
        service = NodeAgentMQTTService(agent, command_client)
        service.start()
        server = None
        print(f"FABLE node agent {node.node_id} connected to MQTT")
    else:
        service = None
        server = NodeAgentTCPServer(agent, args.host, args.port)
        print(f"FABLE node agent {node.node_id} listening on {server.address[0]}:{server.address[1]}")
    if prewarm_provider_ids:
        print("Pre-warming providers: " + ", ".join(prewarm_provider_ids), flush=True)
        local_runtime.prewarm(prewarm_provider_ids)
        print("Provider pre-warm complete", flush=True)
    if synchronized_replay is not None:
        barrier_file = os.environ.get("FABLE_REPLAY_START_BARRIER_FILE")
        if barrier_file:
            print(f"Waiting for replay start barrier: {barrier_file}", flush=True)
            while not Path(barrier_file).exists():
                time.sleep(0.05)
        synchronized_replay.start()
        print(f"Replay {synchronized_replay.replay_id} ready with synchronized source barrier")
    try:
        if server is not None:
            server.serve_forever()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.close()
        if service is not None:
            service.close()
        if source_relay is not None:
            source_relay.close()
        if audio_relay is not None:
            audio_relay.close()
        if vision_relay is not None:
            vision_relay.close()
        if evidence_relay is not None:
            evidence_relay.close()
        if pinned_source_ids:
            local_runtime.unpin_sources(pinned_source_ids)
        if result_client is not None:
            result_client.stop()
        if history_recorder is not None:
            history_recorder.close()
        if historical_store is not None:
            historical_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
