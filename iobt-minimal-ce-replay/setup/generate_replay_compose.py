#!/usr/bin/env python3
"""Generate a scenario-agnostic persistent replay compose file.

Unlike setup/generate_compose.py, this does not bake a scenario/date into the
compose file. It discovers node folder names once, mounts the parent data roots,
and starts persistent replay supervisors. The web UI later selects the scenario
by publishing /replay/config and /replay/sync.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

DEFAULT_DATA_PARENTS = [
    Path("/media/brianw/Extreme SSD/West Point Experimentation"),
    Path("/media/brianw/Extreme SSD/GQ Data"),
]
ENV_DATA_ROOTS = "IOBT_HOST_DATA_ROOTS"


def yaml_quote(value: str | Path) -> str:
    text = str(value)
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'


def parse_env_roots() -> list[Path]:
    raw = os.environ.get(ENV_DATA_ROOTS, "").strip()
    if not raw:
        return []
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]


def data_parent_candidates(cli_data_dirs: list[str] | None) -> list[Path]:
    raw: list[Path] = []
    if cli_data_dirs:
        raw.extend(Path(p).expanduser() for p in cli_data_dirs)
    else:
        raw.extend(parse_env_roots())
        raw.extend(DEFAULT_DATA_PARENTS)
    seen: set[str] = set()
    out: list[Path] = []
    for p in raw:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def mcp_hostname(node_folder: str) -> str:
    if node_folder.startswith("orin"):
        suffix = node_folder.replace("orin", "")
        return f"dvpg_gq_orin_{suffix}"
    return node_folder.replace("-", "_")


def discover_nodes(data_roots: Iterable[Path]) -> list[str]:
    nodes: set[str] = set()
    for root in data_roots:
        if not root.is_dir():
            continue
        for date_dir in root.iterdir():
            if not (date_dir.is_dir() and re.fullmatch(r"\d{8}", date_dir.name)):
                continue
            for child in date_dir.iterdir():
                if child.is_dir() and child.name != "GPS":
                    nodes.add(child.name)
    return sorted(nodes)


def emit_extra_hosts(lines: list[str], indent: str = "    ") -> None:
    lines.append(f"{indent}extra_hosts:\n")
    lines.append(f"{indent}  - \"host.docker.internal:host-gateway\"\n")


def emit_build(lines: list[str], dockerfile: str, build_network: str) -> None:
    lines.append("    build:\n")
    lines.append("      context: .\n")
    lines.append(f"      dockerfile: {dockerfile}\n")
    if build_network != "default":
        lines.append(f"      network: {build_network}\n")


def emit_env(lines: list[str], env: list[str], indent: str = "    ") -> None:
    lines.append(f"{indent}environment:\n")
    for item in env:
        lines.append(f"{indent}  - {item}\n")


def emit_data_root_volumes(lines: list[str], roots: list[Path], indent: str = "      ") -> None:
    aliases = ["west_point", "gq"]
    for i, root in enumerate(roots):
        alias = aliases[i] if i < len(aliases) else f"root{i+1}"
        lines.append(f"{indent}- {yaml_quote(str(root) + f':/data_roots/{alias}:ro')}\n")


def container_data_roots(roots: list[Path]) -> str:
    aliases = ["west_point", "gq"]
    vals = []
    for i, _root in enumerate(roots):
        alias = aliases[i] if i < len(aliases) else f"root{i+1}"
        vals.append(f"/data_roots/{alias}")
    return os.pathsep.join(vals)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate persistent replay compose file")
    ap.add_argument("--data-dir", action="append", default=None, help="Parent root containing <YYYYMMDD>/orin*/... data. Can be repeated.")
    ap.add_argument("--nodes", nargs="*", help="Node folders to include, e.g. orin11 orin12. If omitted, discover all nodes under data roots.")
    ap.add_argument("--device", "--only-device", action="append", default=None, help="Convenience alias for running only one or more node folders, e.g. --device orin11. Can be repeated.")
    ap.add_argument("--compose-out", default="compose.replay.yaml")
    ap.add_argument("--include-yolo", dest="include_yolo", action="store_true", default=None, help="Include local YOLO detector containers. This is now the default; kept for compatibility.")
    ap.add_argument("--no-yolo", dest="include_yolo", action="store_false", help="Do not include YOLO detector containers.")
    ap.add_argument("--load-yolo-model", default="true", choices=["true", "false"], help="Set true for actual YOLO inference. Default is true.")
    ap.add_argument("--yolo-debug", action="store_true", help="Preset for debugging YOLO on selected nodes: enable YOLO model, annotated images, frame/status debug, and disable ReSpeaker/audio, GPS, and CE.")
    ap.add_argument("--no-audio-detector", action="store_true", help="Do not include local audio detector containers.")
    ap.add_argument("--no-zed", action="store_true", help="Do not include ZED replay supervisors.")
    ap.add_argument("--no-respeaker", action="store_true", help="Do not include ReSpeaker replay supervisors.")
    ap.add_argument("--no-gps", action="store_true", help="Do not include GPS replay supervisor.")
    ap.add_argument("--no-ce", action="store_true", help="Do not include starter complex-event detector.")
    ap.add_argument("--debug-raw-mqtt", action="store_true", help="Let ZED replay publish low-rate RGB/depth MQTT debug streams.")
    ap.add_argument("--zed-gpu", dest="zed_gpu", action="store_true", default=None, help="Add gpus: all to ZED replay services. This is now the default because pyzed needs libcuda.")
    ap.add_argument("--no-zed-gpu", dest="zed_gpu", action="store_false", help="Do not add Compose GPU reservation to ZED replay services.")
    ap.add_argument("--yolo-gpu", dest="yolo_gpu", action="store_true", default=None, help="Add gpus: all to YOLO detector services. This is now the default when YOLO is included.")
    ap.add_argument("--no-yolo-gpu", dest="yolo_gpu", action="store_false", help="Do not add Compose GPU reservation to YOLO detector services; use CPU/auto inside the container.")
    ap.add_argument("--yolo-debug-images", action="store_true", help="Publish low-rate YOLO annotated JPEG frames to MQTT for the /device web display. Off by default to keep MQTT compact.")
    ap.add_argument("--yolo-annotated-fps", type=float, default=1.0, help="Max FPS for YOLO annotated debug images when --yolo-debug-images is enabled.")
    ap.add_argument("--detector-debug-status", action="store_true", help="Publish synthetic detector health/debug status under /debug/<node>/... topics. Default is off so replay/event topics only contain replayed or detector event data.")
    ap.add_argument("--yolo-debug-status", action="store_true", help="Publish synthetic YOLO health/debug status under /debug/<node>/analytics/yolo/status. Default is off.")
    ap.add_argument("--yolo-frame-debug", action="store_true", help="Publish a 1Hz /debug/<node>/analytics/yolo/frame probe after decoded ZED frames arrive. Useful for distinguishing no-video from no-detections.")
    ap.add_argument("--audio-debug-status", action="store_true", help="Publish synthetic audio-detector health/debug status under /debug/<node>/audio_detector/status. Default is off.")
    ap.add_argument("--yolo-status-fps", type=float, default=1.0, help="YOLO debug/status and frame-probe MQTT frequency when enabled.")
    ap.add_argument("--audio-threshold", type=float, default=-30.0)
    ap.add_argument("--audio-status-fps", type=float, default=1.0, help="Audio debug-status MQTT frequency when --detector-debug-status or --audio-debug-status is enabled.")
    ap.add_argument("--audio-idle-status", action="store_true", help="Also publish audio debug status before any audio frames arrive. Only applies when audio debug status is enabled.")
    ap.add_argument("--build-network", default="host", choices=["host", "default"], help="Network mode used while building images. Default host helps when Docker bridge DNS cannot resolve apt/pip domains; use default to omit this Compose build option.")
    args = ap.parse_args()

    # Defaults are optimized for the real replay stack: all replay services, CE,
    # YOLO model loaded, and GPU access for ZED/YOLO. Use the --no-* flags for
    # lighter or CPU-only runs.
    if args.include_yolo is None:
        args.include_yolo = True
    if args.zed_gpu is None:
        args.zed_gpu = True
    if args.yolo_gpu is None:
        args.yolo_gpu = True

    if args.yolo_debug:
        args.include_yolo = True
        args.load_yolo_model = "true"
        args.yolo_debug_images = True
        args.yolo_frame_debug = True
        args.yolo_debug_status = True
        args.no_respeaker = True
        args.no_audio_detector = True
        args.no_gps = True
        args.no_ce = True

    root = Path(__file__).resolve().parents[1]
    data_roots = data_parent_candidates(args.data_dir)
    if args.device:
        nodes = args.device
    elif args.nodes:
        nodes = args.nodes
    else:
        nodes = discover_nodes(data_roots)
    if not nodes:
        nodes = ["orin11", "orin12"]
        print("WARNING: no node folders discovered; writing default nodes orin11/orin12. Use --nodes to override.")

    zed_settings = root / "setup" / "zed_settings"
    iobt_roots = container_data_roots(data_roots)
    out = Path(args.compose_out).resolve()

    lines: list[str] = []
    lines.append("# Scenario-agnostic persistent replay stack.\n")
    lines.append("# Generated by setup/generate_replay_compose.py.\n")
    lines.append("# Use the web UI or tools/replay_control.py to publish /replay/config and /replay/sync.\n")
    lines.append("services:\n")

    for node in nodes:
        hostname = mcp_hostname(node)
        host_tmp = f"/tmp/iobt-{node}"

        if not args.no_zed:
            lines.append(f"\n  zed-{node}:\n")
            emit_build(lines, "services/replay/zed/Dockerfile", args.build_network)
            lines.append("    image: iobt-minimal/zed-replay:latest\n")
            lines.append(f"    container_name: zed-replay-{node}\n")
            lines.append("    working_dir: /app\n")
            cmd = f"python3 -u /app/replay_entrypoint.py --service zed --node-folder {node} --node-name {hostname} --downsample-for-mqtt 1.0"
            if args.debug_raw_mqtt:
                cmd += " --debug-raw-mqtt"
            lines.append(f"    command: {yaml_quote(cmd)}\n")
            lines.append("    privileged: true\n")
            if args.zed_gpu:
                lines.append("    gpus: all\n")
            emit_extra_hosts(lines)
            lines.append("    volumes:\n")
            lines.append("      - /etc/timezone:/etc/timezone:ro\n")
            emit_data_root_volumes(lines, data_roots)
            lines.append(f"      - {yaml_quote(str(zed_settings) + ':/usr/local/zed/settings/:ro')}\n")
            lines.append(f"      - {yaml_quote(host_tmp + ':/tmp')}\n")
            emit_env(lines, [
                "TEST_CONTROL=False",
                f"MCP_NODE_NAME={hostname}",
                "MQTT_HOST_IP=${MQTT_HOST_IP:-host.docker.internal}",
                "MQTT_PORT=${MQTT_PORT:-1883}",
                "MCP_CONTAINER_OUTPUT_DIR=/output",
                "SERIALIZER=msgpack",
                f"IOBT_DATA_ROOTS={iobt_roots}",
                f"IOBT_NODE_FOLDER={node}",
            ])

            if args.include_yolo:
                lines.append(f"\n  yolo-{node}:\n")
                emit_build(lines, "services/analytics/yolo_detector/Dockerfile", args.build_network)
                lines.append("    image: iobt-minimal/yolo-detector:latest\n")
                lines.append(f"    container_name: yolo-detector-{node}\n")
                lines.append("    working_dir: /app\n")
                lines.append("    command: python3 -u /app/app.py\n")
                if args.yolo_gpu:
                    lines.append("    gpus: all\n")
                emit_extra_hosts(lines)
                lines.append("    volumes:\n")
                lines.append(f"      - {yaml_quote(host_tmp + ':/tmp')}\n")
                emit_env(lines, [
                    "TEST_CONTROL=False",
                    f"MCP_NODE_NAME={hostname}",
                    "MQTT_HOST_IP=${MQTT_HOST_IP:-host.docker.internal}",
                    "MQTT_PORT=${MQTT_PORT:-1883}",
                    "MCP_CONTAINER_OUTPUT_DIR=/tmp",
                    "SERIALIZER=msgpack",
                    "SOURCE=local",
                    f"LOAD_MODEL={args.load_yolo_model}",
                    "YOLO_MODEL=/app/yolov8n.pt",
                    "YOLO_DEVICE=${YOLO_DEVICE:-auto}",
                    f"YOLO_PUBLISH_ANNOTATED={str(args.yolo_debug_images).lower()}",
                    f"YOLO_ANNOTATED_FPS={args.yolo_annotated_fps}",
                    f"YOLO_PUBLISH_STATUS={str(args.detector_debug_status or args.yolo_debug_status).lower()}",
                    f"YOLO_PUBLISH_FRAME_STATUS={str(args.detector_debug_status or args.yolo_debug_status or args.yolo_frame_debug).lower()}",
                    f"YOLO_STATUS_FPS={args.yolo_status_fps}",
                    f"YOLO_FRAME_STATUS_FPS={args.yolo_status_fps}",
                    "YOLO_ANNOTATED_WIDTH=960",
                    "YOLO_ANNOTATED_JPEG_QUALITY=70",
                ])

        if not args.no_respeaker:
            lines.append(f"\n  respeaker-{node}:\n")
            emit_build(lines, "services/replay/respeaker/Dockerfile", args.build_network)
            lines.append("    image: iobt-minimal/respeaker-replay:latest\n")
            lines.append(f"    container_name: respeaker-replay-{node}\n")
            lines.append("    working_dir: /app\n")
            cmd = f"python3 -u /app/replay_entrypoint.py --service respeaker --node-folder {node} --node-name {hostname}"
            lines.append(f"    command: {yaml_quote(cmd)}\n")
            lines.append("    privileged: true\n")
            emit_extra_hosts(lines)
            lines.append("    volumes:\n")
            lines.append("      - /etc/timezone:/etc/timezone:ro\n")
            emit_data_root_volumes(lines, data_roots)
            lines.append(f"      - {yaml_quote(host_tmp + ':/tmp')}\n")
            emit_env(lines, [
                "TEST_CONTROL=False",
                f"MCP_NODE_NAME={hostname}",
                "MQTT_HOST_IP=${MQTT_HOST_IP:-host.docker.internal}",
                "MQTT_PORT=${MQTT_PORT:-1883}",
                "MCP_CONTAINER_OUTPUT_DIR=/output",
                "SERIALIZER=msgpack",
                f"IOBT_DATA_ROOTS={iobt_roots}",
                f"IOBT_NODE_FOLDER={node}",
            ])

            if not args.no_audio_detector:
                lines.append(f"\n  audio-detector-{node}:\n")
                emit_build(lines, "services/analytics/audio_detector/Dockerfile", args.build_network)
                lines.append("    image: iobt-minimal/audio-detector:latest\n")
                lines.append(f"    container_name: audio-detector-{node}\n")
                lines.append("    working_dir: /app\n")
                lines.append("    command: python3 -u /app/app.py\n")
                emit_extra_hosts(lines)
                lines.append("    volumes:\n")
                lines.append(f"      - {yaml_quote(host_tmp + ':/tmp')}\n")
                emit_env(lines, [
                    "TEST_CONTROL=False",
                    f"MCP_NODE_NAME={hostname}",
                    "MQTT_HOST_IP=${MQTT_HOST_IP:-host.docker.internal}",
                    "MQTT_PORT=${MQTT_PORT:-1883}",
                    "MCP_CONTAINER_OUTPUT_DIR=/tmp",
                    "SERIALIZER=msgpack",
                    f"DETECTION_THRESHOLD={args.audio_threshold}",
                    f"AUDIO_PUBLISH_STATUS={str(args.detector_debug_status or args.audio_debug_status).lower()}",
                    f"AUDIO_STATUS_FPS={args.audio_status_fps}",
                    f"AUDIO_PUBLISH_IDLE_STATUS={str(args.audio_idle_status).lower()}",
                    "AUDIO_PRINT_IDLE_STATUS=false",
                ])

    if not args.no_gps:
        lines.append("\n  gps-replay:\n")
        emit_build(lines, "services/replay/gps/Dockerfile", args.build_network)
        lines.append("    image: iobt-minimal/gps-replay:latest\n")
        lines.append("    container_name: gps-replay\n")
        lines.append("    working_dir: /app\n")
        lines.append("    command: \"python3 -u /app/replay_entrypoint.py --service gps --node-name x86server\"\n")
        emit_extra_hosts(lines)
        lines.append("    volumes:\n")
        emit_data_root_volumes(lines, data_roots)
        lines.append("      - /tmp/iobt-gps:/tmp\n")
        emit_env(lines, [
            "TEST_CONTROL=False",
            "MCP_NODE_NAME=x86server",
            "MQTT_HOST_IP=${MQTT_HOST_IP:-host.docker.internal}",
            "MQTT_PORT=${MQTT_PORT:-1883}",
            "MCP_CONTAINER_OUTPUT_DIR=/tmp",
            "SERIALIZER=msgpack",
            f"IOBT_DATA_ROOTS={iobt_roots}",
        ])

    if not args.no_ce:
        lines.append("\n  complex-event-detector:\n")
        emit_build(lines, "services/orchestration/complex_event_detector/Dockerfile", args.build_network)
        lines.append("    image: iobt-minimal/complex-event-detector:latest\n")
        lines.append("    container_name: complex-event-detector\n")
        lines.append("    command: python3 -u /app/app.py\n")
        emit_extra_hosts(lines)
        emit_env(lines, [
            "MQTT_HOST_IP=${MQTT_HOST_IP:-host.docker.internal}",
            "MQTT_PORT=${MQTT_PORT:-1883}",
            "CE_WINDOW_SEC=${CE_WINDOW_SEC:-5.0}",
            "CE_OUTPUT_TOPIC=/complex_events/demo",
        ])

    out.write_text("".join(lines))
    print(f"Data roots mounted: {[str(p) for p in data_roots]}")
    print(f"Nodes included: {nodes}")
    print(f"Wrote: {out}")
    print(f"Build network: {args.build_network}")
    print(f"YOLO included: {args.include_yolo} (model={args.load_yolo_model}, gpu={args.yolo_gpu})")
    print(f"ZED included: {not args.no_zed} (gpu={args.zed_gpu})")
    print(f"ReSpeaker/audio included: {not args.no_respeaker}/{not args.no_audio_detector}")
    print(f"GPS/CE included: {not args.no_gps}/{not args.no_ce}")
    if args.yolo_debug:
        print("YOLO debug preset enabled: ZED + YOLO only, annotated images + frame/status debug.")
    if not (args.detector_debug_status or args.yolo_debug_status or args.audio_debug_status or args.yolo_frame_debug):
        print("Detector debug status is disabled; normal MQTT output will contain detections/events, not synthetic health messages.")
    elif args.yolo_frame_debug and not (args.detector_debug_status or args.yolo_debug_status):
        print("YOLO frame probe is enabled under /debug/<node>/analytics/yolo/frame; full YOLO status remains disabled.")
    print("Start it with: docker compose -f compose.replay.yaml up --build")


if __name__ == "__main__":
    main()
