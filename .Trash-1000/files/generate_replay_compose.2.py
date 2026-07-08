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
    ap.add_argument("--compose-out", default="compose.replay.yaml")
    ap.add_argument("--include-yolo", action="store_true", help="Include local YOLO detector containers for each node. Can use GPU/CPU resources.")
    ap.add_argument("--load-yolo-model", default="false", choices=["true", "false"], help="Set true for actual YOLO inference; false is useful for plumbing tests.")
    ap.add_argument("--no-audio-detector", action="store_true", help="Do not include local audio detector containers.")
    ap.add_argument("--no-zed", action="store_true", help="Do not include ZED replay supervisors.")
    ap.add_argument("--no-respeaker", action="store_true", help="Do not include ReSpeaker replay supervisors.")
    ap.add_argument("--no-gps", action="store_true", help="Do not include GPS replay supervisor.")
    ap.add_argument("--no-ce", action="store_true", help="Do not include starter complex-event detector.")
    ap.add_argument("--debug-raw-mqtt", action="store_true", help="Let ZED replay publish low-rate RGB/depth MQTT debug streams.")
    ap.add_argument("--zed-gpu", action="store_true", help="Add gpus: all to ZED replay services. Default is off so Docker works without NVIDIA Container Toolkit.")
    ap.add_argument("--yolo-gpu", action="store_true", help="Add gpus: all to YOLO detector services. Default is off; YOLO can still use CPU or auto-detect inside the container.")
    ap.add_argument("--audio-threshold", type=float, default=-30.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_roots = data_parent_candidates(args.data_dir)
    nodes = args.nodes if args.nodes else discover_nodes(data_roots)
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
            lines.append("    build:\n      context: .\n      dockerfile: services/replay/zed/Dockerfile\n")
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
                lines.append("    build:\n      context: .\n      dockerfile: services/analytics/yolo_detector/Dockerfile\n")
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
                ])

        if not args.no_respeaker:
            lines.append(f"\n  respeaker-{node}:\n")
            lines.append("    build:\n      context: .\n      dockerfile: services/replay/respeaker/Dockerfile\n")
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
                lines.append("    build:\n      context: .\n      dockerfile: services/analytics/audio_detector/Dockerfile\n")
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
                ])

    if not args.no_gps:
        lines.append("\n  gps-replay:\n")
        lines.append("    build:\n      context: .\n      dockerfile: services/replay/gps/Dockerfile\n")
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
        lines.append("    build:\n      context: .\n      dockerfile: services/orchestration/complex_event_detector/Dockerfile\n")
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
    print("Start it with: docker compose -f compose.replay.yaml up --build")


if __name__ == "__main__":
    main()
