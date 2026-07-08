#!/usr/bin/env python3
"""Generate a minimal replay + local detector + CE docker compose file.

Expected real-data layout, matching the current local-containers setup.py pattern:

  <data-dir>/<YYYYMMDD>/
    orin11/
      <scenario>_dvpg_gq_orin_11_zed.svo2
      <scenario>_dvpg_gq_orin_11_zed.csv
      <scenario>_dvpg_gq_orin_11_respeaker.flac
      <scenario>_dvpg_gq_orin_11_respeaker.csv
    orin12/
      ...
    GPS/
      <object>/
        <scenario>_*_gps.csv

Example:
  python3 setup/generate_compose.py --scenario 20260414_134838 --data-dir /data/iobt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def mcp_hostname(node_folder: str) -> str:
    if node_folder.startswith("orin"):
        suffix = node_folder.replace("orin", "")
        return f"dvpg_gq_orin_{suffix}"
    return node_folder.replace("-", "_")


def has_sensor_file(node_dir: Path, scenario: str, keyword: str) -> bool:
    if not node_dir.is_dir():
        return False
    return any(p.name.startswith(scenario) and keyword in p.name for p in node_dir.iterdir() if p.is_file())


def discover(data_dir: Path, scenario: str, node_prefixes: list[str] | None):
    date = scenario.split("_")[0]
    date_dir = data_dir / date
    if not date_dir.is_dir():
        raise SystemExit(f"ERROR: date directory not found: {date_dir}")

    node_dirs = [p for p in date_dir.iterdir() if p.is_dir() and p.name != "GPS"]
    if node_prefixes:
        node_dirs = [p for p in node_dirs if any(p.name.startswith(prefix) for prefix in node_prefixes)]

    zed_nodes = []
    respeaker_nodes = []
    for p in sorted(node_dirs):
        if has_sensor_file(p, scenario, "zed"):
            zed_nodes.append(p.name)
        if has_sensor_file(p, scenario, "respeaker"):
            respeaker_nodes.append(p.name)

    gps_dir = date_dir / "GPS"
    has_gps = False
    if gps_dir.is_dir():
        has_gps = any(
            f.is_file() and f.name.startswith(scenario) and "gps.csv" in f.name
            for obj in gps_dir.iterdir() if obj.is_dir()
            for f in obj.iterdir()
        )

    return date_dir, zed_nodes, respeaker_nodes, has_gps


def service_common_env(hostname: str, output_dir: str = "/output") -> list[str]:
    return [
        "TEST_CONTROL=False",
        f"MCP_NODE_NAME={hostname}",
        "MQTT_HOST_IP=${MQTT_HOST_IP:-host.docker.internal}",
        "MQTT_PORT=${MQTT_PORT:-1883}",
        f"MCP_CONTAINER_OUTPUT_DIR={output_dir}",
        "SERIALIZER=msgpack",
    ]


def emit_env(lines: list[str], env: list[str], indent: str = "    "):
    lines.append(f"{indent}environment:\n")
    for item in env:
        lines.append(f"{indent}  - {item}\n")


def emit_extra_hosts(lines: list[str], indent: str = "    "):
    lines.append(f"{indent}extra_hosts:\n")
    lines.append(f"{indent}  - \"host.docker.internal:host-gateway\"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="Scenario ID, e.g. 20260414_134838")
    ap.add_argument("--data-dir", required=True, help="Root containing <YYYYMMDD>/orin*/... data")
    ap.add_argument("--compose-out", default=None, help="Output compose file. Default: compose.generated.<scenario>.yaml")
    ap.add_argument("--start", type=float, default=0.0, help="Replay start offset in seconds")
    ap.add_argument("--end", type=float, default=-1.0, help="Replay end offset; -1 means full recording")
    ap.add_argument("--nodes", nargs="*", help="Optional node folder prefixes/names, e.g. orin11 orin12")
    ap.add_argument("--no-zed", action="store_true", help="Do not include ZED replay containers")
    ap.add_argument("--no-respeaker", action="store_true", help="Do not include ReSpeaker replay containers")
    ap.add_argument("--no-gps", action="store_true", help="Do not include GPS replay container")
    ap.add_argument("--no-yolo", action="store_true", help="Do not include colocated YOLO detectors")
    ap.add_argument("--no-audio-detector", action="store_true", help="Do not include colocated audio detectors")
    ap.add_argument("--no-ce", action="store_true", help="Do not include the starter complex-event detector")
    ap.add_argument("--debug-raw-mqtt", action="store_true", help="Let ZED replay publish RGB/depth MQTT debug streams")
    ap.add_argument("--load-yolo-model", default="true", choices=["true", "false"], help="Set false for plumbing tests without model inference")
    ap.add_argument("--audio-threshold", type=float, default=-30.0, help="Audio loudness threshold in dB")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir).expanduser().resolve()
    date_dir, zed_nodes, respeaker_nodes, has_gps = discover(data_dir, args.scenario, args.nodes)

    if args.no_zed:
        zed_nodes = []
    if args.no_respeaker:
        respeaker_nodes = []
    if args.no_gps:
        has_gps = False

    out = Path(args.compose_out or root / f"compose.generated.{args.scenario}.yaml").resolve()
    zed_settings = root / "setup" / "zed_settings"

    lines: list[str] = []
    lines.append(f"# Generated by setup/generate_compose.py for {args.scenario}\n")
    lines.append("services:\n")

    active_nodes = sorted(set(zed_nodes + respeaker_nodes))

    for node in active_nodes:
        node_dir = date_dir / node
        hostname = mcp_hostname(node)
        host_tmp = f"/tmp/iobt-{node}"

        if node in zed_nodes:
            flags = "" if args.debug_raw_mqtt else " --no_rgb_mqtt --no_depth_mqtt"
            lines.append(f"\n  zed-{node}:\n")
            lines.append("    build:\n")
            lines.append("      context: .\n")
            lines.append("      dockerfile: services/replay/zed/Dockerfile\n")
            lines.append(f"    image: iobt-minimal/zed-replay:latest\n")
            lines.append(f"    container_name: zed-replay-{node}\n")
            lines.append("    working_dir: /app\n")
            lines.append(f"    command: >\n      python3 -u /app/app.py --scenario {args.scenario} --start {args.start} --end {args.end}{flags}\n")
            lines.append("    privileged: true\n")
            lines.append("    gpus: all\n")
            emit_extra_hosts(lines)
            lines.append("    volumes:\n")
            lines.append("      - /etc/timezone:/etc/timezone:ro\n")
            lines.append(f"      - {node_dir}:/output:ro\n")
            lines.append(f"      - {zed_settings}:/usr/local/zed/settings/:ro\n")
            lines.append(f"      - {host_tmp}:/tmp\n")
            emit_env(lines, service_common_env(hostname))

            if not args.no_yolo:
                lines.append(f"\n  yolo-{node}:\n")
                lines.append("    build:\n")
                lines.append("      context: .\n")
                lines.append("      dockerfile: services/analytics/yolo_detector/Dockerfile\n")
                lines.append("    image: iobt-minimal/yolo-detector:latest\n")
                lines.append(f"    container_name: yolo-detector-{node}\n")
                lines.append("    working_dir: /app\n")
                lines.append("    command: python3 -u /app/app.py\n")
                emit_extra_hosts(lines)
                lines.append("    volumes:\n")
                lines.append(f"      - {host_tmp}:/tmp\n")
                env = service_common_env(hostname, output_dir="/tmp") + [
                    "SOURCE=local",
                    f"LOAD_MODEL={args.load_yolo_model}",
                    "YOLO_MODEL=/app/yolov8n.pt",
                    "YOLO_DEVICE=${YOLO_DEVICE:-auto}",
                ]
                emit_env(lines, env)
                lines.append(f"    depends_on:\n      - zed-{node}\n")

        if node in respeaker_nodes:
            lines.append(f"\n  respeaker-{node}:\n")
            lines.append("    build:\n")
            lines.append("      context: .\n")
            lines.append("      dockerfile: services/replay/respeaker/Dockerfile\n")
            lines.append("    image: iobt-minimal/respeaker-replay:latest\n")
            lines.append(f"    container_name: respeaker-replay-{node}\n")
            lines.append("    working_dir: /app\n")
            lines.append(f"    command: >\n      python3 -u /app/app.py --scenario {args.scenario} --start {args.start} --end {args.end}\n")
            lines.append("    privileged: true\n")
            emit_extra_hosts(lines)
            lines.append("    volumes:\n")
            lines.append("      - /etc/timezone:/etc/timezone:ro\n")
            lines.append(f"      - {node_dir}:/output:ro\n")
            lines.append(f"      - {host_tmp}:/tmp\n")
            emit_env(lines, service_common_env(hostname))

            if not args.no_audio_detector:
                lines.append(f"\n  audio-detector-{node}:\n")
                lines.append("    build:\n")
                lines.append("      context: .\n")
                lines.append("      dockerfile: services/analytics/audio_detector/Dockerfile\n")
                lines.append("    image: iobt-minimal/audio-detector:latest\n")
                lines.append(f"    container_name: audio-detector-{node}\n")
                lines.append("    working_dir: /app\n")
                lines.append("    command: python3 -u /app/app.py\n")
                emit_extra_hosts(lines)
                lines.append("    volumes:\n")
                lines.append(f"      - {host_tmp}:/tmp\n")
                env = service_common_env(hostname, output_dir="/tmp") + [
                    f"DETECTION_THRESHOLD={args.audio_threshold}",
                ]
                emit_env(lines, env)
                lines.append(f"    depends_on:\n      - respeaker-{node}\n")

    if has_gps:
        gps_dir = date_dir / "GPS"
        lines.append("\n  gps-replay:\n")
        lines.append("    build:\n")
        lines.append("      context: .\n")
        lines.append("      dockerfile: services/replay/gps/Dockerfile\n")
        lines.append("    image: iobt-minimal/gps-replay:latest\n")
        lines.append("    container_name: gps-replay\n")
        lines.append("    working_dir: /app\n")
        lines.append(f"    command: >\n      python3 -u /app/app.py --scenario {args.scenario} --start {args.start} --end {args.end}\n")
        emit_extra_hosts(lines)
        lines.append("    volumes:\n")
        lines.append(f"      - {gps_dir}:/data/{args.scenario}/GPS:ro\n")
        lines.append("      - /tmp/iobt-gps:/tmp\n")
        emit_env(lines, service_common_env("x86server", output_dir="/tmp"))

    if not args.no_ce:
        lines.append("\n  complex-event-detector:\n")
        lines.append("    build:\n")
        lines.append("      context: .\n")
        lines.append("      dockerfile: services/orchestration/complex_event_detector/Dockerfile\n")
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

    print(f"Scenario: {args.scenario}")
    print(f"Data date dir: {date_dir}")
    print(f"ZED nodes: {zed_nodes}")
    print(f"ReSpeaker nodes: {respeaker_nodes}")
    print(f"GPS included: {has_gps}")
    print(f"Wrote: {out}")
    if not zed_nodes and not respeaker_nodes and not has_gps:
        print("WARNING: no replay services were generated. Check --scenario, --data-dir, and file naming.")


if __name__ == "__main__":
    main()
