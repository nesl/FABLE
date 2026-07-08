#!/usr/bin/env python3
"""Generate a minimal replay + local detector + CE docker compose file.

The generator now understands Brian's external SSD layout by default:

  /media/brianw/Extreme SSD/West Point Experimentation/<YYYYMMDD>/orin*/...
  /media/brianw/Extreme SSD/GQ Data/<YYYYMMDD>/orin*/...

For a scenario such as 20260414_134838, the date component is 20260414. The
script searches each parent root for a matching 20260414 directory, then mounts
that date's node folders into the generated replay containers.

Expected real-data layout:

  <data-parent>/<YYYYMMDD>/
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

Examples:
  # Auto-search the default SSD roots.
  python3 setup/generate_compose.py --scenario 20260414_134838

  # Override or add an alternate parent root.
  python3 setup/generate_compose.py --scenario 20260414_134838 --data-dir /data/iobt

  # Search multiple roots explicitly.
  python3 setup/generate_compose.py --scenario 20260414_134838 --data-dir /data/gq --data-dir /data/west_point
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_DATA_PARENTS = [
    Path("/media/brianw/Extreme SSD/West Point Experimentation"),
    Path("/media/brianw/Extreme SSD/GQ Data"),
]

# Optional environment override. Use ':'-separated paths, e.g.
# IOBT_DATA_ROOTS="/data/west_point:/data/gq"
ENV_DATA_ROOTS = "IOBT_DATA_ROOTS"
SCENARIO_RE = re.compile(r"^(\d{8}_\d{6})")


def yaml_quote(value: str | Path) -> str:
    """Return a conservative double-quoted YAML scalar."""
    text = str(value)
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'


def parse_env_roots() -> list[Path]:
    raw = os.environ.get(ENV_DATA_ROOTS, "").strip()
    if not raw:
        return []
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]


def data_parent_candidates(cli_data_dirs: list[str] | None) -> list[Path]:
    raw_roots: list[Path] = []
    if cli_data_dirs:
        raw_roots.extend(Path(p).expanduser() for p in cli_data_dirs)
    else:
        raw_roots.extend(parse_env_roots())
        raw_roots.extend(DEFAULT_DATA_PARENTS)

    # Preserve order but deduplicate after expanding user markers.
    seen: set[str] = set()
    roots: list[Path] = []
    for root in raw_roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            roots.append(root)
    return roots


def scenario_date(scenario: str) -> str:
    # Scenario IDs normally look like 20260414_134838. If the user passes only a
    # date, we still use it for root discovery, but replay apps usually need the
    # full prefix to find exact files.
    return scenario.split("_")[0]


def mcp_hostname(node_folder: str) -> str:
    if node_folder.startswith("orin"):
        suffix = node_folder.replace("orin", "")
        return f"dvpg_gq_orin_{suffix}"
    return node_folder.replace("-", "_")


def has_sensor_file(node_dir: Path, scenario: str, keyword: str) -> bool:
    if not node_dir.is_dir():
        return False
    return any(p.name.startswith(scenario) and keyword in p.name for p in node_dir.iterdir() if p.is_file())


def resolve_date_dir(data_parents: Iterable[Path], scenario: str) -> tuple[Path, Path]:
    date = scenario_date(scenario)
    searched: list[Path] = []
    matches: list[tuple[Path, Path]] = []

    for parent in data_parents:
        parent = parent.expanduser()
        # If the user accidentally passes the date directory itself, accept it.
        candidate = parent if parent.name == date else parent / date
        searched.append(candidate)
        if candidate.is_dir():
            root = candidate.parent if candidate.name == date else parent
            matches.append((root, candidate))

    if not matches:
        msg = [
            f"ERROR: no data directory found for scenario/date {scenario!r}.",
            "Searched:",
        ]
        msg.extend(f"  - {p}" for p in searched)
        msg.extend([
            "",
            "Pass --data-dir /path/to/parent if the data lives somewhere else, or set",
            f"{ENV_DATA_ROOTS} with colon-separated parent paths.",
        ])
        raise SystemExit("\n".join(msg))

    if len(matches) > 1:
        print("WARNING: multiple matching date directories found; using the first:", file=sys.stderr)
        for _root, date_dir in matches:
            print(f"  - {date_dir}", file=sys.stderr)

    return matches[0]


def discover(data_parents: list[Path], scenario: str, node_prefixes: list[str] | None):
    data_root, date_dir = resolve_date_dir(data_parents, scenario)

    node_dirs = [p for p in date_dir.iterdir() if p.is_dir() and p.name != "GPS"]
    if node_prefixes:
        node_dirs = [p for p in node_dirs if any(p.name.startswith(prefix) for prefix in node_prefixes)]

    zed_nodes: list[str] = []
    respeaker_nodes: list[str] = []
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

    return data_root, date_dir, zed_nodes, respeaker_nodes, has_gps


def list_scenarios(data_parents: list[Path]) -> None:
    """Best-effort scenario listing from filenames under date/orin* and date/GPS."""
    found: dict[str, set[str]] = {}
    for parent in data_parents:
        parent = parent.expanduser()
        if not parent.is_dir():
            continue
        for date_dir in sorted(parent.iterdir()):
            if not (date_dir.is_dir() and re.fullmatch(r"\d{8}", date_dir.name)):
                continue
            scenarios: set[str] = set()
            for child in date_dir.iterdir():
                if not child.is_dir():
                    continue
                if child.name == "GPS":
                    for obj in child.iterdir():
                        if obj.is_dir():
                            for f in obj.iterdir():
                                m = SCENARIO_RE.match(f.name)
                                if m:
                                    scenarios.add(m.group(1))
                else:
                    for f in child.iterdir():
                        m = SCENARIO_RE.match(f.name)
                        if m:
                            scenarios.add(m.group(1))
            if scenarios:
                found.setdefault(str(parent), set()).update(scenarios)

    if not found:
        print("No scenarios found under:")
        for parent in data_parents:
            print(f"  - {parent}")
        return

    for parent, scenarios in found.items():
        print(parent)
        for scenario in sorted(scenarios):
            print(f"  {scenario}")


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
    ap.add_argument("--scenario", help="Scenario ID, e.g. 20260414_134838")
    ap.add_argument(
        "--data-dir",
        action="append",
        default=None,
        help=(
            "Parent root containing <YYYYMMDD>/orin*/... data. Can be repeated. "
            "If omitted, searches IOBT_DATA_ROOTS and then the default Extreme SSD roots."
        ),
    )
    ap.add_argument("--list-scenarios", action="store_true", help="List scenarios found under the configured data roots and exit")
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
    data_parents = data_parent_candidates(args.data_dir)

    if args.list_scenarios:
        list_scenarios(data_parents)
        return

    if not args.scenario:
        ap.error("--scenario is required unless --list-scenarios is used")

    data_root, date_dir, zed_nodes, respeaker_nodes, has_gps = discover(data_parents, args.scenario, args.nodes)

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
    lines.append(f"# Data root: {data_root}\n")
    lines.append(f"# Mounted date directory: {date_dir}\n")
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
            lines.append("    image: iobt-minimal/zed-replay:latest\n")
            lines.append(f"    container_name: zed-replay-{node}\n")
            lines.append("    working_dir: /app\n")
            lines.append(f"    command: >\n      python3 -u /app/app.py --scenario {args.scenario} --start {args.start} --end {args.end}{flags}\n")
            lines.append("    privileged: true\n")
            lines.append("    gpus: all\n")
            emit_extra_hosts(lines)
            lines.append("    volumes:\n")
            lines.append("      - /etc/timezone:/etc/timezone:ro\n")
            lines.append(f"      - {yaml_quote(str(node_dir) + ':/output:ro')}\n")
            lines.append(f"      - {yaml_quote(str(zed_settings) + ':/usr/local/zed/settings/:ro')}\n")
            lines.append(f"      - {yaml_quote(host_tmp + ':/tmp')}\n")
            emit_env(lines, service_common_env(hostname) + [
                f"IOBT_DATA_ROOT={data_root}",
                f"IOBT_DATE_DIR={date_dir}",
                f"IOBT_SCENARIO={args.scenario}",
            ])

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
                lines.append(f"      - {yaml_quote(host_tmp + ':/tmp')}\n")
                env = service_common_env(hostname, output_dir="/tmp") + [
                    "SOURCE=local",
                    f"LOAD_MODEL={args.load_yolo_model}",
                    "YOLO_MODEL=/app/yolov8n.pt",
                    "YOLO_DEVICE=${YOLO_DEVICE:-auto}",
                    f"IOBT_DATA_ROOT={data_root}",
                    f"IOBT_DATE_DIR={date_dir}",
                    f"IOBT_SCENARIO={args.scenario}",
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
            lines.append(f"      - {yaml_quote(str(node_dir) + ':/output:ro')}\n")
            lines.append(f"      - {yaml_quote(host_tmp + ':/tmp')}\n")
            emit_env(lines, service_common_env(hostname) + [
                f"IOBT_DATA_ROOT={data_root}",
                f"IOBT_DATE_DIR={date_dir}",
                f"IOBT_SCENARIO={args.scenario}",
            ])

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
                lines.append(f"      - {yaml_quote(host_tmp + ':/tmp')}\n")
                env = service_common_env(hostname, output_dir="/tmp") + [
                    f"DETECTION_THRESHOLD={args.audio_threshold}",
                    f"IOBT_DATA_ROOT={data_root}",
                    f"IOBT_DATE_DIR={date_dir}",
                    f"IOBT_SCENARIO={args.scenario}",
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
        lines.append(f"      - {yaml_quote(str(gps_dir) + f':/data/{args.scenario}/GPS:ro')}\n")
        lines.append(f"      - {yaml_quote('/tmp/iobt-gps:/tmp')}\n")
        emit_env(lines, service_common_env("x86server", output_dir="/tmp") + [
            f"IOBT_DATA_ROOT={data_root}",
            f"IOBT_DATE_DIR={date_dir}",
            f"IOBT_SCENARIO={args.scenario}",
        ])

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
    print(f"Data root: {data_root}")
    print(f"Data date dir: {date_dir}")
    print(f"ZED nodes: {zed_nodes}")
    print(f"ReSpeaker nodes: {respeaker_nodes}")
    print(f"GPS included: {has_gps}")
    print(f"Wrote: {out}")
    if not zed_nodes and not respeaker_nodes and not has_gps:
        print("WARNING: no replay services were generated. Check --scenario, data roots, and file naming.")


if __name__ == "__main__":
    main()
