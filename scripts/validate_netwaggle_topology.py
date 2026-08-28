#!/usr/bin/env python3
"""Bounded live validation of every configured NetWaggle anchor namespace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def _run(argv: tuple[str, ...], timeout: float = 5) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("topology", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ping-timeout", type=int, default=2)
    args = parser.parse_args()
    document = json.loads(args.topology.read_text(encoding="utf-8"))
    gateway = str(document["gateway"]["ip"]).split("/", 1)[0]
    rows = []
    observed_ips: set[str] = set()
    for node in document["logical_nodes"]:
        container = str(node["anchor_container"])
        expected_ip = str(node["ip"]).split("/", 1)[0]
        inspect = _run(
            (
                "docker",
                "inspect",
                "--format",
                "{{.State.Running}} {{.State.Pid}}",
                container,
            )
        )
        address = _run(
            (
                "docker",
                "exec",
                container,
                "ip",
                "-j",
                "address",
                "show",
                "dev",
                "netwaggle0",
            )
        )
        ping = _run(
            (
                "docker",
                "exec",
                container,
                "ping",
                "-c",
                "1",
                "-W",
                str(args.ping_timeout),
                gateway,
            ),
            timeout=args.ping_timeout + 3,
        )
        actual_ips: list[str] = []
        if address.returncode == 0:
            for interface in json.loads(address.stdout):
                actual_ips.extend(
                    str(item["local"])
                    for item in interface.get("addr_info", ())
                    if item.get("family") == "inet"
                )
        running_parts = inspect.stdout.strip().split()
        running = (
            inspect.returncode == 0
            and len(running_parts) == 2
            and running_parts[0] == "true"
            and int(running_parts[1]) > 1
        )
        valid = (
            running
            and expected_ip in actual_ips
            and expected_ip not in observed_ips
            and ping.returncode == 0
        )
        observed_ips.add(expected_ip)
        rows.append(
            {
                "logical_node": node["name"],
                "container": container,
                "expected_ip": expected_ip,
                "actual_ips": actual_ips,
                "running": running,
                "gateway_reachable": ping.returncode == 0,
                "valid": valid,
            }
        )
    result = {
        "schema_version": "fable.netwaggle_topology_validation.v1",
        "topology": document["name"],
        "expected_node_count": len(document["logical_nodes"]),
        "validated_node_count": sum(row["valid"] for row in rows),
        "unique_expected_ip_count": len(observed_ips),
        "successful": bool(rows) and all(row["valid"] for row in rows),
        "nodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["successful"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
