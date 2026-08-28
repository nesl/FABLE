#!/usr/bin/env python3
"""Fail-closed validation for a completed physical-proxy experiment cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from netwaggle.netwaggle.topology import NetWaggleTopology
from netwaggle.netwaggle.util import load_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--since-wall-time",
        type=float,
        default=0.0,
        help="Ignore proxy records before this Unix wall timestamp.",
    )
    args = parser.parse_args()
    topology = NetWaggleTopology.from_dict(load_json(args.topology))
    rows = [
        json.loads(line)
        for line in args.metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [
        row for row in rows
        if float(row.get("wall_time", 0.0)) >= args.since_wall_time
    ]
    failures: list[str] = []
    results = []
    for proxy in topology.external_proxies:
        if not proxy.required:
            continue
        connected = [
            row for row in rows
            if row.get("proxy") == proxy.name and row.get("event") == "CONNECTED"
        ]
        closed = [
            row for row in rows
            if row.get("proxy") == proxy.name and row.get("event") == "CLOSED"
        ]
        transferred = sum(
            int(row.get("client_to_target_bytes", 0))
            + int(row.get("target_to_client_bytes", 0))
            for row in closed
        )
        source_valid = all(
            not proxy.expected_source_ip
            or row.get("outbound_source_ip") == proxy.expected_source_ip
            for row in connected
        )
        successful_closes = [
            row for row in closed
            if not row.get("error")
            and (
                int(row.get("client_to_target_bytes", 0))
                + int(row.get("target_to_client_bytes", 0))
            ) > 0
        ]
        first_success_wall_time = min(
            (float(row.get("wall_time", 0.0)) for row in successful_closes),
            default=None,
        )
        errors_before_success = [
            row.get("error") for row in closed
            if row.get("error")
            and (
                first_success_wall_time is None
                or float(row.get("wall_time", 0.0)) <= first_success_wall_time
            )
        ]
        errors_after_success = [
            row.get("error") for row in closed
            if row.get("error")
            and first_success_wall_time is not None
            and float(row.get("wall_time", 0.0)) > first_success_wall_time
        ]
        # A physical client may retry after the experiment has deliberately torn
        # down its broker.  Such teardown retries do not erase an already proven
        # proxy path.  Fail closed if no transfer succeeded or if errors preceded
        # the first successful transfer.
        valid = bool(
            connected
            and successful_closes
            and transferred > 0
            and source_valid
            and not errors_before_success
        )
        if not valid:
            failures.append(proxy.name)
        results.append({
            "proxy": proxy.name,
            "connections": len(connected),
            "closed_connections": len(closed),
            "bytes": transferred,
            "source_valid": source_valid,
            "errors_before_success": errors_before_success,
            "post_success_errors": errors_after_success,
            "valid": valid,
        })
    report = {
        "schema_version": "netwaggle.external_proxy_validation.v1",
        "validated": not failures,
        "since_wall_time": args.since_wall_time,
        "failed_required_proxies": failures,
        "proxies": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["validated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
