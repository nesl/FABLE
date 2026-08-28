#!/usr/bin/env python3
"""Collect versioned, non-headline inventory profiles from physical nodes."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess


COMMAND_TEMPLATES = {
    "rpi_device_01": (
        "{rpi_host}",
        "hostname; uname -m; nproc; awk '/MemTotal/{{print $2}}' /proc/meminfo; "
        "df -B1 {rpi_root} | tail -1; ip -j route get {host_address}; "
        "timedatectl show -p NTPSynchronized -p SystemClockSynchronized; "
        "python3 --version; docker --version",
    ),
    "orin_edge_01": (
        "{jetson_host}",
        "hostname; uname -m; nproc; awk '/MemTotal/{{print $2}}' /proc/meminfo; "
        "df -B1 {jetson_root} | tail -1; ip -j route get {host_address}; "
        "chronyc tracking; /usr/bin/python3 --version; timeout 1 /usr/bin/tegrastats --interval 100; "
        "/usr/bin/python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'; "
        "docker --version",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rpi-host", default="rpi")
    parser.add_argument("--jetson-host", default="jetson")
    parser.add_argument("--host-address", required=True)
    parser.add_argument("--rpi-root", default="/opt/fable")
    parser.add_argument("--jetson-root", default="/opt/fable")
    args = parser.parse_args()
    observations = {}
    substitutions = vars(args)
    for node_id, (host_template, command_template) in COMMAND_TEMPLATES.items():
        host = host_template.format(**substitutions)
        command = command_template.format(**substitutions)
        completed = subprocess.run(
            ["ssh", "-i", str(args.identity_file), "-o", "BatchMode=yes", host, command],
            check=False, capture_output=True, text=True, timeout=30,
        )
        observations[node_id] = {
            "host": host,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    document = {
        "schema_version": "fable.physical_profile_inventory.v1",
        "profile_version": "physical-inventory-v1",
        "measured_at": datetime.now(UTC).isoformat(),
        "headline_evaluation_input": False,
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    failed = [key for key, value in observations.items() if value["returncode"]]
    print(json.dumps({"output": str(args.output), "failed": failed}, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
