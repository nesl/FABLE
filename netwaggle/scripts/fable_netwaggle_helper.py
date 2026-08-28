#!/usr/bin/env python3
"""Narrow client for the root-side NetWaggle dynamic control socket."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import re


ALLOWED_CONDITIONS = {"N0", "N2", "W1", "W2", "L1"}
CONTROL_SOCKET = "/run/netwaggle/fable-control.sock"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("NETWORK_PROFILE", "LINK_STATE"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--condition", choices=tuple(sorted(ALLOWED_CONDITIONS)), required=True)
    parser.add_argument("--action", choices=("APPLY", "RESTORE", "FAIL"), required=True)
    parser.add_argument("--condition-epoch", type=int, required=True)
    parser.add_argument(
        "--socket",
        default=CONTROL_SOCKET,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.condition_epoch < 0:
        parser.error("--condition-epoch cannot be negative")
    if args.kind == "NETWORK_PROFILE":
        if not (
            args.target in {"site_to_cloud", "site_backbone"}
            or re.fullmatch(
                r"sensor_uplink:s_(?:orin(?:[1-9]|[12][0-9]|30)|mobile_archive_[1-6])",
                args.target,
            )
        ):
            parser.error("invalid network-profile target")
        if args.action not in {"APPLY", "RESTORE"}:
            parser.error("NETWORK_PROFILE accepts APPLY or RESTORE")
    else:
        match = re.fullmatch(
            r"link:(s_(?:orin(?:[1-9]|[12][0-9]|30)|mob[1-6])):(s_edge)",
            args.target,
        ) or re.fullmatch(
            r"link:(s_edge):(s_(?:orin(?:[1-9]|[12][0-9]|30)|mob[1-6]))",
            args.target,
        )
        if match is None:
            parser.error("link state is restricted to a sensor-to-edge uplink")
        if args.action not in {"FAIL", "RESTORE"}:
            parser.error("LINK_STATE accepts FAIL or RESTORE")
    if not os.path.isabs(args.socket):
        parser.error("--socket must be absolute")
    if os.geteuid() == 0 and args.socket != CONTROL_SOCKET:
        parser.error("root helper cannot override the fixed control socket")
    request = {
        "schema_version": "netwaggle.dynamic_control.v1",
        "kind": args.kind,
        "target": args.target,
        "condition": args.condition,
        "action": args.action,
        "condition_epoch": args.condition_epoch,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        # A measured full-topology profile application can take tens of
        # seconds while tc/OVS state settles and is inspected.
        client.settimeout(60)
        client.connect(args.socket)
        client.sendall((json.dumps(request, sort_keys=True) + "\n").encode("utf-8"))
        payload = bytearray()
        while b"\n" not in payload:
            chunk = client.recv(4096)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > 1024 * 1024:
                raise RuntimeError("NetWaggle response exceeds 1 MiB")
    response = json.loads(bytes(payload).split(b"\n", 1)[0].decode("utf-8"))
    print(json.dumps(response, sort_keys=True))
    return 0 if response.get("validated") is True else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "netwaggle.dynamic_control_result.v1",
                    "validated": False,
                    "measurements": {},
                    "reason": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            )
        )
        raise SystemExit(4)
