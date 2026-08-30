#!/usr/bin/env python3
"""Read-only endpoint and input readiness checks for one deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import yaml

from evaluation.deployment import load_runtime_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("deployment", type=Path)
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()
    state = load_runtime_state(args.deployment)
    raw = yaml.safe_load(args.deployment.read_text(encoding="utf-8")) or {}
    checks = []
    for node_id, spec in raw.get("nodes", {}).items():
        host, port = spec.get("agent_host"), spec.get("agent_port", 8765)
        if host is None:
            checks.append({"node_id": node_id, "status": "NO_ENDPOINT"})
            continue
        try:
            with socket.create_connection((str(host), int(port)), timeout=args.timeout):
                status, error = "READY", None
        except OSError as exc:
            status, error = "UNREACHABLE", f"{type(exc).__name__}: {exc}"
        checks.append({"node_id": node_id, "host": host, "port": port, "status": status, "error": error})
    result = {
        "schema_version": "fable.deployment_preflight.v1",
        "deployment": str(args.deployment.resolve()),
        "node_count": len(state.nodes),
        "source_count": len(state.sources),
        "checks": checks,
        "ready": bool(checks) and all(row["status"] == "READY" for row in checks),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
