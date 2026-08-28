#!/usr/bin/env python3
"""Generate a NetWaggle-aware compose file from the replay compose file.

The generated compose file keeps the replay service definitions intact, but
places each logical-node bundle into a shared namespace owned by an anchor
container named netwaggle-node-<node>. NetWaggle then attaches those anchor
namespaces to Mininet.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


ANCHOR_IMAGE = "fable/netwaggle-anchor:alpine3.20"
ANCHOR_COMMAND = [
    "sh", "-c",
    "iperf3 --server --daemon; trap : TERM INT; sleep infinity & wait",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def env_list_to_map(env: Any) -> dict[str, str]:
    if env is None:
        return {}
    if isinstance(env, dict):
        return {str(k): str(v) for k, v in env.items()}
    out: dict[str, str] = {}
    for item in env:
        text = str(item)
        if "=" in text:
            k, v = text.split("=", 1)
            out[k] = v
    return out


def env_map_to_list(env: dict[str, str]) -> list[str]:
    return [f"{k}={v}" for k, v in env.items()]


def service_matches(service_name: str, service: dict[str, Any], containers: set[str], services: set[str]) -> bool:
    return service_name in services or str(service.get("container_name", "")) in containers


def anchor_service(anchor_name: str) -> dict[str, Any]:
    return {
        "image": ANCHOR_IMAGE,
        "build": {"context": "../netwaggle", "dockerfile": "Dockerfile.anchor"},
        "container_name": anchor_name,
        "command": ANCHOR_COMMAND,
        "network_mode": "none",
        "healthcheck": {
            "test": ["CMD-SHELL", "command -v ping && command -v iperf3"],
            "interval": "5s",
            "timeout": "2s",
            "retries": 12,
        },
    }


def transform_compose(compose: dict[str, Any], node_map: dict[str, Any]) -> dict[str, Any]:
    mqtt_host_ip = node_map.get("mqtt_host_ip") or node_map.get("gateway", {}).get("ip", "10.255.0.1/16").split("/", 1)[0]
    out = copy.deepcopy(compose)
    services = out.setdefault("services", {})
    original_services = copy.deepcopy(services)

    for node in node_map.get("logical_nodes", node_map.get("nodes", [])):
        name = node["name"]
        anchor = node.get("anchor_container", f"netwaggle-node-{name}")
        anchor_key = node.get("anchor_service", anchor)
        containers = set(node.get("containers", []))
        service_names = set(node.get("services", []))
        services[anchor_key] = anchor_service(anchor)

        matched_any = False
        for svc_name, svc in original_services.items():
            if not isinstance(svc, dict):
                continue
            if not service_matches(svc_name, svc, containers, service_names):
                continue
            matched_any = True
            target = services[svc_name]
            target["network_mode"] = f"service:{anchor_key}"
            # These conflict with or are meaningless under network_mode:service.
            target.pop("networks", None)
            target.pop("ports", None)
            target.pop("hostname", None)
            target.pop("extra_hosts", None)
            deps = target.setdefault("depends_on", [])
            if isinstance(deps, list):
                if anchor_key not in deps:
                    deps.insert(0, anchor_key)
            elif isinstance(deps, dict):
                deps[anchor_key] = {"condition": "service_started"}
            else:
                target["depends_on"] = [anchor_key]

            env = env_list_to_map(target.get("environment"))
            if "MQTT_HOST_IP" in env:
                env["MQTT_HOST_IP"] = f"${{MQTT_HOST_IP:-{mqtt_host_ip}}}"
            if "MQTT_HOST" in env:
                env["MQTT_HOST"] = f"${{MQTT_HOST:-{mqtt_host_ip}}}"

            # Evaluation-mode visibility: make Docker logs show which MQTT/local
            # messages are being emitted while replay runs through NetWaggle.
            # Operators can override these at compose runtime, e.g.
            #   IOBT_LOG_NET_PUBLISH=false docker compose ...
            env.setdefault("IOBT_LOG_NET_PUBLISH", "${IOBT_LOG_NET_PUBLISH:-true}")
            env.setdefault("IOBT_LOG_LOCAL_PUBLISH", "${IOBT_LOG_LOCAL_PUBLISH:-true}")
            env.setdefault("IOBT_LOG_NET_PUBLISH_EVERY_N", "${IOBT_LOG_NET_PUBLISH_EVERY_N:-1}")
            env.setdefault("IOBT_LOG_LOCAL_PUBLISH_EVERY_N", "${IOBT_LOG_LOCAL_PUBLISH_EVERY_N:-30}")
            env.setdefault("IOBT_PUBLISH_READINESS", "${IOBT_PUBLISH_READINESS:-true}")
            env.setdefault("IOBT_READINESS_RETAIN", "${IOBT_READINESS_RETAIN:-true}")
            if env:
                target["environment"] = env_map_to_list(env)

        if not matched_any:
            print(f"WARNING: no replay services matched logical node {name!r}. Check node_map containers/services.")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate compose.netwaggle.yaml from compose.replay.yaml.")
    ap.add_argument("--compose-in", default="compose.replay.yaml")
    ap.add_argument("--compose-out", default="compose.netwaggle.yaml")
    ap.add_argument("--node-map", required=True, help="NetWaggle node map/topology JSON.")
    args = ap.parse_args()

    compose = load_yaml(Path(args.compose_in))
    with Path(args.node_map).open("r", encoding="utf-8") as f:
        node_map = json.load(f)
    out = transform_compose(compose, node_map)
    write_yaml(Path(args.compose_out), out)
    print(f"Wrote {args.compose_out}")
    print("Start anchors/replay with: docker compose -f compose.netwaggle.yaml up --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
