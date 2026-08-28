"""Deterministic canonical deployment and NetWaggle profile generation."""

from __future__ import annotations

from typing import Any


MIN_DEVICES = 5
MAX_DEVICES = 20
# Linux IFNAMSIZ is 16 including the terminating NUL. Mininet derives link
# names such as ``<switch>-eth1``, so descriptive switch identifiers must
# leave room for that suffix.
SITE_LOCAL_SWITCH = "s_site"


def build_site_local_deployment(
    device_count: int = MAX_DEVICES,
    *,
    first_device_number: int = 11,
) -> dict[str, Any]:
    if not MIN_DEVICES <= device_count <= MAX_DEVICES:
        raise ValueError(
            f"device_count must be between {MIN_DEVICES} and {MAX_DEVICES}"
        )
    if first_device_number < 1 or first_device_number + device_count - 1 > 239:
        raise ValueError("device address range must remain within 1..239")

    devices = []
    for number in range(first_device_number, first_device_number + device_count):
        name = f"orin{number}"
        devices.append(
            {
                "name": name,
                "tier": "embedded",
                "anchor_container": f"netwaggle-node-{name}",
                "switch": f"s_{name}",
                "ip": f"10.255.{number}.2/16",
                "gateway": "10.255.0.1",
                "containers": [
                    f"zed-replay-{name}",
                    f"yolo-detector-{name}",
                    f"respeaker-replay-{name}",
                    f"audio-detector-{name}",
                ],
            }
        )

    infrastructure = [
        {
            "name": "site_local",
            "tier": "server",
            "anchor_container": "netwaggle-node-site-local",
            "switch": SITE_LOCAL_SWITCH,
            "ip": "10.255.240.2/16",
            "gateway": "10.255.0.1",
            "containers": [
                "fable-agent-x86server",
                "fable-identity-x86server",
            ],
        },
        {
            "name": "cloud1",
            "tier": "cloud",
            "anchor_container": "netwaggle-node-cloud1",
            "switch": "s_cloud",
            "ip": "10.255.250.2/16",
            "gateway": "10.255.0.1",
            "containers": [
                "complex-event-detector",
                "fable-agent-cloud1",
                "fable-vlm-cloud1",
            ],
        },
    ]
    switches = [f"s_{node['name']}" for node in devices]
    switches.extend(["s_edge", SITE_LOCAL_SWITCH, "s_cloud"])
    links = [
        {
            "from": node["switch"],
            "to": "s_edge",
            "bw": 100,
            "delay": "5ms",
            "jitter": "1ms",
            "loss": 0.1,
        }
        for node in devices
    ]
    links.extend(
        [
            {
                "from": "s_edge",
                "to": SITE_LOCAL_SWITCH,
                "bw": 1000,
                "delay": "0.5ms",
                "jitter": "0.1ms",
                "loss": 0,
            },
            {
                "from": SITE_LOCAL_SWITCH,
                "to": "s_cloud",
                "bw": 200,
                "delay": "25ms",
                "jitter": "3ms",
                "loss": 0.1,
            },
        ]
    )
    return {
        "schema_version": "fable.deployment_topology.v1",
        "name": f"site-local-{device_count}node",
        "device_count": device_count,
        "gateway": {
            "ip": "10.255.0.1/16",
            "switch": "s_cloud",
            "host_ifname": "nwg-host",
            "switch_ifname": "nwg-sw",
            "mtu": 1500,
        },
        "mqtt_host_ip": "10.255.0.1",
        "switches": switches,
        "links": links,
        "logical_nodes": devices + infrastructure,
    }


def build_network_profile(
    deployment: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    if condition not in {"N0", "W1", "W2", "L1"}:
        raise ValueError("condition must be N0, W1, W2, or L1")
    links = [dict(link) for link in deployment["links"]]
    if condition in {"W1", "W2"}:
        wan = next(
            link
            for link in links
            if {link["from"], link["to"]} == {SITE_LOCAL_SWITCH, "s_cloud"}
        )
        wan.update(
            {
                "bw": 10,
                "delay": "75ms",
                "jitter": "10ms",
                "loss": 1,
                "max_queue_size": 100,
            }
            if condition == "W1"
            else {
                "bw": 2,
                "delay": "150ms",
                "jitter": "25ms",
                "loss": 3,
                "max_queue_size": 50,
            }
        )
    elif condition == "L1":
        selected_switch = deployment["logical_nodes"][0]["switch"]
        uplink = next(
            link
            for link in links
            if {link["from"], link["to"]} == {selected_switch, "s_edge"}
        )
        uplink.update(
            {
                "bw": 10,
                "delay": "50ms",
                "jitter": "15ms",
                "loss": 5,
                "max_queue_size": 100,
            }
        )
    return {
        "schema_version": "netwaggle.profile.v1",
        "name": condition,
        "topology": deployment["name"],
        "links": links,
    }


def validate_unique_network_identities(deployment: dict[str, Any]) -> None:
    nodes = deployment["logical_nodes"]
    for field in ("name", "anchor_container", "ip"):
        values = [node[field] for node in nodes]
        if len(values) != len(set(values)):
            raise ValueError(f"deployment has duplicate {field} values")
    degrees = {switch: 0 for switch in deployment["switches"]}
    for link in deployment["links"]:
        degrees[link["from"]] += 1
        degrees[link["to"]] += 1
    for switch, degree in degrees.items():
        if len(f"{switch}-eth{max(1, degree)}") > 15:
            raise ValueError(
                f"switch name cannot form a Linux interface name: {switch}"
            )
