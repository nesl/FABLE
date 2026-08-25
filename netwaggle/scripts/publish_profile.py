#!/usr/bin/env python3
"""Publish the active NetWaggle topology/profile to MQTT for the web UI.

This is intentionally host-side and non-privileged. It publishes retained metadata
under /netwaggle/profile so the UI can show configured per-node path delays next
to the observed latency probes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

try:
    import paho.mqtt.client as mqtt
except Exception as exc:  # pragma: no cover - runtime dependency message
    raise SystemExit("Missing paho-mqtt. Install with: /usr/bin/python3 -m pip install paho-mqtt") from exc


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_delay_ms(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*(us|µs|ms|s)?", text)
    if not m:
        return 0.0
    n = float(m.group(1))
    unit = m.group(2) or "ms"
    if unit in {"us", "µs"}:
        return n / 1000.0
    if unit == "s":
        return n * 1000.0
    return n


def merged_links(topology: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    base: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for link in topology.get("links", []):
        a = link.get("from") or link.get("src")
        b = link.get("to") or link.get("dst")
        if not a or not b:
            continue
        key = tuple(sorted((str(a), str(b))))
        base[key] = dict(link)
    for link in profile.get("links", []):
        a = link.get("from") or link.get("src")
        b = link.get("to") or link.get("dst")
        if not a or not b:
            continue
        key = tuple(sorted((str(a), str(b))))
        old = base.get(key, {})
        old.update(link)
        base[key] = old
    return list(base.values())


def shortest_path(nodes: List[str], links: List[Dict[str, Any]], src: str, dst: str) -> Tuple[List[str], float, Any, float]:
    graph: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {n: [] for n in nodes}
    for link in links:
        a = str(link.get("from") or link.get("src"))
        b = str(link.get("to") or link.get("dst"))
        graph.setdefault(a, []).append((b, link))
        graph.setdefault(b, []).append((a, link))

    dist: Dict[str, float] = {src: 0.0}
    prev: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    unseen = set(graph)
    while unseen:
        cur = min(unseen, key=lambda n: dist.get(n, math.inf))
        if dist.get(cur, math.inf) == math.inf:
            break
        unseen.remove(cur)
        if cur == dst:
            break
        for nxt, link in graph.get(cur, []):
            nd = dist[cur] + parse_delay_ms(link.get("delay"))
            if nd < dist.get(nxt, math.inf):
                dist[nxt] = nd
                prev[nxt] = (cur, link)

    if dst not in dist:
        return [], math.inf, None, 0.0

    path = [dst]
    bottleneck_bw = None  # type: Any
    loss_sum = 0.0
    cur = dst
    while cur != src:
        p, link = prev[cur]
        path.append(p)
        bw = link.get("bw")
        if bw is not None:
            try:
                bottleneck_bw = float(bw) if bottleneck_bw is None else min(bottleneck_bw, float(bw))
            except Exception:
                pass
        try:
            loss_sum += float(link.get("loss") or 0.0)
        except Exception:
            pass
        cur = p
    path.reverse()
    return path, dist[dst], bottleneck_bw, loss_sum


def build_payload(topology: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    gateway_switch = str(topology.get("gateway", {}).get("switch") or "")
    switches = [str(s) for s in topology.get("switches", [])]
    links = merged_links(topology, profile)
    node_payload: Dict[str, Any] = {}

    for node in topology.get("logical_nodes", []):
        name = str(node.get("name"))
        sw = str(node.get("switch"))
        path, delay_ms, bw, loss = shortest_path(switches, links, sw, gateway_switch)
        node_payload[name] = {
            "node": name,
            "fable_node_id": node.get("fable_node_id"),
            "tier": node.get("tier"),
            "anchor_container": node.get("anchor_container") or f"netwaggle-node-{name}",
            "ip": node.get("ip"),
            "switch": sw,
            "gateway_switch": gateway_switch,
            "path": path,
            "configured_one_way_ms": None if math.isinf(delay_ms) else delay_ms,
            "configured_rtt_ms": None if math.isinf(delay_ms) else 2.0 * delay_ms,
            "bottleneck_bw_mbps": bw,
            "path_loss_percent_sum": loss,
        }

    return {
        "source": "netwaggle_profile_publisher",
        "published_at": time.time(),
        "topology_name": topology.get("name"),
        "profile_name": profile.get("name"),
        "profile_description": profile.get("description"),
        "mqtt_host_ip": topology.get("mqtt_host_ip") or topology.get("gateway", {}).get("ip"),
        "gateway": topology.get("gateway", {}),
        "fable_gateway_node_id": topology.get("fable_gateway_node_id"),
        "nodes": node_payload,
        "links": links,
        "note": "Configured delay is the Mininet/TC path delay from the logical node switch to the gateway switch; observed probes include MQTT broker and web-UI receive overhead.",
    }


def make_client(client_id: str) -> mqtt.Client:
    try:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    except AttributeError:
        return mqtt.Client(client_id=client_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--mqtt-host", default="localhost")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--topic", default="/netwaggle/profile")
    ap.add_argument("--retain", action="store_true", default=True)
    ap.add_argument("--no-retain", dest="retain", action="store_false")
    args = ap.parse_args()

    payload = build_payload(load_json(args.topology), load_json(args.profile))
    client = make_client("netwaggle-profile-publisher")
    client.connect(args.mqtt_host, args.mqtt_port, keepalive=30)
    client.loop_start()
    info = client.publish(args.topic, json.dumps(payload, separators=(",", ":")), qos=1, retain=args.retain)
    info.wait_for_publish(timeout=5)
    client.loop_stop()
    client.disconnect()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
