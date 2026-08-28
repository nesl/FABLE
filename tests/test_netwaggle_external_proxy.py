from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from netwaggle.netwaggle.external_proxy import ExternalProxyManager
from netwaggle.netwaggle.dynamic_control import _retarget_sensor_profile
from netwaggle.netwaggle.topology import ExternalProxy, NetWaggleTopology
from netwaggle.netwaggle.util import load_json


def _free_port() -> int:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    return int(port)


def _echo_server(port: int, ready: threading.Event) -> None:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(1)
    ready.set()
    connection, _ = listener.accept()
    with connection:
        payload = connection.recv(1024)
        connection.sendall(payload.upper())
    listener.close()


def test_external_proxy_forwards_and_records_exact_bytes(tmp_path) -> None:
    target_port = _free_port()
    listen_port = _free_port()
    ready = threading.Event()
    threading.Thread(
        target=_echo_server, args=(target_port, ready), daemon=True
    ).start()
    assert ready.wait(2)
    metrics = tmp_path / "proxy.jsonl"
    manager = ExternalProxyManager(
        [
            ExternalProxy(
                name="test",
                listen_host="127.0.0.1",
                listen_port=listen_port,
                target_host="127.0.0.1",
                target_port=target_port,
            )
        ],
        metrics,
    )
    manager.start()
    with socket.create_connection(("127.0.0.1", listen_port), timeout=2) as client:
        client.sendall(b"physical")
        client.shutdown(socket.SHUT_WR)
        assert client.recv(1024) == b"PHYSICAL"
    deadline = time.monotonic() + 2
    rows = []
    while time.monotonic() < deadline:
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        if any(row["event"] == "CLOSED" for row in rows):
            break
        time.sleep(0.01)
    manager.stop()
    closed = next(row for row in rows if row["event"] == "CLOSED")
    assert closed["client_to_target_bytes"] == len(b"physical")
    assert closed["target_to_client_bytes"] == len(b"PHYSICAL")
    assert closed["error"] is None


def test_physical_external_topology_is_strict_and_complete() -> None:
    topology = NetWaggleTopology.from_dict(
        load_json("netwaggle/configs/physical_external_3node.json")
    )
    assert len(topology.external_proxies) == 4
    assert {item.outbound_namespace_anchor for item in topology.external_proxies} >= {
        "netwaggle-node-physical-rpi",
        "netwaggle-node-physical-jetson",
    }
    document = {
        "name": "bad",
        "gateway": {"ip": "10.0.0.1/24", "switch": "s1"},
        "switches": ["s1"],
        "external_proxies": [{
            "name": "bad", "listen_host": "127.0.0.1", "listen_port": 1,
            "target_host": "127.0.0.1", "target_port": 2,
            "arbitrary_command": "sh",
        }],
    }
    with pytest.raises(ValueError, match="unknown external proxy fields"):
        NetWaggleTopology.from_dict(document)


def test_external_proxy_rejects_unknown_anchor() -> None:
    with pytest.raises(ValueError, match="unknown anchor"):
        NetWaggleTopology.from_dict({
            "name": "bad-anchor",
            "gateway": {"ip": "10.0.0.1/24", "switch": "s1"},
            "switches": ["s1"],
            "external_proxies": [{
                "name": "bad", "listen_host": "127.0.0.1", "listen_port": 1,
                "target_host": "127.0.0.1", "target_port": 2,
                "outbound_namespace_anchor": "not-allowlisted",
            }],
        })


def test_physical_profile_can_target_either_external_uplink() -> None:
    topology = NetWaggleTopology.from_dict(
        load_json("netwaggle/configs/physical_external_3node.json")
    )
    profile = load_json(
        "netwaggle/configs/profiles/physical_external_constrained.json"
    )
    retargeted = _retarget_sensor_profile(topology, profile, "s_jetson")
    changed = next(
        row for row in retargeted["links"]
        if {row["from"], row["to"]} == {"s_jetson", "s_edge"}
    )
    unchanged = next(
        row for row in retargeted["links"]
        if {row["from"], row["to"]} == {"s_rpi", "s_edge"}
    )
    assert changed["bw"] == 10
    assert changed["delay"] == "25ms"
    assert unchanged["bw"] == 1000
