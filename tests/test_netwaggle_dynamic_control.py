from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from netwaggle.netwaggle.dynamic_control import (
    DynamicProfileServer,
    apply_runtime_profile,
    load_condition_map,
)
from netwaggle.netwaggle.topology import NetWaggleTopology
from netwaggle.netwaggle.util import NetWaggleError, load_json


ROOT = Path(__file__).resolve().parents[1]


class FakeNode:
    def __init__(self, name):
        self.name = name


class FakeIntf:
    def __init__(self, node, name):
        self.node = FakeNode(node)
        self.name = name
        self.configurations = []
        self.up = True

    def config(self, **parameters):
        self.configurations.append(parameters)

    def ifconfig(self, state):
        self.up = state == "up"

    def isUp(self):
        return self.up


class FakeLink:
    def __init__(self, left, right, index):
        self.intf1 = FakeIntf(left, f"fake{index}a")
        self.intf2 = FakeIntf(right, f"fake{index}b")


class FakeNet:
    def __init__(self, topology):
        self.links = [
            FakeLink(link.src, link.dst, index)
            for index, link in enumerate(topology.links)
        ]


def _topology():
    return NetWaggleTopology.from_dict(
        load_json(ROOT / "netwaggle/configs/fable_single_host.json")
    )


def test_runtime_profile_reconfigures_existing_links_without_rebuilding() -> None:
    base = _topology()
    degraded = base.with_profile(
        load_json(ROOT / "netwaggle/configs/profiles/cloud_degraded.json")
    )
    net = FakeNet(base)

    measurements = apply_runtime_profile(net, degraded)

    assert measurements["configured_link_count"] == len(base.links)
    assert not measurements["qdisc_validated"]
    assert all(link.intf1.configurations for link in net.links)
    assert all(link.intf2.configurations for link in net.links)
    cloud_link = next(
        link
        for link in net.links
        if {link.intf1.node.name, link.intf2.node.name}
        == {"s_edge", "s_cloud"}
    )
    assert cloud_link.intf1.configurations[-1]["delay"] == "120ms"
    assert cloud_link.intf1.configurations[-1]["bw"] == 5


def test_runtime_profile_cannot_change_topology_structure() -> None:
    base = _topology()
    incomplete = NetWaggleTopology(
        name="incomplete",
        switches=base.switches,
        links=base.links[:-1],
        gateway=base.gateway,
        attachments=base.attachments,
    )
    with pytest.raises(NetWaggleError, match="every existing topology link"):
        apply_runtime_profile(FakeNet(base), incomplete)


def test_condition_map_is_confined_to_its_config_root() -> None:
    mapping = load_condition_map(
        ROOT / "netwaggle/configs/fable_dynamic_conditions.json"
    )
    assert set(mapping) == {"N0", "W1", "W2", "L1"}
    assert all(path.parent.name == "profiles" for path in mapping.values())


def test_socket_helper_applies_only_mapped_condition(tmp_path) -> None:
    base = _topology()
    socket_path = tmp_path / "control.sock"
    server = DynamicProfileServer(
        net=FakeNet(base),
        base_topology=base,
        condition_profiles=load_condition_map(
            ROOT / "netwaggle/configs/fable_dynamic_conditions.json"
        ),
        socket_path=socket_path,
        qdisc_inspector=lambda interface: [
            {"kind": "netem", "dev": interface, "options": {"delay": "120ms"}}
        ],
    )
    server.start()
    try:
        completed = subprocess.run(
            (
                sys.executable,
                str(ROOT / "netwaggle/scripts/fable_netwaggle_helper.py"),
                "--kind",
                "NETWORK_PROFILE",
                "--target",
                "site_to_cloud",
                "--condition",
                "W1",
                "--action",
                "APPLY",
                "--condition-epoch",
                "1",
                "--socket",
                str(socket_path),
            ),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    finally:
        server.stop()

    assert completed.returncode == 0
    response = json.loads(completed.stdout)
    assert response["validated"]
    assert response["profile"] == "cloud_degraded"
    assert response["measurements"]["qdisc_validated"]
    assert response["measurements"]["qdisc_interface_count"] == 2 * len(base.links)


def test_qdisc_readback_must_return_state() -> None:
    base = _topology()
    with pytest.raises(NetWaggleError, match="returned no state"):
        apply_runtime_profile(
            FakeNet(base),
            base,
            qdisc_inspector=lambda _interface: [],
        )


def test_runtime_profile_applies_directional_parameters() -> None:
    base = _topology()
    first = base.links[0]
    directional = NetWaggleTopology(
        name=base.name,
        switches=base.switches,
        links=(
            type(first)(
                src=first.src,
                dst=first.dst,
                bw=10,
                delay="10ms",
                reverse={"bw": 2, "delay": "80ms"},
            ),
            *base.links[1:],
        ),
        gateway=base.gateway,
        attachments=base.attachments,
    )
    net = FakeNet(base)
    apply_runtime_profile(net, directional)
    live = net.links[0]
    assert live.intf1.configurations[-1]["bw"] == 10
    assert live.intf2.configurations[-1]["bw"] == 2
    assert live.intf2.configurations[-1]["delay"] == "80ms"


def test_link_failure_and_restore_are_read_back() -> None:
    base = _topology()
    net = FakeNet(base)
    server = DynamicProfileServer(
        net=net,
        base_topology=base,
        condition_profiles=load_condition_map(
            ROOT / "netwaggle/configs/fable_dynamic_conditions.json"
        ),
        socket_path=Path("/tmp/unused-netwaggle-test.sock"),
        qdisc_inspector=lambda _interface: [{"kind": "netem"}],
    )
    target = f"link:{base.links[0].src}:{base.links[0].dst}"
    failed = server.handle_request({
        "schema_version": "netwaggle.dynamic_control.v1",
        "kind": "LINK_STATE",
        "target": target,
        "condition": "L1",
        "action": "FAIL",
        "condition_epoch": 1,
    })
    assert failed["validated"]
    assert failed["measurements"]["interface_up"] == [False, False]
    restored = server.handle_request({
        "schema_version": "netwaggle.dynamic_control.v1",
        "kind": "LINK_STATE",
        "target": target,
        "condition": "N0",
        "action": "RESTORE",
        "condition_epoch": 2,
    })
    assert restored["measurements"]["interface_up"] == [True, True]


@pytest.mark.parametrize(
    ("target", "condition"),
    (("site_backbone", "N2"), ("sensor_uplink:s_mobile_archive_6", "L1")),
)
def test_updated_rq3a_targets_reconfigure_only_the_selected_path(
    tmp_path, target, condition
) -> None:
    base = NetWaggleTopology.from_dict(
        load_json(ROOT / "netwaggle/configs/site_evaluation_29node.json")
    )
    net = FakeNet(base)
    server = DynamicProfileServer(
        net=net,
        base_topology=base,
        condition_profiles=load_condition_map(
            ROOT
            / "netwaggle/configs/profiles/site_evaluation_29node/CONDITION_MAP.json"
        ),
        socket_path=tmp_path / "unused.sock",
        qdisc_inspector=lambda _interface: [{"kind": "netem"}],
    )
    response = server.handle_request(
        {
            "schema_version": "netwaggle.dynamic_control.v1",
            "kind": "NETWORK_PROFILE",
            "target": target,
            "condition": condition,
            "action": "APPLY",
            "condition_epoch": 1,
        }
    )

    assert response["validated"]
    assert response["measurements"]["configured_link_count"] == len(base.links)
    changed = []
    for authored, live in zip(base.links, net.links):
        latest = live.intf1.configurations[-1]
        if latest.get("bw") != authored.bw or latest.get("delay") != authored.delay:
            changed.append({authored.src, authored.dst})
    expected = (
        {"s_edge", "s_site"}
        if target == "site_backbone"
        else {"s_mob6", "s_edge"}
    )
    assert changed == [expected]
