#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from .docker_attach import attach_container, detach_container
from .dynamic_control import DynamicProfileServer, load_condition_map
from .external_proxy import ExternalProxyManager
from .topology import NetWaggleTopology
from .util import NetWaggleError, load_json, run, sudo_check

try:
    from mininet.clean import cleanup as mn_cleanup
    from mininet.link import TCLink
    from mininet.log import setLogLevel
    from mininet.net import Mininet
    from mininet.node import Controller, OVSKernelSwitch
except Exception:  # pragma: no cover - import is expected to fail outside Mininet hosts
    mn_cleanup = None
    TCLink = None
    setLogLevel = None
    Mininet = None
    Controller = None
    OVSKernelSwitch = None


def ensure_mininet_imports() -> None:
    if Mininet is None:
        raise NetWaggleError(
            "Could not import Mininet. Install Mininet/Mininet-WiFi on the host, "
            "then run this script with sudo."
        )


def attach_gateway(gw) -> None:
    run(["ip", "link", "del", gw.host_ifname], check=False)
    run(["ip", "link", "del", gw.switch_ifname], check=False)
    run(["ovs-vsctl", "--if-exists", "del-port", gw.switch, gw.switch_ifname], check=False)
    run(["ip", "link", "add", gw.host_ifname, "type", "veth", "peer", "name", gw.switch_ifname])
    run(["ip", "link", "set", gw.host_ifname, "mtu", str(gw.mtu)])
    run(["ip", "link", "set", gw.switch_ifname, "mtu", str(gw.mtu)])
    run(["ip", "addr", "flush", "dev", gw.host_ifname], check=False)
    run(["ip", "addr", "add", gw.ip, "dev", gw.host_ifname])
    run(["ip", "link", "set", gw.host_ifname, "up"])
    run(["ovs-vsctl", "add-port", gw.switch, gw.switch_ifname])
    run(["ip", "link", "set", gw.switch_ifname, "up"])


def detach_gateway(gw) -> None:
    run(["ovs-vsctl", "--if-exists", "del-port", gw.switch, gw.switch_ifname], check=False)
    run(["ip", "link", "del", gw.host_ifname], check=False)
    run(["ip", "link", "del", gw.switch_ifname], check=False)


def _dpid_for_index(index: int) -> str:
    """Return a stable OpenFlow datapath ID for a configured switch.

    Mininet can derive DPIDs automatically only for canonical switch names such
    as s1, s2, or s23. NetWaggle uses descriptive switch names like
    s_orin11 and s_cloud so experiment configs remain readable. Explicitly
    assigning DPIDs lets us keep those readable names while satisfying
    Mininet/OVS.
    """
    return f"{index + 1:016x}"


def build_mininet(topo: NetWaggleTopology):
    ensure_mininet_imports()
    setLogLevel("info")
    net = Mininet(controller=Controller, switch=OVSKernelSwitch, link=TCLink, autoSetMacs=True, build=False)
    controller = net.addController("c0")
    switches = {
        name: net.addSwitch(name, dpid=_dpid_for_index(idx))
        for idx, name in enumerate(topo.switches)
    }
    for link in topo.links:
        if link.src not in switches or link.dst not in switches:
            raise NetWaggleError(f"Link references unknown switch: {link.src} -> {link.dst}")
        net.addLink(switches[link.src], switches[link.dst], **link.tc_params())
    net.build()
    controller.start()
    for sw in switches.values():
        sw.start([controller])
    return net


def run_topology(
    topo: NetWaggleTopology,
    *,
    hold: bool,
    no_cli: bool,
    base_topology: NetWaggleTopology,
    condition_profiles: dict[str, Path] | None = None,
    control_socket: Path | None = None,
    proxy_metrics: Path | None = None,
) -> None:
    net = None
    profile_server = None
    proxy_manager = None
    should_stop = False

    def _stop(_sig=None, _frame=None):
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        net = build_mininet(topo)
        attach_gateway(topo.gateway)
        for att in topo.attachments:
            attach_container(att)
            print(f"Attached {att.anchor_container} as {att.ip} on {att.switch} via {att.container_ifname}", flush=True)
        print(f"Host-side MQTT/gateway IP is {topo.gateway.ip_without_prefix}", flush=True)
        if topo.external_proxies:
            proxy_manager = ExternalProxyManager(topo.external_proxies, proxy_metrics)
            proxy_manager.start()
            print(f"External physical proxies: {len(topo.external_proxies)}", flush=True)
        if condition_profiles is not None and control_socket is not None:
            profile_server = DynamicProfileServer(
                net=net,
                base_topology=base_topology,
                condition_profiles=condition_profiles,
                socket_path=control_socket,
            )
            profile_server.start()
            print(f"Dynamic profile control socket: {control_socket}", flush=True)

        if not no_cli:
            from mininet.cli import CLI
            CLI(net)
        elif hold:
            print("NetWaggle is running. Press Ctrl-C to stop.", flush=True)
            while not should_stop:
                time.sleep(1.0)
    finally:
        print("Cleaning up NetWaggle attachments...", flush=True)
        if profile_server is not None:
            profile_server.stop()
        if proxy_manager is not None:
            proxy_manager.stop()
        for att in reversed(topo.attachments):
            detach_container(att)
        detach_gateway(topo.gateway)
        if net is not None:
            net.stop()
        if mn_cleanup is not None:
            mn_cleanup()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a NetWaggle Mininet topology and attach Docker node namespaces.")
    ap.add_argument("--topology", required=True, help="Topology JSON file.")
    ap.add_argument("--profile", help="Optional profile JSON that replaces topology links with network-condition links.")
    ap.add_argument("--hold", action="store_true", help="Keep running without opening the Mininet CLI.")
    ap.add_argument("--no-cli", action="store_true", help="Do not open the Mininet CLI.")
    ap.add_argument("--dry-run", action="store_true", help="Load and validate config, but do not touch networking.")
    ap.add_argument("--control-socket", type=Path, help="Absolute Unix socket for typed runtime profile changes.")
    ap.add_argument("--condition-map", type=Path, help="Allowlisted condition-to-profile map for dynamic control.")
    ap.add_argument("--proxy-metrics", type=Path, help="Optional JSONL metrics path for external physical proxies.")
    args = ap.parse_args(argv)

    try:
        base_topology = NetWaggleTopology.from_dict(load_json(args.topology))
        profile = load_json(args.profile) if args.profile else None
        topo = base_topology.with_profile(profile)
        if bool(args.control_socket) != bool(args.condition_map):
            raise NetWaggleError("--control-socket and --condition-map must be supplied together")
        if args.control_socket is not None and not args.control_socket.is_absolute():
            raise NetWaggleError("--control-socket must be absolute")
        condition_profiles = load_condition_map(args.condition_map) if args.condition_map else None
        print(f"Loaded topology: {topo.name}")
        print(f"Switches: {', '.join(topo.switches)}")
        print(f"Links: {len(topo.links)}; logical node attachments: {len(topo.attachments)}")
        print(f"External physical proxies: {len(topo.external_proxies)}")
        if condition_profiles is not None:
            print(f"Dynamic conditions: {', '.join(sorted(condition_profiles))}")
            print(f"Dynamic profile control socket: {args.control_socket}")
        if args.dry_run:
            return 0
        sudo_check()
        run_topology(
            topo,
            hold=args.hold or args.no_cli,
            no_cli=args.no_cli,
            base_topology=base_topology,
            condition_profiles=condition_profiles,
            control_socket=args.control_socket,
            proxy_metrics=args.proxy_metrics,
        )
        return 0
    except NetWaggleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
