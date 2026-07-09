#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re

from .topology import NetWaggleTopology
from .util import load_json, run, sudo_check


def cleanup_from_topology(path: str) -> None:
    topo = NetWaggleTopology.from_dict(load_json(path))
    for att in topo.attachments:
        run(["ovs-vsctl", "--if-exists", "del-port", att.switch, att.sw_ifname], check=False)
        run(["ip", "link", "del", att.sw_ifname], check=False)
    run(["ovs-vsctl", "--if-exists", "del-port", topo.gateway.switch, topo.gateway.switch_ifname], check=False)
    run(["ip", "link", "del", topo.gateway.host_ifname], check=False)
    run(["ip", "link", "del", topo.gateway.switch_ifname], check=False)


def cleanup_common() -> None:
    # Best-effort cleanup for common stale interfaces if a topology file is not available.
    proc = run(["ip", "-o", "link", "show"], capture=True, check=False)
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            m = re.match(r"\d+: ([^:@]+)", line)
            if not m:
                continue
            name = m.group(1)
            if name.startswith(("nw", "ct")) or name in {"nwg-host", "nwg-sw"}:
                run(["ip", "link", "del", name], check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Best-effort NetWaggle cleanup.")
    ap.add_argument("--topology", help="Topology JSON used for the run.")
    args = ap.parse_args()
    sudo_check()
    if args.topology:
        cleanup_from_topology(args.topology)
    cleanup_common()
    run(["mn", "-c"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
