#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .topology import NetWaggleTopology
from .util import load_json, run


def iface_stats(iface: str) -> dict[str, int | str]:
    base = Path("/sys/class/net") / iface / "statistics"
    out: dict[str, int | str] = {"iface": iface}
    for key in ["rx_bytes", "tx_bytes", "rx_packets", "tx_packets", "rx_dropped", "tx_dropped", "rx_errors", "tx_errors"]:
        try:
            out[key] = int((base / key).read_text().strip())
        except Exception:
            out[key] = -1
    qdisc = run(["tc", "-s", "qdisc", "show", "dev", iface], capture=True, check=False)
    if qdisc.returncode == 0:
        out["qdisc"] = qdisc.stdout.strip()
    return out


def collect_once(topo: NetWaggleTopology) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    now = time.time()
    ifaces = [topo.gateway.host_ifname, topo.gateway.switch_ifname]
    ifaces.extend(att.sw_ifname for att in topo.attachments)
    for iface in ifaces:
        row = iface_stats(iface)
        row["ts"] = now
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect simple NetWaggle interface/qdisc metrics as JSONL.")
    ap.add_argument("--topology", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    topo = NetWaggleTopology.from_dict(load_json(args.topology))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        while True:
            for row in collect_once(topo):
                f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            if args.once:
                break
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

