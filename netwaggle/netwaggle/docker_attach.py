from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .util import NetWaggleError, run, safe_ifname


@dataclass(frozen=True)
class DockerAttachment:
    name: str
    anchor_container: str
    switch: str
    ip: str
    gateway: str
    mtu: int = 1500
    container_ifname: str = "netwaggle0"
    switch_ifname: str | None = None

    @property
    def sw_ifname(self) -> str:
        return self.switch_ifname or safe_ifname("nw", self.name)

    @property
    def ct_tmp_ifname(self) -> str:
        return safe_ifname("ct", self.name)


def docker_pid(container_name: str) -> int:
    proc = run(["docker", "inspect", "--format", "{{json .State}}", container_name], capture=True)
    try:
        state = json.loads(proc.stdout)
        pid = int(state.get("Pid", 0))
        running = bool(state.get("Running", False))
    except Exception as exc:  # pragma: no cover - defensive error message
        raise NetWaggleError(f"Could not parse docker inspect for {container_name}: {exc}") from exc
    if not running or pid <= 1:
        raise NetWaggleError(f"Container {container_name!r} is not running or has no usable PID. Start docker compose first.")
    return pid


def attach_container(att: DockerAttachment) -> None:
    """Attach a container network namespace to an OVS switch using a veth pair."""
    pid = docker_pid(att.anchor_container)
    sw = att.sw_ifname
    ct = att.ct_tmp_ifname

    # Best-effort cleanup of stale interfaces from earlier failed runs.
    run(["ip", "link", "del", sw], check=False)
    run(["ip", "link", "del", ct], check=False)
    run(["ovs-vsctl", "--if-exists", "del-port", att.switch, sw], check=False)
    run(["nsenter", "-t", str(pid), "-n", "ip", "link", "del", att.container_ifname], check=False)

    run(["ip", "link", "add", sw, "type", "veth", "peer", "name", ct])
    run(["ip", "link", "set", sw, "mtu", str(att.mtu)])
    run(["ip", "link", "set", ct, "mtu", str(att.mtu)])
    run(["ovs-vsctl", "add-port", att.switch, sw])
    run(["ip", "link", "set", sw, "up"])

    run(["ip", "link", "set", ct, "netns", str(pid)])
    run(["nsenter", "-t", str(pid), "-n", "ip", "link", "set", "lo", "up"])
    run(["nsenter", "-t", str(pid), "-n", "ip", "link", "set", ct, "name", att.container_ifname])
    run(["nsenter", "-t", str(pid), "-n", "ip", "addr", "flush", "dev", att.container_ifname])
    run(["nsenter", "-t", str(pid), "-n", "ip", "addr", "add", att.ip, "dev", att.container_ifname])
    run(["nsenter", "-t", str(pid), "-n", "ip", "link", "set", att.container_ifname, "up"])
    run(["nsenter", "-t", str(pid), "-n", "ip", "route", "replace", "default", "via", att.gateway])


def detach_container(att: DockerAttachment) -> None:
    """Best-effort detach. Safe to call even if the container is gone."""
    run(["ovs-vsctl", "--if-exists", "del-port", att.switch, att.sw_ifname], check=False)
    run(["ip", "link", "del", att.sw_ifname], check=False)
    try:
        pid = docker_pid(att.anchor_container)
    except NetWaggleError:
        return
    run(["nsenter", "-t", str(pid), "-n", "ip", "link", "del", att.container_ifname], check=False)
