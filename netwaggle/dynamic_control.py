"""Runtime profile control for an already-running NetWaggle topology."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import re
from typing import Any, Callable, Dict, List

from .topology import NetWaggleTopology
from .util import NetWaggleError, load_json


ALLOWED_REQUEST_KEYS = {
    "schema_version",
    "kind",
    "target",
    "condition",
    "action",
    "condition_epoch",
}


# This alias is evaluated at import time even with postponed annotations.
# NetWaggle deliberately uses Ubuntu's system Python for Mininet, which is
# Python 3.8 on the evaluation host and cannot evaluate ``list[...]``.
QdiscInspector = Callable[[str], List[Dict[str, Any]]]


def inspect_qdisc(interface: str) -> list[dict[str, Any]]:
    """Read back kernel qdisc state without accepting caller-controlled argv."""

    completed = subprocess.run(
        ["tc", "-j", "qdisc", "show", "dev", interface],
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0:
        raise NetWaggleError(
            f"tc qdisc inspection failed for {interface}: "
            f"{completed.stderr.strip() or 'unknown error'}"
        )
    document = json.loads(completed.stdout or "[]")
    if not isinstance(document, list) or not document:
        raise NetWaggleError(f"no qdisc state returned for {interface}")
    return document


def apply_runtime_profile(
    net,
    topology: NetWaggleTopology,
    *,
    qdisc_inspector: QdiscInspector | None = None,
) -> dict[str, Any]:
    """Reconfigure existing TCLink interfaces; topology structure cannot change."""

    existing = {
        frozenset((link.intf1.node.name, link.intf2.node.name)): link
        for link in net.links
    }
    configured: list[dict[str, Any]] = []
    requested_keys: set[frozenset[str]] = set()
    for configured_link in topology.links:
        key = frozenset((configured_link.src, configured_link.dst))
        if len(key) != 2:
            raise NetWaggleError("runtime profile links must join two switches")
        requested_keys.add(key)
        try:
            live_link = existing[key]
        except KeyError as exc:
            raise NetWaggleError(
                "runtime profile cannot add a link: "
                f"{configured_link.src}->{configured_link.dst}"
            ) from exc
        forward = configured_link.tc_params()
        reverse = configured_link.tc_params(reverse=True)
        if live_link.intf1.node.name == configured_link.src:
            src_intf, dst_intf = live_link.intf1, live_link.intf2
        else:
            src_intf, dst_intf = live_link.intf2, live_link.intf1
        src_intf.config(**forward)
        dst_intf.config(**reverse)
        configured.append(
            {
                "from": configured_link.src,
                "to": configured_link.dst,
                **forward,
                "forward": forward,
                "reverse": reverse,
                "interfaces": [live_link.intf1.name, live_link.intf2.name],
            }
        )
    if requested_keys != set(existing):
        missing = sorted(
            ":".join(sorted(item)) for item in set(existing) - requested_keys
        )
        raise NetWaggleError(
            "runtime profile must configure every existing topology link; "
            f"missing={missing}"
        )
    result: dict[str, Any] = {
        "configured_links": configured,
        "configured_link_count": len(configured),
        "qdisc_validated": False,
    }
    if qdisc_inspector is not None:
        qdisc_state = []
        for link in configured:
            for interface in link["interfaces"]:
                qdiscs = qdisc_inspector(interface)
                if not isinstance(qdiscs, list) or not qdiscs:
                    raise NetWaggleError(
                        f"qdisc inspector returned no state for {interface}"
                    )
                qdisc_state.append(
                    {
                        "interface": interface,
                        "qdiscs": qdiscs,
                    }
                )
        result["qdisc_state"] = qdisc_state
        result["qdisc_interface_count"] = len(qdisc_state)
        result["qdisc_validated"] = len(qdisc_state) == 2 * len(configured)
    return result


class DynamicProfileServer:
    """Root-side control socket with fixed condition-to-profile mappings."""

    def __init__(
        self,
        *,
        net,
        base_topology: NetWaggleTopology,
        condition_profiles: dict[str, Path],
        socket_path: str | Path,
        qdisc_inspector: QdiscInspector = inspect_qdisc,
    ) -> None:
        self.net = net
        self.base_topology = base_topology
        self.condition_profiles = {
            condition: path.resolve(strict=True)
            for condition, path in condition_profiles.items()
        }
        if not self.condition_profiles:
            raise ValueError("dynamic control requires condition profiles")
        self.socket_path = Path(socket_path)
        self.qdisc_inspector = qdisc_inspector
        if not self.socket_path.is_absolute():
            raise ValueError("NetWaggle control socket path must be absolute")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._server: socket.socket | None = None
        self._failed_links: set[frozenset[str]] = set()

    def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            if not self.socket_path.is_socket():
                raise NetWaggleError(
                    f"refusing to replace non-socket path: {self.socket_path}"
                )
            self.socket_path.unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        # When launched through sudo, retain the invoking user's primary group
        # as the narrow trust boundary for the typed control protocol.  Doing
        # this immediately after bind avoids a race with the wrapper script.
        sudo_gid = os.environ.get("SUDO_GID")
        if os.geteuid() == 0 and sudo_gid is not None:
            os.chown(self.socket_path, 0, int(sudo_gid))
        os.chmod(self.socket_path, 0o660)
        server.listen(4)
        server.settimeout(0.5)
        self._server = server
        self._thread = threading.Thread(
            target=self._serve,
            name="netwaggle-dynamic-control",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._server is not None:
            self._server.close()
        if self.socket_path.exists() and self.socket_path.is_socket():
            self.socket_path.unlink()

    def _serve(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                connection, _ = self._server.accept()
            # Python 3.8's socket timeout is not reliably caught by the
            # built-in TimeoutError on this Mininet host.
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            with connection:
                try:
                    response = self.handle_request(_read_request(connection))
                except Exception as exc:
                    response = {
                        "schema_version": "netwaggle.dynamic_control_result.v1",
                        "validated": False,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "measurements": {},
                    }
                try:
                    connection.sendall(
                        (json.dumps(response, sort_keys=True) + "\n").encode("utf-8")
                    )
                # Applying a full topology profile can outlive an impatient
                # client.  A disconnected client must not kill the sole
                # dynamic-control accept loop and invalidate later requests.
                except (BrokenPipeError, ConnectionResetError):
                    continue

    def handle_request(self, document: dict[str, Any]) -> dict[str, Any]:
        unknown = set(document) - ALLOWED_REQUEST_KEYS
        if unknown:
            raise NetWaggleError(
                f"unknown dynamic-control fields: {sorted(unknown)}"
            )
        if document.get("schema_version") != "netwaggle.dynamic_control.v1":
            raise NetWaggleError("unsupported dynamic-control schema")
        kind = document.get("kind")
        target = str(document.get("target") or "")
        action = str(document.get("action") or "")
        if kind == "LINK_STATE":
            return self._handle_link_state(
                target=target,
                action=action,
                condition_epoch=int(document.get("condition_epoch", 0)),
            )
        if kind != "NETWORK_PROFILE":
            raise NetWaggleError("unsupported dynamic-control kind")
        _validate_profile_target(target)
        if action not in {"APPLY", "RESTORE"}:
            raise NetWaggleError("invalid dynamic-control action")
        condition = str(document.get("condition") or "")
        try:
            profile_path = self.condition_profiles[condition]
        except KeyError as exc:
            raise NetWaggleError(
                f"condition is not mapped by the running NetWaggle process: {condition}"
            ) from exc
        profile = load_json(profile_path)
        if action == "APPLY" and target.startswith("sensor_uplink:"):
            profile = _retarget_sensor_profile(
                self.base_topology, profile, target.split(":", 1)[1]
            )
        elif action == "APPLY" and target == "site_to_cloud":
            profile = _retarget_wan_profile(self.base_topology, profile)
        elif action == "APPLY" and target == "site_backbone":
            profile = _retarget_site_backbone_profile(self.base_topology, profile)
        topology = self.base_topology.with_profile(profile)
        if action == "APPLY":
            _validate_profile_scope(self.base_topology, topology, target)
        measurements = apply_runtime_profile(
            self.net,
            topology,
            qdisc_inspector=self.qdisc_inspector,
        )
        if not measurements["qdisc_validated"]:
            raise NetWaggleError("runtime profile qdisc readback was not validated")
        return {
            "schema_version": "netwaggle.dynamic_control_result.v1",
            "validated": True,
            "condition": condition,
            "condition_epoch": int(document.get("condition_epoch", 0)),
            "profile": str(profile.get("name") or profile_path.stem),
            "measurements": measurements,
            "reason": "Mininet TCLink interfaces reconfigured and qdisc state read back",
        }

    def _handle_link_state(
        self,
        *,
        target: str,
        action: str,
        condition_epoch: int,
    ) -> dict[str, Any]:
        match = re.fullmatch(r"link:([A-Za-z0-9_.-]+):([A-Za-z0-9_.-]+)", target)
        if match is None:
            raise NetWaggleError("link target must be link:<switch>:<switch>")
        if action not in {"FAIL", "RESTORE"}:
            raise NetWaggleError("LINK_STATE accepts only FAIL or RESTORE")
        key = frozenset(match.groups())
        sensor_switch = next((item for item in key if item != "s_edge"), "")
        if "s_edge" not in key or re.fullmatch(
            r"s_(?:orin(?:[1-9]|[12][0-9]|30)|mob[1-6]|rpi|jetson)", sensor_switch
        ) is None:
            raise NetWaggleError(
                "LINK_STATE is restricted to one declared sensor-to-edge uplink"
            )
        live = next(
            (
                item
                for item in self.net.links
                if frozenset((item.intf1.node.name, item.intf2.node.name)) == key
            ),
            None,
        )
        if live is None:
            raise NetWaggleError("link target does not exist in the running topology")
        state = "down" if action == "FAIL" else "up"
        live.intf1.ifconfig(state)
        live.intf2.ifconfig(state)
        if action == "FAIL":
            self._failed_links.add(key)
        else:
            self._failed_links.discard(key)
        readback = [_interface_up(live.intf1), _interface_up(live.intf2)]
        expected = action == "RESTORE"
        if any(item is not expected for item in readback):
            raise NetWaggleError("link administrative-state readback mismatch")
        return {
            "schema_version": "netwaggle.dynamic_control_result.v1",
            "validated": True,
            "condition": "LINK_DOWN" if action == "FAIL" else "N0",
            "condition_epoch": condition_epoch,
            "profile": "link_state",
            "measurements": {
                "target": target,
                "interfaces": [live.intf1.name, live.intf2.name],
                "interface_up": readback,
                "failed_link_count": len(self._failed_links),
            },
            "reason": "Mininet link administrative state changed and read back",
        }


def _interface_up(interface) -> bool:
    if hasattr(interface, "isUp"):
        return bool(interface.isUp())
    return bool(getattr(interface, "up", False))


def _validate_profile_target(target: str) -> None:
    if target in {"site_to_cloud", "site_backbone"}:
        return
    if re.fullmatch(
        r"sensor_uplink:s_(?:orin(?:[1-9]|[12][0-9]|30)|mobile_archive_[1-6]|rpi|jetson)",
        target,
    ):
        return
    raise NetWaggleError("network-profile target is not allowlisted")


def _retarget_sensor_profile(
    base: NetWaggleTopology,
    profile: dict[str, Any],
    sensor: str,
) -> dict[str, Any]:
    """Move one declared sensor-uplink impairment to the requested sensor.

    L1 is a condition class, not a hard-coded device.  The condition-map file
    provides its impairment parameters; the typed target chooses the one link
    receiving them.  All other links remain at the running base profile.
    """

    sensor = _sensor_switch_name(sensor)
    base_by_key = {
        frozenset((item.src, item.dst)): item for item in base.links
    }
    requested_key = next(
        (
            key for key in base_by_key
            if sensor in key and "s_edge" in key
        ),
        None,
    )
    if requested_key is None:
        raise NetWaggleError(f"sensor uplink does not exist: {sensor}")
    candidates = []
    for row in profile.get("links", []):
        key = frozenset((str(row.get("from")), str(row.get("to"))))
        original = base_by_key.get(key)
        if original is None:
            continue
        proposed = type(original).from_dict(row)
        if _link_signature(proposed) != _link_signature(original) and "s_edge" in key:
            candidates.append(row)
    if len(candidates) != 1:
        raise NetWaggleError(
            "targeted sensor profile must define exactly one sensor-uplink impairment"
        )
    impairment = candidates[0]
    requested = base_by_key[requested_key]
    override = {
        "from": requested.src,
        "to": requested.dst,
        **{
            key: impairment[key]
            for key in ("bw", "delay", "jitter", "loss", "max_queue_size", "reverse")
            if key in impairment
        },
    }
    return {
        "name": f"{profile.get('name', 'profile')}@{sensor}",
        "links": _complete_profile_links(base, [override]),
    }


def _retarget_wan_profile(
    base: NetWaggleTopology, profile: dict[str, Any]
) -> dict[str, Any]:
    """Discard profile rows outside the site-to-cloud trust target."""

    rows = [
        row
        for row in profile.get("links", [])
        if "s_cloud" in {str(row.get("from")), str(row.get("to"))}
    ]
    if not rows:
        raise NetWaggleError("WAN profile has no site-to-cloud impairment")
    return {
        "name": profile.get("name", "profile"),
        "links": _complete_profile_links(base, rows),
    }


def _retarget_site_backbone_profile(
    base: NetWaggleTopology, profile: dict[str, Any]
) -> dict[str, Any]:
    """Apply only the declared edge-to-site impairment from an N2 profile."""

    rows = [
        row
        for row in profile.get("links", [])
        if {str(row.get("from")), str(row.get("to"))} == {"s_edge", "s_site"}
    ]
    if len(rows) != 1:
        raise NetWaggleError(
            "site-backbone profile must have exactly one s_edge-to-s_site impairment"
        )
    return {
        "name": profile.get("name", "profile"),
        "links": _complete_profile_links(base, rows),
    }


def _complete_profile_links(
    base: NetWaggleTopology, overrides: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key = {
        frozenset((str(row["from"]), str(row["to"]))): row
        for row in overrides
    }
    rows = []
    for link in base.links:
        key = frozenset((link.src, link.dst))
        if key in by_key:
            rows.append(by_key[key])
            continue
        row = {"from": link.src, "to": link.dst, **link.tc_params()}
        if link.reverse:
            row["reverse"] = link.reverse
        rows.append(row)
    return rows


def _link_signature(link) -> tuple[Any, ...]:
    return (
        link.bw,
        link.delay,
        link.jitter,
        link.loss,
        link.max_queue_size,
        json.dumps(link.reverse, sort_keys=True) if link.reverse else None,
    )


def _validate_profile_scope(
    base: NetWaggleTopology,
    profiled: NetWaggleTopology,
    target: str,
) -> None:
    base_links = {frozenset((item.src, item.dst)): item for item in base.links}
    changed = {
        key
        for item in profiled.links
        if (key := frozenset((item.src, item.dst))) in base_links
        and _link_signature(item) != _link_signature(base_links[key])
    }
    if not changed:
        raise NetWaggleError("APPLY profile does not change any link")
    if target == "site_to_cloud":
        allowed = {
            key for key in base_links if "s_cloud" in key
        }
    elif target == "site_backbone":
        allowed = {
            key for key in base_links if key == frozenset(("s_edge", "s_site"))
        }
    else:
        sensor = _sensor_switch_name(target.split(":", 1)[1])
        allowed = {key for key in base_links if sensor in key}
    if not changed.issubset(allowed):
        raise NetWaggleError(
            "profile changes links outside the requested target: "
            + ", ".join(sorted(":".join(sorted(item)) for item in changed - allowed))
        )


def _read_request(connection: socket.socket) -> dict[str, Any]:
    payload = bytearray()
    while b"\n" not in payload:
        chunk = connection.recv(4096)
        if not chunk:
            break
        payload.extend(chunk)
        if len(payload) > 64 * 1024:
            raise NetWaggleError("dynamic-control request exceeds 64 KiB")
    document = json.loads(bytes(payload).split(b"\n", 1)[0].decode("utf-8"))
    if not isinstance(document, dict):
        raise NetWaggleError("dynamic-control request must be an object")
    return document


def load_condition_map(path: str | Path) -> dict[str, Path]:
    source = Path(path).resolve(strict=True)
    document = load_json(source)
    allowed = {"schema_version", "conditions"}
    if set(document) - allowed:
        raise NetWaggleError("condition map contains unknown fields")
    if document.get("schema_version") != "netwaggle.condition_map.v1":
        raise NetWaggleError("unsupported condition-map schema")
    conditions = document.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        raise NetWaggleError("condition map requires conditions")
    root = source.parent
    result = {}
    for condition, relative in conditions.items():
        if not isinstance(condition, str) or not isinstance(relative, str):
            raise NetWaggleError("condition map keys and paths must be strings")
        candidate = (root / relative).resolve(strict=True)
        if root not in candidate.parents:
            raise NetWaggleError("condition profile escapes condition-map root")
        result[condition] = candidate
    return result


def _sensor_switch_name(sensor: str) -> str:
    """Translate stable public sensor targets to compact dataplane names."""

    match = re.fullmatch(r"s_mobile_archive_([1-6])", sensor)
    return f"s_mob{match.group(1)}" if match is not None else sensor

