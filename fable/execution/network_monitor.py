"""Low-overhead online network measurements for RuntimeState.

Normal runtime operation should use frequent/lightweight ping probes plus passive
throughput observations from FABLE's own transfers.  iperf3 is supported as an
explicit/on-demand refresh because it actively consumes the capacity it measures.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence

from fable.planning.runtime_state import LinkState

_PING_RTT = re.compile(
    r"(?:rtt|round-trip) min/avg/max(?:/(?:mdev|stddev))? = "
    r"[0-9.]+/(?P<average>[0-9.]+)/[0-9.]+(?:/[0-9.]+)? ms"
)


@dataclass(frozen=True, slots=True)
class NodeEndpoint:
    node_id: str
    host: str
    ssh_target: str | None = None
    iperf_port: int = 5201


CommandRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


def _run(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(argv), shell=False, check=False, capture_output=True, text=True, timeout=timeout
    )


class NetworkMonitor:
    def __init__(
        self,
        *,
        ping_count: int = 3,
        iperf_seconds: int = 2,
        passive_alpha: float = 0.35,
        command_runner: CommandRunner = _run,
    ) -> None:
        if ping_count < 1:
            raise ValueError("ping_count must be positive")
        if iperf_seconds < 1:
            raise ValueError("iperf_seconds must be positive")
        if not 0 < passive_alpha <= 1:
            raise ValueError("passive_alpha must be in (0,1]")
        self.ping_count = ping_count
        self.iperf_seconds = iperf_seconds
        self.passive_alpha = passive_alpha
        self.command_runner = command_runner
        self._links: dict[tuple[str, str], LinkState] = {}

    def measure_latency(self, source: NodeEndpoint, destination: NodeEndpoint) -> LinkState:
        cmd = self._remote_command(
            source,
            ("ping", "-n", "-c", str(self.ping_count), "-W", "2", destination.host),
        )
        completed = self.command_runner(cmd, float(3 + self.ping_count * 2))
        key = (source.node_id, destination.node_id)
        previous = self._links.get(key)
        if completed.returncode != 0:
            link = LinkState(
                source.node_id,
                destination.node_id,
                latency_ms=float("inf"),
                bandwidth_mbps=None if previous is None else previous.bandwidth_mbps,
                available=False,
                measured_at=datetime.now(timezone.utc),
                latency_source="ping",
                bandwidth_source=None if previous is None else previous.bandwidth_source,
            )
            self._links[key] = link
            return link
        match = _PING_RTT.search(completed.stdout)
        if match is None:
            raise RuntimeError("ping output did not contain an RTT summary")
        link = LinkState(
            source.node_id,
            destination.node_id,
            latency_ms=float(match.group("average")) / 2.0,
            bandwidth_mbps=None if previous is None else previous.bandwidth_mbps,
            available=True,
            measured_at=datetime.now(timezone.utc),
            latency_source="ping",
            bandwidth_source=None if previous is None else previous.bandwidth_source,
        )
        self._links[key] = link
        return link

    def measure_bandwidth(self, source: NodeEndpoint, destination: NodeEndpoint) -> LinkState:
        """Run an explicit iperf3 throughput probe.

        This is intentionally a separate operation so callers do not accidentally
        run intrusive throughput tests at every network refresh.
        """
        cmd = self._remote_command(
            source,
            (
                "iperf3", "--client", destination.host, "--port", str(destination.iperf_port),
                "--json", "--time", str(self.iperf_seconds),
            ),
        )
        completed = self.command_runner(cmd, float(5 + self.iperf_seconds))
        key = (source.node_id, destination.node_id)
        previous = self._links.get(key)
        throughput: float | None = None
        if completed.returncode == 0:
            try:
                document = json.loads(completed.stdout)
                end = document["end"]
                summary = end.get("sum_received") or end.get("sum")
                throughput = float(summary["bits_per_second"]) / 1_000_000.0
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                throughput = None
        link = LinkState(
            source.node_id,
            destination.node_id,
            latency_ms=0.0 if previous is None else previous.latency_ms,
            bandwidth_mbps=throughput,
            available=(previous.available if previous is not None else completed.returncode == 0),
            measured_at=datetime.now(timezone.utc),
            latency_source=None if previous is None else previous.latency_source,
            bandwidth_source="iperf3" if throughput is not None else None,
        )
        self._links[key] = link
        return link

    def record_transfer(
        self,
        source_node: str,
        destination_node: str,
        *,
        size_bytes: int,
        duration_ms: float,
    ) -> LinkState:
        """Update achievable throughput from an application transfer.

        This adds no probe traffic and is therefore the preferred steady-state
        bandwidth signal once FABLE is actively moving data.
        """
        if size_bytes < 0 or duration_ms <= 0:
            raise ValueError("size_bytes must be >=0 and duration_ms must be >0")
        observed = (float(size_bytes) * 8.0) / (duration_ms / 1000.0) / 1_000_000.0
        key = (source_node, destination_node)
        previous = self._links.get(key)
        if previous is not None and previous.bandwidth_mbps is not None:
            observed = (
                self.passive_alpha * observed
                + (1.0 - self.passive_alpha) * previous.bandwidth_mbps
            )
        link = LinkState(
            source_node,
            destination_node,
            latency_ms=0.0 if previous is None else previous.latency_ms,
            bandwidth_mbps=observed,
            available=True if previous is None else previous.available,
            measured_at=datetime.now(timezone.utc),
            latency_source=None if previous is None else previous.latency_source,
            bandwidth_source="passive",
        )
        self._links[key] = link
        return link

    def measure(
        self,
        source: NodeEndpoint,
        destination: NodeEndpoint,
        *,
        measure_bandwidth: bool = False,
    ) -> LinkState:
        """Refresh latency and optionally run an explicit iperf3 probe."""
        link = self.measure_latency(source, destination)
        if measure_bandwidth and link.available:
            link = self.measure_bandwidth(source, destination)
        return link

    def link_state(self, source_node: str, destination_node: str) -> LinkState | None:
        return self._links.get((source_node, destination_node))

    @staticmethod
    def _remote_command(source: NodeEndpoint, argv: Sequence[str]) -> tuple[str, ...]:
        if source.ssh_target:
            return ("ssh", source.ssh_target, "--", *argv)
        return tuple(argv)
