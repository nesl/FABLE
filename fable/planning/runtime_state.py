"""Small runtime-condition model used by the physical planner."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from heapq import heappop, heappush
from itertools import count
from math import inf
from typing import Mapping


@dataclass(frozen=True, slots=True)
class NodeState:
    node_id: str
    node_type: str
    available: bool = True
    cpu_free: float = inf
    memory_mb_free: float = inf
    gpu_memory_mb_free: float = inf


@dataclass(frozen=True, slots=True)
class SourceState:
    source_id: str
    node_id: str
    data_type: str  # video_frame | audio_window | multichannel_audio | ...
    site_id: str | None = None
    sample_bytes: int = 0
    available: bool = True

    @property
    def site(self) -> str:
        return self.site_id or self.node_id


@dataclass(frozen=True, slots=True)
class LinkState:
    source_node: str
    destination_node: str
    latency_ms: float
    bandwidth_mbps: float | None = None
    available: bool = True
    measured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latency_source: str | None = None
    bandwidth_source: str | None = None


@dataclass(frozen=True, slots=True)
class RunningProvider:
    provider_id: str
    node_id: str
    source_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    provider_id: str
    node_type: str = "*"
    startup_ms: float = 0.0
    execution_ms: float = 0.0
    cpu: float = 0.0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    output_bytes: int = 0
    quality: float = 1.0


@dataclass(slots=True)
class RuntimeState:
    nodes: dict[str, NodeState]
    sources: dict[str, SourceState]
    links: tuple[LinkState, ...] = ()
    running: tuple[RunningProvider, ...] = ()
    profiles: dict[tuple[str, str], ProviderProfile] = field(default_factory=dict)
    object_sources: dict[str, str] = field(default_factory=dict)

    def profile_for(self, provider_id: str, node_type: str) -> ProviderProfile | None:
        return self.profiles.get((provider_id, node_type)) or self.profiles.get((provider_id, "*"))

    def running_key(self, provider_id: str, node_id: str, source_ids: tuple[str, ...]) -> bool:
        expected = (provider_id, node_id, tuple(sorted(source_ids)))
        return any(
            (row.provider_id, row.node_id, tuple(sorted(row.source_ids))) == expected
            for row in self.running
        )

    def sites(self) -> tuple[str, ...]:
        return tuple(sorted({source.site for source in self.sources.values() if source.available}))

    def update_link(self, link: LinkState) -> None:
        """Replace one directed runtime link measurement."""
        rows = [
            row for row in self.links
            if not (row.source_node == link.source_node and row.destination_node == link.destination_node)
        ]
        rows.append(link)
        self.links = tuple(sorted(rows, key=lambda row: (row.source_node, row.destination_node)))

    def update_node(self, node: NodeState) -> None:
        self.nodes[node.node_id] = node

    def replace_running_for_node(self, node_id: str, rows: tuple[RunningProvider, ...]) -> None:
        keep = [row for row in self.running if row.node_id != node_id]
        keep.extend(rows)
        self.running = tuple(sorted(keep, key=lambda row: (row.node_id, row.provider_id, row.source_ids)))

    def sources_of_type(self, data_type: str, *, site_id: str | None = None) -> tuple[SourceState, ...]:
        return tuple(
            sorted(
                (
                    source
                    for source in self.sources.values()
                    if source.available
                    and source.data_type == data_type
                    and (site_id is None or source.site == site_id)
                    and self.nodes.get(source.node_id, NodeState(source.node_id, "unknown", False)).available
                ),
                key=lambda row: row.source_id,
            )
        )

    def path_metrics(self, source_node: str, destination_node: str) -> tuple[float, float | None] | None:
        """Return minimum-latency path and its bottleneck measured throughput.

        ``bandwidth_mbps=None`` means throughput has not been measured.  Such a
        link can still carry small metadata, but the planner rejects large data
        transfers when it cannot estimate their completion time.
        """

        if source_node == destination_node:
            return 0.0, inf
        adjacency: dict[str, list[LinkState]] = {}
        for link in self.links:
            if not link.available:
                continue
            adjacency.setdefault(link.source_node, []).append(link)
        sequence = count()
        queue: list[tuple[float, int, str, float | None]] = [(0.0, next(sequence), source_node, inf)]
        best: dict[str, float] = {source_node: 0.0}
        while queue:
            latency, _, node, bottleneck = heappop(queue)
            if node == destination_node:
                return latency, bottleneck
            if latency > best.get(node, inf):
                continue
            for link in adjacency.get(node, ()):
                new_latency = latency + float(link.latency_ms)
                if new_latency >= best.get(link.destination_node, inf):
                    continue
                if link.bandwidth_mbps is None:
                    new_bandwidth = None
                elif bottleneck is None:
                    new_bandwidth = link.bandwidth_mbps
                else:
                    new_bandwidth = min(float(bottleneck), float(link.bandwidth_mbps))
                best[link.destination_node] = new_latency
                heappush(queue, (new_latency, next(sequence), link.destination_node, new_bandwidth))
        return None


def transfer_time_ms(size_bytes: int, path: tuple[float, float | None] | None) -> float | None:
    if path is None:
        return None
    latency_ms, bandwidth_mbps = path
    if size_bytes <= 0:
        return float(latency_ms)
    if bandwidth_mbps is None or bandwidth_mbps <= 0:
        return None
    if bandwidth_mbps == inf:
        return float(latency_ms)
    serialization_ms = (size_bytes * 8.0) / (bandwidth_mbps * 1_000_000.0) * 1000.0
    return float(latency_ms) + serialization_ms
