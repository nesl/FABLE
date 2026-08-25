"""Deployment graph for sensing, compute, network, and locality decisions."""

from __future__ import annotations

import heapq
from collections.abc import Iterable

from fable.common.time import EventTimeInterval, RAW_BUFFER_ALIGNMENT_TOLERANCE

from .models import DeploymentNode, NetworkLink, NetworkPath, SensorSource


class DeploymentGraphError(ValueError):
    """Raised for invalid or disconnected deployment metadata."""


class DeploymentGraph:
    def __init__(
        self,
        *,
        nodes: Iterable[DeploymentNode],
        sources: Iterable[SensorSource] = (),
        links: Iterable[NetworkLink] = (),
    ) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self.sources = {source.source_id: source for source in sources}
        self.links = tuple(links)
        if not self.nodes:
            raise DeploymentGraphError("deployment graph requires at least one compute node")
        self._validate()

    def _validate(self) -> None:
        for source in self.sources.values():
            if source.node_id not in self.nodes:
                raise DeploymentGraphError(
                    f"source {source.source_id} references unknown node {source.node_id}"
                )
        for link in self.links:
            if link.source_node_id not in self.nodes or link.target_node_id not in self.nodes:
                raise DeploymentGraphError("network link references an unknown node")

    def node(self, node_id: str) -> DeploymentNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise DeploymentGraphError(f"unknown deployment node: {node_id}") from exc

    def source(self, source_id: str) -> SensorSource:
        try:
            return self.sources[source_id]
        except KeyError as exc:
            raise DeploymentGraphError(f"unknown sensor source: {source_id}") from exc

    def candidate_nodes(
        self,
        *,
        required_capabilities: Iterable[str] = (),
        allowed_node_ids: Iterable[str] = (),
        allowed_regions: Iterable[str] = (),
    ) -> tuple[DeploymentNode, ...]:
        required = set(required_capabilities)
        allowed_ids = set(allowed_node_ids)
        allowed_regions_set = set(allowed_regions)
        result = []
        for node in self.nodes.values():
            if not node.available:
                continue
            if allowed_ids and node.node_id not in allowed_ids:
                continue
            if allowed_regions_set and node.region not in allowed_regions_set:
                continue
            if not required.issubset(set(node.capabilities)):
                continue
            result.append(node)
        return tuple(sorted(result, key=lambda item: item.node_id))

    def candidate_sources(
        self,
        *,
        data_type: str,
        interval: EventTimeInterval,
        eligible_source_ids: Iterable[str] = (),
        eligible_regions: Iterable[str] = (),
        require_live: bool = False,
    ) -> tuple[SensorSource, ...]:
        allowed_sources = set(eligible_source_ids)
        allowed_regions = set(eligible_regions)
        result = []
        for source in self.sources.values():
            if not source.available:
                continue
            if allowed_sources and source.source_id not in allowed_sources:
                continue
            if allowed_regions and not (
                source.region in allowed_regions
                or bool(set(source.coverage_regions) & allowed_regions)
            ):
                continue
            if data_type not in source.live_data_types:
                continue
            if require_live:
                result.append(source)
                continue
            if source.raw_buffer_interval is None or source.raw_buffer_interval.contains_interval(
                interval,
                tolerance=RAW_BUFFER_ALIGNMENT_TOLERANCE,
            ):
                result.append(source)
        return tuple(sorted(result, key=lambda item: item.source_id))

    def shortest_path(self, source_node_id: str, target_node_id: str) -> NetworkPath | None:
        if source_node_id == target_node_id:
            raise DeploymentGraphError("shortest_path is only defined between distinct nodes")
        adjacency: dict[str, list[tuple[str, NetworkLink]]] = {node_id: [] for node_id in self.nodes}
        for link in self.links:
            if not link.available:
                continue
            adjacency[link.source_node_id].append((link.target_node_id, link))
            if link.bidirectional:
                adjacency[link.target_node_id].append((link.source_node_id, link))

        queue: list[tuple[int, str, tuple[str, ...], float]] = [
            (0, source_node_id, (source_node_id,), float("inf"))
        ]
        best: dict[str, int] = {}
        while queue:
            latency, current, path, bottleneck = heapq.heappop(queue)
            if current in best and best[current] <= latency:
                continue
            best[current] = latency
            if current == target_node_id:
                return NetworkPath(
                    node_ids=path,
                    latency_ms=latency,
                    bottleneck_bandwidth_mbps=bottleneck,
                )
            for neighbor, link in sorted(adjacency.get(current, ()), key=lambda item: item[0]):
                heapq.heappush(
                    queue,
                    (
                        latency + link.latency_ms,
                        neighbor,
                        (*path, neighbor),
                        min(bottleneck, link.bandwidth_mbps),
                    ),
                )
        return None

    def estimate_transfer_ms(self, path: NetworkPath, bytes_count: int) -> int:
        serialization_ms = 0
        if bytes_count > 0:
            serialization_ms = int(
                (bytes_count * 8 / (path.bottleneck_bandwidth_mbps * 1_000_000)) * 1000
            )
        return path.latency_ms + serialization_ms
