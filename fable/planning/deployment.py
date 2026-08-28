"""Deployment graph for sensing, compute, network, and locality decisions."""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from collections.abc import Mapping

from fable.common.time import EventTimeInterval, RAW_BUFFER_ALIGNMENT_TOLERANCE

from .models import ComputeCapacity, DeploymentNode, NetworkLink, NetworkPath, SensorSource


class DeploymentGraphError(ValueError):
    """Raised for invalid or disconnected deployment metadata."""


class DeploymentGraph:
    def __init__(
        self,
        *,
        nodes: Iterable[DeploymentNode],
        sources: Iterable[SensorSource] = (),
        links: Iterable[NetworkLink] = (),
        resource_pools: Mapping[str, ComputeCapacity] | None = None,
    ) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self.sources = {source.source_id: source for source in sources}
        self.links = tuple(links)
        supplied_pools = dict(resource_pools or {})
        self.resource_pools: dict[str, ComputeCapacity] = supplied_pools or {
            node.resource_pool_id or node.node_id: node.capacity
            for node in self.nodes.values()
        }
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
        for node in self.nodes.values():
            pool_id = node.resource_pool_id or node.node_id
            if pool_id not in self.resource_pools:
                raise DeploymentGraphError(
                    f"node {node.node_id} references unknown resource pool {pool_id}"
                )

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

    def resource_pool(self, node_id: str) -> tuple[str, ComputeCapacity]:
        """Return the physical pool charged by a logical deployment node."""
        node = self.node(node_id)
        pool_id = node.resource_pool_id or node.node_id
        return pool_id, self.resource_pools[pool_id]

    def with_resource_reservations(
        self, reservations: Mapping[str, ComputeCapacity]
    ) -> "DeploymentGraph":
        """Return a copy with reservations subtracted from physical pools."""
        unknown = set(reservations) - set(self.resource_pools)
        if unknown:
            raise DeploymentGraphError(f"unknown resource pools: {sorted(unknown)}")
        pools = dict(self.resource_pools)
        for pool_id, used in reservations.items():
            capacity = pools[pool_id]
            pools[pool_id] = ComputeCapacity(
                cpu_cores=max(0.0, capacity.cpu_cores - used.cpu_cores),
                memory_mb=max(0, capacity.memory_mb - used.memory_mb),
                gpu_memory_mb=max(0, capacity.gpu_memory_mb - used.gpu_memory_mb),
            )
        nodes = tuple(
            node.model_copy(
                update={"capacity": pools[node.resource_pool_id or node.node_id]}
            )
            for node in self.nodes.values()
        )
        return DeploymentGraph(
            nodes=nodes, sources=self.sources.values(), links=self.links,
            resource_pools=pools,
        )

    def with_node_availability(
        self, node_id: str, *, available: bool
    ) -> "DeploymentGraph":
        """Return a copy with one node and its local sources enabled/disabled."""
        self.node(node_id)
        nodes = tuple(
            node.model_copy(update={"available": available})
            if node.node_id == node_id else node
            for node in self.nodes.values()
        )
        sources = tuple(
            source.model_copy(update={"available": available})
            if source.node_id == node_id else source
            for source in self.sources.values()
        )
        return DeploymentGraph(
            nodes=nodes, sources=sources, links=self.links,
            resource_pools=self.resource_pools,
        )

    def with_degraded_node_links(
        self, node_id: str, *, bandwidth_mbps: float,
        added_latency_ms: int = 0, policy_tag: str | None = None,
    ) -> "DeploymentGraph":
        """Return a copy with bounded degradation on every incident link."""
        self.node(node_id)
        links = tuple(
            link.model_copy(update={
                "bandwidth_mbps": min(link.bandwidth_mbps, bandwidth_mbps),
                "latency_ms": link.latency_ms + added_latency_ms,
                "policy_tags": tuple(dict.fromkeys((*link.policy_tags, policy_tag)))
                if policy_tag else link.policy_tags,
            }) if node_id in {link.source_node_id, link.target_node_id} else link
            for link in self.links
        )
        return DeploymentGraph(
            nodes=self.nodes.values(), sources=self.sources.values(), links=links,
            resource_pools=self.resource_pools,
        )

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
