"""Live deployment overlay used by the deployed planner.

The authored deployment remains the stable topology/capability description. This
module overlays node heartbeats plus typed node/link updates to produce an
immutable planning snapshot for each replan.  Every material runtime change
advances ``resource_epoch`` so live planning policies can distinguish semantic
progress from operating-condition changes.
"""

from __future__ import annotations

import threading
from typing import Iterable

from fable.common.enums import NodeAvailability
from fable.common.schemas import (
    NodeHeartbeat,
    RuntimeLinkUpdate,
    RuntimeNodeUpdate,
)

from .deployment import DeploymentGraph
from .models import ComputeCapacity, DeploymentNode, NetworkLink


class RuntimeDeploymentView:
    """Combines static deployment metadata with the latest runtime observations."""

    def __init__(self, base: DeploymentGraph) -> None:
        self.base = base
        self._heartbeats: dict[str, NodeHeartbeat] = {}
        self._node_availability: dict[str, NodeAvailability] = {}
        self._node_capacity: dict[str, ComputeCapacity] = {}
        self._links: dict[tuple[str, str], RuntimeLinkUpdate] = {}
        self._resource_epoch = 0
        self._lock = threading.RLock()

    @property
    def resource_epoch(self) -> int:
        with self._lock:
            return self._resource_epoch

    def record_heartbeat(self, heartbeat: NodeHeartbeat) -> bool:
        self.base.node(heartbeat.node_id)
        with self._lock:
            prior = self._heartbeats.get(heartbeat.node_id)
            if prior is not None and (heartbeat.sent_at, heartbeat.sequence) < (
                prior.sent_at,
                prior.sequence,
            ):
                return False
            changed = prior != heartbeat
            self._heartbeats[heartbeat.node_id] = heartbeat
            self._node_availability[heartbeat.node_id] = heartbeat.availability
            if changed:
                self._resource_epoch += 1
            return changed

    def set_node_availability(
        self,
        node_id: str,
        availability: NodeAvailability | bool,
    ) -> bool:
        self.base.node(node_id)
        resolved = (
            availability
            if isinstance(availability, NodeAvailability)
            else NodeAvailability.AVAILABLE if availability else NodeAvailability.UNAVAILABLE
        )
        with self._lock:
            previous = self._node_availability.get(node_id)
            changed = previous != resolved
            self._node_availability[node_id] = resolved
            heartbeat = self._heartbeats.get(node_id)
            if heartbeat is not None:
                self._heartbeats[node_id] = heartbeat.model_copy(
                    update={"availability": resolved}
                )
            if changed:
                self._resource_epoch += 1
            return changed

    def apply_node_update(self, update: RuntimeNodeUpdate) -> bool:
        """Apply one explicit runtime override and advance the resource epoch if needed."""

        self.base.node(update.node_id)
        with self._lock:
            changed = self._apply_node_update_locked(update)
            if changed:
                self._resource_epoch += 1
            return changed

    def apply_node_updates(self, updates: Iterable[RuntimeNodeUpdate]) -> bool:
        with self._lock:
            changed = False
            for update in updates:
                self.base.node(update.node_id)
                changed = self._apply_node_update_locked(update) or changed
            if changed:
                self._resource_epoch += 1
            return changed

    def _apply_node_update_locked(self, update: RuntimeNodeUpdate) -> bool:
        changed = False
        if update.available is not None:
            availability = (
                NodeAvailability.AVAILABLE
                if update.available
                else NodeAvailability.UNAVAILABLE
            )
            if self._node_availability.get(update.node_id) != availability:
                self._node_availability[update.node_id] = availability
                changed = True

        if any(
            value is not None
            for value in (
                update.cpu_available_cores,
                update.memory_available_mb,
                update.gpu_memory_available_mb,
            )
        ):
            node = self.base.node(update.node_id)
            heartbeat = self._heartbeats.get(update.node_id)
            prior = self._node_capacity.get(update.node_id)
            if prior is None:
                if heartbeat is not None:
                    prior = ComputeCapacity(
                        cpu_cores=min(node.capacity.cpu_cores, heartbeat.capacity.cpu_free_cores),
                        memory_mb=min(node.capacity.memory_mb, heartbeat.capacity.memory_free_mb),
                        gpu_memory_mb=min(node.capacity.gpu_memory_mb, heartbeat.capacity.gpu_free_mb),
                    )
                else:
                    prior = node.capacity
            capacity = ComputeCapacity(
                cpu_cores=(
                    prior.cpu_cores
                    if update.cpu_available_cores is None
                    else min(node.capacity.cpu_cores, update.cpu_available_cores)
                ),
                memory_mb=(
                    prior.memory_mb
                    if update.memory_available_mb is None
                    else min(node.capacity.memory_mb, update.memory_available_mb)
                ),
                gpu_memory_mb=(
                    prior.gpu_memory_mb
                    if update.gpu_memory_available_mb is None
                    else min(node.capacity.gpu_memory_mb, update.gpu_memory_available_mb)
                ),
            )
            if self._node_capacity.get(update.node_id) != capacity:
                self._node_capacity[update.node_id] = capacity
                changed = True
        return changed

    def update_link(self, update: RuntimeLinkUpdate) -> bool:
        self.base.node(update.source_node_id)
        self.base.node(update.target_node_id)
        with self._lock:
            changed = self._update_link_locked(update)
            if changed:
                self._resource_epoch += 1
            return changed

    def update_links(self, updates: Iterable[RuntimeLinkUpdate]) -> bool:
        with self._lock:
            changed = False
            for update in updates:
                self.base.node(update.source_node_id)
                self.base.node(update.target_node_id)
                changed = self._update_link_locked(update) or changed
            if changed:
                self._resource_epoch += 1
            return changed

    def apply_updates(
        self,
        *,
        node_updates: Iterable[RuntimeNodeUpdate] = (),
        link_updates: Iterable[RuntimeLinkUpdate] = (),
    ) -> tuple[bool, int, int]:
        """Atomically apply one disturbance and advance the epoch at most once."""

        node_updates = tuple(node_updates)
        link_updates = tuple(link_updates)
        for update in node_updates:
            self.base.node(update.node_id)
        for update in link_updates:
            self.base.node(update.source_node_id)
            self.base.node(update.target_node_id)
        with self._lock:
            previous_epoch = self._resource_epoch
            changed = False
            for update in node_updates:
                changed = self._apply_node_update_locked(update) or changed
            for update in link_updates:
                changed = self._update_link_locked(update) or changed
            if changed:
                self._resource_epoch += 1
            return changed, previous_epoch, self._resource_epoch

    def _update_link_locked(self, update: RuntimeLinkUpdate) -> bool:
        key = tuple(sorted((update.source_node_id, update.target_node_id)))
        previous = self._links.get(key)
        changed = previous != update
        self._links[key] = update
        return changed

    def snapshot(self) -> DeploymentGraph:
        with self._lock:
            heartbeats = dict(self._heartbeats)
            node_availability = dict(self._node_availability)
            capacity_overrides = dict(self._node_capacity)
            overrides = dict(self._links)

        nodes: list[DeploymentNode] = []
        for node in self.base.nodes.values():
            heartbeat = heartbeats.get(node.node_id)
            availability = node_availability.get(node.node_id)
            capacity = node.capacity
            # A heartbeat reports an instantaneous free-capacity sample, not a
            # replacement for the node's schedulable capacity contract.  Phase
            # 3 checks whether a realization can fit on the machine at all;
            # Phase 5's CapacityLedger is the authority for incremental logical
            # reservations and observed hard memory/GPU limits.  Substituting a
            # busy sample here makes an already-running shared worker erase its
            # own otherwise feasible continuation from the alternative graph.
            # Explicit disturbance overrides below remain authoritative.
            if node.node_id in capacity_overrides:
                capacity = capacity_overrides[node.node_id]
            nodes.append(
                node.model_copy(
                    update={
                        "capacity": capacity,
                        "available": (
                            node.available
                            and availability not in (
                                NodeAvailability.SUSPECT,
                                NodeAvailability.UNAVAILABLE,
                            )
                        ),
                    }
                )
            )

        links: list[NetworkLink] = []
        consumed: set[tuple[str, str]] = set()
        for link in self.base.links:
            key = tuple(sorted((link.source_node_id, link.target_node_id)))
            override = overrides.get(key)
            if override is None:
                links.append(link.model_copy(deep=True))
                continue
            consumed.add(key)
            links.append(
                link.model_copy(
                    update={
                        "latency_ms": (
                            link.latency_ms
                            if override.latency_ms is None
                            else int(round(override.latency_ms))
                        ),
                        "bandwidth_mbps": (
                            link.bandwidth_mbps
                            if override.bandwidth_mbps is None
                            else override.bandwidth_mbps
                        ),
                        "available": link.available and override.available,
                    }
                )
            )
        for key, override in overrides.items():
            if key in consumed:
                continue
            # A runtime-only link must specify both physical quantities.
            if override.latency_ms is None or override.bandwidth_mbps is None:
                continue
            links.append(
                NetworkLink(
                    source_node_id=override.source_node_id,
                    target_node_id=override.target_node_id,
                    latency_ms=int(round(override.latency_ms)),
                    bandwidth_mbps=override.bandwidth_mbps,
                    available=override.available,
                    bidirectional=True,
                )
            )

        sources = tuple(source.model_copy(deep=True) for source in self.base.sources.values())
        return DeploymentGraph(nodes=tuple(nodes), sources=sources, links=tuple(links))
