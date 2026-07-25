"""Transactional capacity accounting for provider-instance reservations."""

from __future__ import annotations

from collections.abc import Iterable

from fable.common.schemas import ResourceReservation
from fable.planning.deployment import DeploymentGraph


class CapacityError(ValueError):
    """Raised when a reservation is invalid or exceeds node capacity."""


class CapacityLedger:
    """Tracks incremental capacity by provider instance, not by semantic demand.

    A shared provider is reserved once even when several hypotheses attach
    leases.  This is the critical distinction between planning every demand
    independently and provider-token sharing.
    """

    def __init__(self, deployment: DeploymentGraph) -> None:
        self.deployment = deployment
        self._reservations: dict[str, ResourceReservation] = {}

    @property
    def reservations(self) -> tuple[tuple[str, ResourceReservation], ...]:
        return tuple(sorted(self._reservations.items(), key=lambda item: item[0]))

    def used(self, node_id: str) -> ResourceReservation:
        cpu = memory = gpu = network = 0
        cpu_f = 0.0
        for reservation in self._reservations.values():
            if reservation.node_id != node_id:
                continue
            cpu_f += reservation.cpu_cores
            memory += reservation.memory_mb
            gpu += reservation.gpu_memory_mb
            network += reservation.network_bytes
        return ResourceReservation(
            node_id=node_id,
            cpu_cores=cpu_f,
            memory_mb=memory,
            gpu_memory_mb=gpu,
            network_bytes=network,
        )

    def available(self, node_id: str) -> ResourceReservation:
        node = self.deployment.node(node_id)
        used = self.used(node_id)
        return ResourceReservation(
            node_id=node_id,
            cpu_cores=max(0.0, node.capacity.cpu_cores - used.cpu_cores),
            memory_mb=max(0, node.capacity.memory_mb - used.memory_mb),
            gpu_memory_mb=max(0, node.capacity.gpu_memory_mb - used.gpu_memory_mb),
            network_bytes=0,
        )

    def can_reserve(self, reservations: Iterable[tuple[str, ResourceReservation]]) -> tuple[bool, str]:
        additions: dict[str, ResourceReservation] = {}
        for owner_id, reservation in reservations:
            if owner_id in self._reservations:
                continue
            existing = additions.get(reservation.node_id)
            if existing is None:
                additions[reservation.node_id] = reservation
            else:
                additions[reservation.node_id] = ResourceReservation(
                    node_id=reservation.node_id,
                    cpu_cores=existing.cpu_cores + reservation.cpu_cores,
                    memory_mb=existing.memory_mb + reservation.memory_mb,
                    gpu_memory_mb=existing.gpu_memory_mb + reservation.gpu_memory_mb,
                    network_bytes=existing.network_bytes + reservation.network_bytes,
                )
        for node_id, addition in additions.items():
            available = self.available(node_id)
            if addition.cpu_cores > available.cpu_cores + 1e-9:
                return False, (
                    f"node {node_id} lacks CPU: needs {addition.cpu_cores:.3f}, "
                    f"available {available.cpu_cores:.3f}"
                )
            if addition.memory_mb > available.memory_mb:
                return False, (
                    f"node {node_id} lacks memory: needs {addition.memory_mb} MB, "
                    f"available {available.memory_mb} MB"
                )
            if addition.gpu_memory_mb > available.gpu_memory_mb:
                return False, (
                    f"node {node_id} lacks GPU memory: needs {addition.gpu_memory_mb} MB, "
                    f"available {available.gpu_memory_mb} MB"
                )
        return True, ""

    def reserve(self, owner_id: str, reservation: ResourceReservation) -> None:
        if owner_id in self._reservations:
            if self._reservations[owner_id] != reservation:
                raise CapacityError(f"reservation owner {owner_id} already has different capacity")
            return
        feasible, reason = self.can_reserve(((owner_id, reservation),))
        if not feasible:
            raise CapacityError(reason)
        self._reservations[owner_id] = reservation

    def release(self, owner_id: str) -> ResourceReservation | None:
        return self._reservations.pop(owner_id, None)
