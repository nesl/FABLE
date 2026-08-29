"""Minimal execution agent for one compute node."""
from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
from typing import Callable

from fable.planning.runtime_state import NodeState, RunningProvider

from .plan_reconciler import ProviderInstanceKey, ProviderInstanceSpec
from .provider_runtime import ProviderRuntime


@dataclass(frozen=True, slots=True)
class NodeStatus:
    node_id: str
    node_type: str
    available: bool
    cpu_free: float
    memory_mb_free: float
    gpu_memory_mb_free: float
    running: tuple[RunningProvider, ...] = ()

    def as_node_state(self) -> NodeState:
        return NodeState(
            self.node_id,
            self.node_type,
            self.available,
            self.cpu_free,
            self.memory_mb_free,
            self.gpu_memory_mb_free,
        )


@dataclass(frozen=True, slots=True)
class CommandResult:
    ok: bool
    message: str = ""


ResourceProbe = Callable[[], NodeState]


class NodeAgent:
    def __init__(
        self,
        node: NodeState,
        provider_runtime: ProviderRuntime,
        *,
        resource_probe: ResourceProbe | None = None,
    ) -> None:
        self.node = node
        self.provider_runtime = provider_runtime
        self.resource_probe = resource_probe

    def start(self, spec: ProviderInstanceSpec) -> CommandResult:
        if spec.key.node_id != self.node.node_id:
            return CommandResult(False, f"provider is placed on {spec.key.node_id!r}, not this node")
        try:
            self.provider_runtime.start(spec)
            ready = getattr(self.provider_runtime, "ready", None)
            if callable(ready) and not ready(spec.key):
                return CommandResult(False, "provider started but did not become ready")
        except Exception as exc:  # keep transport boundary small and explicit
            return CommandResult(False, str(exc))
        return CommandResult(True, "ready")

    def stop(self, key: ProviderInstanceKey) -> CommandResult:
        if key.node_id != self.node.node_id:
            return CommandResult(False, f"provider is placed on {key.node_id!r}, not this node")
        try:
            self.provider_runtime.stop(key)
        except Exception as exc:
            return CommandResult(False, str(exc))
        return CommandResult(True, "stopped")

    def status(self) -> NodeStatus:
        current = self.resource_probe() if self.resource_probe is not None else self.node
        running = tuple(
            RunningProvider(key.provider_id, key.node_id, key.source_ids)
            for key in self.provider_runtime.running()
        )
        return NodeStatus(
            current.node_id,
            current.node_type,
            current.available,
            current.cpu_free,
            current.memory_mb_free,
            current.gpu_memory_mb_free,
            running,
        )


class SystemResourceProbe:
    """Small Linux-oriented resource probe for real node-agent status.

    ``cpu_free`` is an approximate number of currently free CPU cores based on
    the 1-minute load average. ``memory_mb_free`` comes from MemAvailable. GPU
    memory is queried from nvidia-smi when available; nodes without NVIDIA GPUs
    report zero free GPU memory.
    """

    def __init__(self, node_id: str, node_type: str) -> None:
        self.node_id = node_id
        self.node_type = node_type

    def __call__(self) -> NodeState:
        cpu_count = float(os.cpu_count() or 1)
        try:
            load = float(os.getloadavg()[0])
            cpu_free = max(0.0, cpu_count - load)
        except (AttributeError, OSError):
            cpu_free = cpu_count

        memory_mb_free = float("inf")
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        memory_mb_free = float(line.split()[1]) / 1024.0
                        break
        except OSError:
            pass

        gpu_memory_mb_free = 0.0
        try:
            completed = subprocess.run(
                ("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"),
                check=False, capture_output=True, text=True, timeout=2,
            )
            if completed.returncode == 0:
                values = [float(row.strip()) for row in completed.stdout.splitlines() if row.strip()]
                if values:
                    gpu_memory_mb_free = sum(values)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass

        return NodeState(
            self.node_id, self.node_type, True, cpu_free, memory_mb_free, gpu_memory_mb_free
        )
