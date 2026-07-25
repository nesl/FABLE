"""Reusable deterministic Phase-6 test stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from fable.distributed.config import ProviderRuntimeResolver
from fable.distributed.docker_runtime import FakeContainerRuntime
from fable.distributed.heartbeat import CapacitySampler, ReplaySourceProgressTracker
from fable.distributed.models import ProviderRuntimeSpec, RuntimeMode
from fable.distributed.node_agent import NodeAgent
from fable.distributed.orchestrator import DistributedOrchestrator
from fable.distributed.outbox import SQLiteOutbox, SQLiteProcessedLedger
from fable.distributed.persistence import InMemoryStateStore
from fable.distributed.transport import (
    InMemoryBroker,
    InMemoryTransport,
    ReliableMessenger,
)
from fable.planning.provider_registry import ProviderRegistry
from fable.planning.testing import fake_deployment
from fable.scheduling.admission import MultiTenantScheduler
from fable.scheduling.capacity import CapacityLedger
from fable.scheduling.lifecycle import ProviderLifecycleManager


ROOT = Path(__file__).resolve().parents[1]


def provider_registry() -> ProviderRegistry:
    return ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
    )


@dataclass
class Phase6Stack:
    root: Path
    registry: ProviderRegistry
    broker: InMemoryBroker
    store: InMemoryStateStore
    lifecycle: ProviderLifecycleManager
    scheduler: MultiTenantScheduler
    orchestrator: DistributedOrchestrator
    agents: dict[str, NodeAgent]
    received_results: list

    def stop(self) -> None:
        for agent in self.agents.values():
            agent.stop()
        self.orchestrator.stop()


def make_stack(
    tmp_path: Path,
    *,
    nodes: tuple[str, ...] = ("sensor_a",),
    runtimes: dict[tuple[str, str], ProviderRuntimeSpec] | None = None,
    heartbeat_interval: float = 100.0,
) -> Phase6Stack:
    registry = provider_registry()
    lifecycle = ProviderLifecycleManager(
        provider_registry=registry,
        capacity=CapacityLedger(fake_deployment()),
    )
    scheduler = MultiTenantScheduler(lifecycle=lifecycle)
    broker = InMemoryBroker()
    store = InMemoryStateStore()
    received: list = []

    if runtimes is None:
        runtimes = {
            (node_id, "audio_event_classifier"): ProviderRuntimeSpec(
                provider_id="audio_event_classifier",
                provider_contract_version=1,
                node_id=node_id,
                mode=RuntimeMode.REFERENCE,
                reference_delay_ms=5,
            )
            for node_id in nodes
        }

    orchestrator_transport = InMemoryTransport(broker)
    orchestrator = DistributedOrchestrator(
        orchestrator_id="orchestrator",
        transport=orchestrator_transport,
        messenger=ReliableMessenger(
            entity_id="orchestrator",
            transport=orchestrator_transport,
            outbox=SQLiteOutbox(tmp_path / "orchestrator-outbox.sqlite"),
            retry_interval=100,
        ),
        processed_ledger=SQLiteProcessedLedger(tmp_path / "orchestrator-processed.sqlite"),
        store=store,
        scheduler=scheduler,
        lifecycle=lifecycle,
        runtime_resolver=ProviderRuntimeResolver(runtimes),
        on_result=received.append,
        monitor_interval=100,
    )

    agents: dict[str, NodeAgent] = {}
    for node_id in nodes:
        transport = InMemoryTransport(broker)
        agent = NodeAgent(
            node_id=node_id,
            session_id=f"session-{node_id}",
            transport=transport,
            messenger=ReliableMessenger(
                entity_id=node_id,
                transport=transport,
                outbox=SQLiteOutbox(tmp_path / f"{node_id}-outbox.sqlite"),
                retry_interval=100,
            ),
            processed_ledger=SQLiteProcessedLedger(
                tmp_path / f"{node_id}-processed.sqlite"
            ),
            container_runtime=FakeContainerRuntime(),
            progress=ReplaySourceProgressTracker(node_id=node_id),
            state_dir=tmp_path / node_id,
            heartbeat_interval=heartbeat_interval,
            capacity_sampler=CapacitySampler(gpu_free_mb_override=8192),
            allow_fault_injection=True,
        )
        agents[node_id] = agent

    orchestrator.start()
    for agent in agents.values():
        agent.start()
    return Phase6Stack(
        root=tmp_path,
        registry=registry,
        broker=broker,
        store=store,
        lifecycle=lifecycle,
        scheduler=scheduler,
        orchestrator=orchestrator,
        agents=agents,
        received_results=received,
    )


def wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())
