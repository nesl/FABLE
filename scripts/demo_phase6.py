#!/usr/bin/env python3
"""Run the Phase-6 reference path over two logical nodes without Docker/MQTT."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fable.common.time import EventTimeInterval, utc_now
from fable.distributed.demo import build_replay_audio_candidate
from fable.distributed.config import ProviderRuntimeResolver
from fable.distributed.docker_runtime import FakeContainerRuntime
from fable.distributed.heartbeat import CapacitySampler, ReplaySourceProgressTracker
from fable.distributed.models import ProviderRuntimeSpec, RuntimeMode
from fable.distributed.node_agent import NodeAgent
from fable.distributed.orchestrator import DistributedOrchestrator
from fable.distributed.outbox import SQLiteOutbox, SQLiteProcessedLedger
from fable.distributed.persistence import InMemoryStateStore
from fable.distributed.transport import InMemoryBroker, InMemoryTransport, ReliableMessenger
from fable.planning.provider_registry import ProviderRegistry
from fable.planning.testing import fake_deployment
from fable.scheduling.admission import MultiTenantScheduler
from fable.scheduling.capacity import CapacityLedger
from fable.scheduling.lifecycle import ProviderLifecycleManager


def main() -> int:
    registry = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
    )
    lifecycle = ProviderLifecycleManager(
        provider_registry=registry,
        capacity=CapacityLedger(fake_deployment()),
    )
    scheduler = MultiTenantScheduler(lifecycle=lifecycle)
    broker = InMemoryBroker()
    state = Path(tempfile.mkdtemp(prefix="fable-phase6-demo-"))
    store = InMemoryStateStore()
    received = []

    runtimes = {
        (node, "audio_event_classifier"): ProviderRuntimeSpec(
            provider_id="audio_event_classifier",
            provider_contract_version=1,
            node_id=node,
            mode=RuntimeMode.REFERENCE,
            reference_delay_ms=20,
        )
        for node in ("sensor_a", "sensor_b")
    }
    orch_transport = InMemoryTransport(broker)
    orchestrator = DistributedOrchestrator(
        orchestrator_id="orchestrator",
        transport=orch_transport,
        messenger=ReliableMessenger(
            entity_id="orchestrator",
            transport=orch_transport,
            outbox=SQLiteOutbox(state / "orchestrator-outbox.sqlite"),
            retry_interval=100,
        ),
        processed_ledger=SQLiteProcessedLedger(state / "orchestrator-processed.sqlite"),
        store=store,
        scheduler=scheduler,
        lifecycle=lifecycle,
        runtime_resolver=ProviderRuntimeResolver(runtimes),
        on_result=received.append,
        monitor_interval=100,
    )
    agents = []
    for node in ("sensor_a", "sensor_b"):
        transport = InMemoryTransport(broker)
        agents.append(
            NodeAgent(
                node_id=node,
                session_id=f"demo-{node}",
                transport=transport,
                messenger=ReliableMessenger(
                    entity_id=node,
                    transport=transport,
                    outbox=SQLiteOutbox(state / f"{node}-outbox.sqlite"),
                    retry_interval=100,
                ),
                processed_ledger=SQLiteProcessedLedger(state / f"{node}-processed.sqlite"),
                container_runtime=FakeContainerRuntime(),
                progress=ReplaySourceProgressTracker(node_id=node),
                state_dir=state / node,
                heartbeat_interval=100,
                capacity_sampler=CapacitySampler(gpu_free_mb_override=8192),
            )
        )

    orchestrator.start()
    for agent in agents:
        agent.start()
    now = utc_now()
    interval = EventTimeInterval(
        start=now - timedelta(seconds=1), end=now + timedelta(seconds=1)
    )
    candidates = tuple(
        build_replay_audio_candidate(
            provider_registry=registry,
            node_id=node,
            source_id=f"{node}:audio",
            event_interval=interval,
            request_id=f"phase6_{node}",
            deadline_seconds=30,
            now=now,
        )
        for node in ("sensor_a", "sensor_b")
    )
    try:
        batch, commands = orchestrator.submit_candidates(candidates, now=now)
        deadline = time.monotonic() + 2
        while len(received) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        print(f"admitted_plans={len(batch.admitted_plan_ids)} commands={len(commands)}")
        for result in sorted(received, key=lambda item: item.request_id):
            print(
                f"result request={result.request_id} predicate={result.semantic_predicate.predicate_id} "
                f"truth={result.truth.value} node={result.provenance.node_id}"
            )
        print(
            f"durable_results={len(store.list_raw('results'))} "
            f"control_events={len(store.list_events())}"
        )
        return 0 if len(received) == 2 else 1
    finally:
        for agent in agents:
            agent.stop()
        orchestrator.stop()


if __name__ == "__main__":
    raise SystemExit(main())
