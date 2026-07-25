#!/usr/bin/env python3
"""FABLE Phase-6 distributed orchestrator service for the replay stack."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import signal
import threading

from fable.distributed.config import ProviderRuntimeResolver, load_deployment_graph
from fable.distributed.orchestrator import DistributedOrchestrator
from fable.distributed.outbox import SQLiteOutbox, SQLiteProcessedLedger
from fable.distributed.persistence import MongoStateStore
from fable.distributed.transport import PahoMQTTTransport, ReliableMessenger
from fable.planning.provider_registry import ProviderRegistry
from fable.scheduling.admission import MultiTenantScheduler
from fable.scheduling.capacity import CapacityLedger
from fable.scheduling.lifecycle import ProviderLifecycleManager


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("fable.orchestrator.service")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    state_dir = Path(os.environ.get("FABLE_STATE_DIR", "/var/lib/fable/orchestrator"))
    state_dir.mkdir(parents=True, exist_ok=True)
    orchestrator_id = os.environ.get("FABLE_ORCHESTRATOR_ID", "orchestrator")

    registry = ProviderRegistry.from_files(
        catalog_path=os.environ.get(
            "FABLE_PROVIDER_CATALOG", "/workspace/FABLE/providers/registry/catalog.yaml"
        ),
        data_types_path=os.environ.get(
            "FABLE_DATA_TYPES", "/workspace/FABLE/providers/registry/data_types.yaml"
        ),
    )
    deployment = load_deployment_graph(
        os.environ.get(
            "FABLE_DEPLOYMENT_CONFIG", "/workspace/replay/config/fable_deployment.yaml"
        )
    )
    lifecycle = ProviderLifecycleManager(
        provider_registry=registry,
        capacity=CapacityLedger(deployment),
        idle_grace_ms=int(os.environ.get("FABLE_IDLE_GRACE_MS", "2000")),
    )
    scheduler = MultiTenantScheduler(lifecycle=lifecycle)
    resolver = ProviderRuntimeResolver.from_yaml(
        os.environ.get(
            "FABLE_RUNTIME_CONFIG",
            "/workspace/replay/config/fable_provider_runtimes.yaml",
        )
    )

    transport = PahoMQTTTransport(
        host=os.environ.get("MQTT_HOST_IP", "mqtt"),
        port=int(os.environ.get("MQTT_PORT", "1883")),
        client_id=os.environ.get(
            "FABLE_MQTT_CLIENT_ID", f"fable-{orchestrator_id}"
        ),
        keepalive=int(os.environ.get("MQTT_KEEPALIVE", "60")),
    )
    messenger = ReliableMessenger(
        entity_id=orchestrator_id,
        transport=transport,
        outbox=SQLiteOutbox(state_dir / "mqtt-outbox.sqlite"),
        retry_interval=float(os.environ.get("FABLE_OUTBOX_RETRY_SEC", "1.0")),
    )
    store = MongoStateStore(
        os.environ.get("MONGODB_URI", "mongodb://fable-mongo:27017"),
        database=os.environ.get("MONGODB_DATABASE", "fable"),
    )

    def on_result(result) -> None:
        LOGGER.info(
            "predicate result request=%s hypothesis=%s predicate=%s truth=%s node=%s",
            result.request_id,
            result.hypothesis_id,
            result.semantic_predicate.predicate_id,
            result.truth,
            result.provenance.node_id,
        )

    def on_replan(node_id, demand_ids, reason) -> None:
        LOGGER.warning(
            "replanning required node=%s demand_ids=%s reason=%s",
            node_id,
            [str(item) for item in demand_ids],
            reason,
        )

    orchestrator = DistributedOrchestrator(
        orchestrator_id=orchestrator_id,
        transport=transport,
        messenger=messenger,
        processed_ledger=SQLiteProcessedLedger(state_dir / "processed.sqlite"),
        store=store,
        scheduler=scheduler,
        lifecycle=lifecycle,
        runtime_resolver=resolver,
        on_result=on_result,
        on_replan_required=on_replan,
        monitor_interval=float(os.environ.get("FABLE_MONITOR_INTERVAL_SEC", "1.0")),
    )

    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_args: stop.set())

    orchestrator.start()
    if not transport.wait_connected(
        timeout=float(os.environ.get("FABLE_MQTT_CONNECT_TIMEOUT_SEC", "15"))
    ):
        raise RuntimeError("orchestrator could not connect to MQTT")
    report = orchestrator.reconcile(
        require_heartbeats=env_bool("FABLE_REQUIRE_HEARTBEAT_ON_RESTART", False)
    )
    LOGGER.info("restart reconciliation: %s", report.model_dump(mode="json"))
    LOGGER.info("FABLE orchestrator ready id=%s", orchestrator_id)
    try:
        stop.wait()
    finally:
        orchestrator.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
