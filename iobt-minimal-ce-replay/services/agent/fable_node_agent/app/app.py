#!/usr/bin/env python3
"""FABLE node agent that coexists with iobt-minimal-ce-replay services."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import signal
import threading
from uuid import uuid4

from fable.integrations.reference_runtime import SyntheticReferenceRuntime
from fable.distributed.docker_runtime import DockerSDKRuntime, FakeContainerRuntime
from fable.distributed.heartbeat import CapacitySampler, ReplaySourceProgressTracker
from fable.distributed.node_agent import NodeAgent
from fable.distributed.outbox import SQLiteOutbox, SQLiteProcessedLedger
from fable.distributed.segment_store import SegmentStore
from fable.distributed.transport import PahoMQTTTransport, ReliableMessenger
from fable.integrations.replay import build_replay_output_adapter_registry


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("fable.node_agent.service")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    node_id = os.environ["FABLE_NODE_ID"]
    session_id = os.environ.get("FABLE_SESSION_ID", str(uuid4()))
    state_dir = Path(os.environ.get("FABLE_STATE_DIR", f"/var/lib/fable/{node_id}"))
    state_dir.mkdir(parents=True, exist_ok=True)

    transport = PahoMQTTTransport(
        host=os.environ.get("MQTT_HOST_IP", "mqtt"),
        port=int(os.environ.get("MQTT_PORT", "1883")),
        client_id=os.environ.get("FABLE_MQTT_CLIENT_ID", f"fable-agent-{node_id}"),
        keepalive=int(os.environ.get("MQTT_KEEPALIVE", "60")),
    )
    messenger = ReliableMessenger(
        entity_id=node_id,
        transport=transport,
        outbox=SQLiteOutbox(state_dir / "mqtt-outbox.sqlite"),
        retry_interval=float(os.environ.get("FABLE_OUTBOX_RETRY_SEC", "1.0")),
    )
    segment_store = SegmentStore(state_dir / "segments.sqlite")
    aliases_raw = os.environ.get("FABLE_SOURCE_ALIASES_JSON", "{}")
    aliases = json.loads(aliases_raw)
    if not isinstance(aliases, dict):
        raise ValueError("FABLE_SOURCE_ALIASES_JSON must encode a JSON object")
    progress = ReplaySourceProgressTracker(
        node_id=node_id,
        segment_store=segment_store,
        source_aliases={str(key): str(value) for key, value in aliases.items()},
    )
    runtime_kind = os.environ.get("FABLE_CONTAINER_RUNTIME", "docker").lower()
    containers = FakeContainerRuntime() if runtime_kind == "fake" else DockerSDKRuntime()

    agent = NodeAgent(
        node_id=node_id,
        session_id=session_id,
        transport=transport,
        messenger=messenger,
        processed_ledger=SQLiteProcessedLedger(state_dir / "processed.sqlite"),
        container_runtime=containers,
        progress=progress,
        state_dir=state_dir,
        heartbeat_interval=float(os.environ.get("FABLE_HEARTBEAT_INTERVAL_SEC", "1.0")),
        capacity_sampler=CapacitySampler(),
        allow_fault_injection=env_bool("FABLE_ALLOW_FAULT_INJECTION", False),
        output_adapters=build_replay_output_adapter_registry(),
        reference_runtime=SyntheticReferenceRuntime(),
    )

    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_args: stop.set())

    agent.start()
    if not transport.wait_connected(
        timeout=float(os.environ.get("FABLE_MQTT_CONNECT_TIMEOUT_SEC", "15"))
    ):
        raise RuntimeError(f"node agent {node_id} could not connect to MQTT")
    LOGGER.info("FABLE node agent ready node=%s session=%s", node_id, session_id)
    try:
        stop.wait()
    finally:
        agent.stop()
        segment_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
