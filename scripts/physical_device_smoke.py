#!/usr/bin/env python3
"""Exercise the physical Pi and Jetson node agents over the real MQTT broker."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fable.common.time import EventTimeInterval, utc_now
from fable.distributed.config import ProviderRuntimeResolver, load_deployment_graph
from fable.distributed.demo import (
    build_replay_audio_candidate,
    build_replay_vehicle_candidate,
)
from fable.distributed.orchestrator import DistributedOrchestrator
from fable.distributed.outbox import SQLiteOutbox, SQLiteProcessedLedger
from fable.distributed.persistence import InMemoryStateStore
from fable.distributed.transport import PahoMQTTTransport, ReliableMessenger
from fable.planning.provider_registry import ProviderRegistry
from fable.scheduling.admission import MultiTenantScheduler
from fable.scheduling.capacity import CapacityLedger
from fable.scheduling.lifecycle import ProviderLifecycleManager


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="127.0.0.1")
    parser.add_argument("--port", default=1883, type=int)
    parser.add_argument("--timeout", default=15.0, type=float)
    args = parser.parse_args()

    registry = ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
    )
    lifecycle = ProviderLifecycleManager(
        provider_registry=registry,
        capacity=CapacityLedger(load_deployment_graph(ROOT / "config/physical_devices.yaml")),
    )
    transport = PahoMQTTTransport(
        host=args.broker,
        port=args.port,
        client_id=f"physical-smoke-{int(time.time())}",
        clean_session=True,
    )
    results = []
    heartbeats = {}

    with tempfile.TemporaryDirectory(prefix="fable-physical-smoke-") as state_dir:
        state = Path(state_dir)
        orchestrator = DistributedOrchestrator(
            orchestrator_id="physical-smoke-orchestrator",
            transport=transport,
            messenger=ReliableMessenger(
                entity_id="physical-smoke-orchestrator",
                transport=transport,
                outbox=SQLiteOutbox(state / "outbox.sqlite"),
            ),
            processed_ledger=SQLiteProcessedLedger(state / "processed.sqlite"),
            store=InMemoryStateStore(),
            scheduler=MultiTenantScheduler(lifecycle=lifecycle),
            lifecycle=lifecycle,
            runtime_resolver=ProviderRuntimeResolver.from_yaml(
                ROOT / "config/physical_provider_runtimes.yaml"
            ),
            on_result=results.append,
            on_heartbeat=lambda heartbeat: heartbeats.__setitem__(heartbeat.node_id, heartbeat),
            monitor_interval=1.0,
        )

        run_id = int(time.time() * 1000)
        jetson_request = f"physical_smoke_jetson_{run_id}"
        rpi_request = f"physical_smoke_rpi_{run_id}"
        request_nodes = {
            jetson_request: "physical_jetson",
            rpi_request: "physical_rpi",
        }
        orchestrator.start()
        try:
            if not transport.wait_connected(timeout=5.0):
                print("ERROR MQTT connection timed out")
                return 1

            heartbeat_deadline = time.monotonic() + min(args.timeout, 8.0)
            while set(heartbeats) != set(request_nodes.values()) and time.monotonic() < heartbeat_deadline:
                time.sleep(0.1)

            now = utc_now()
            interval = EventTimeInterval(
                start=now - timedelta(seconds=1),
                end=now + timedelta(seconds=1),
            )
            candidates = (
                build_replay_vehicle_candidate(
                    provider_registry=registry,
                    node_id="physical_jetson",
                    source_id="physical_jetson:camera",
                    event_interval=interval,
                    predicate_id="PASSES",
                    request_id=jetson_request,
                    deadline_seconds=60,
                    now=now,
                ),
                build_replay_audio_candidate(
                    provider_registry=registry,
                    node_id="physical_rpi",
                    source_id="physical_rpi:audio",
                    event_interval=interval,
                    request_id=rpi_request,
                    deadline_seconds=60,
                    now=now,
                ),
            )
            # The REFERENCE runtime performs no model inference.  The catalog's
            # production audio profile reserves GPU memory, which the Pi does
            # not have, so this substrate-only smoke explicitly overcommits.
            batch, commands = orchestrator.submit_candidates(
                candidates, now=now, allow_capacity_overcommit=True
            )
            result_deadline = time.monotonic() + args.timeout
            while len({result.request_id for result in results} & set(request_nodes)) < 2:
                if time.monotonic() >= result_deadline:
                    break
                time.sleep(0.1)

            print(f"heartbeats={','.join(sorted(heartbeats)) or 'none'}")
            print(f"admitted_plans={len(batch.admitted_plan_ids)} commands={len(commands)}")
            for record in batch.records:
                print(
                    f"admission candidate={record.candidate_id} "
                    f"decision={record.decision.value} reason={record.reason}"
                )
            observed = set()
            for result in sorted(results, key=lambda item: item.request_id):
                if result.request_id not in request_nodes:
                    continue
                observed.add(result.request_id)
                print(
                    f"result request={result.request_id} "
                    f"predicate={result.semantic_predicate.predicate_id} "
                    f"truth={result.truth.value} node={result.provenance.node_id}"
                )

            return 0 if observed == set(request_nodes) else 1
        finally:
            for request_id, node_id in request_nodes.items():
                try:
                    orchestrator.sweep_request(
                        (node_id,), request_id=request_id, reason="physical smoke test complete"
                    )
                except Exception as exc:
                    print(f"WARNING cleanup failed request={request_id}: {exc}")
            time.sleep(1.0)
            orchestrator.stop()


if __name__ == "__main__":
    raise SystemExit(main())
