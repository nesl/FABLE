"""NetWaggle testbed adapter for FABLE's generic runtime link telemetry."""

from __future__ import annotations

import json
import logging

from fable.distributed.transport import Transport
from fable.orchestration.telemetry import RuntimeLinkCallback
from fable.common.schemas import RuntimeLinkUpdate

LOGGER = logging.getLogger(__name__)


class NetWaggleTelemetrySource:
    """Translate retained NetWaggle profile documents into runtime link updates."""

    def __init__(self, *, topic: str = "/netwaggle/profile") -> None:
        self.topic = topic

    def bind(self, transport: Transport, callback: RuntimeLinkCallback) -> None:
        def on_profile(_topic: str, payload: bytes) -> None:
            try:
                document = json.loads(payload.decode("utf-8"))
                updates = self.parse(document)
                if updates:
                    callback(updates, "NetWaggle network profile changed")
            except Exception:
                LOGGER.exception("invalid NetWaggle profile payload")

        transport.subscribe(self.topic, on_profile, qos=0)

    @staticmethod
    def parse(document: object) -> tuple[RuntimeLinkUpdate, ...]:
        if not isinstance(document, dict):
            return ()
        gateway = document.get("fable_gateway_node_id")
        if not gateway:
            return ()
        raw_nodes = document.get("nodes", ())
        rows = raw_nodes.values() if isinstance(raw_nodes, dict) else raw_nodes
        updates: list[RuntimeLinkUpdate] = []
        for node in rows:
            if not isinstance(node, dict):
                continue
            source = node.get("fable_node_id")
            if not source or source == gateway:
                continue
            latency_ms = node.get("configured_one_way_ms")
            bandwidth = node.get("bottleneck_bw_mbps")
            if latency_ms is None and bandwidth is None:
                continue
            updates.append(
                RuntimeLinkUpdate(
                    source_node_id=str(source),
                    target_node_id=str(gateway),
                    latency_ms=None if latency_ms is None else float(latency_ms),
                    bandwidth_mbps=None if bandwidth is None else float(bandwidth),
                )
            )
        return tuple(updates)
