"""Normalize live distributed-runtime MQTT messages into common evaluation records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

from fable.distributed.models import (
    ArtifactAnnouncement,
    ProviderStatusEvent,
    ReliableNodeHeartbeat,
    ReliablePredicateResult,
)

from evaluation.runner import JsonlEventStore
from evaluation.schemas import (
    ArtifactEvent,
    BaselineId,
    PredicateObservation,
    ProviderLifecycleEvent,
    ResourceSample,
)
from evaluation.schemas.records import EvaluationRecord


@dataclass(frozen=True)
class RuntimeLoggingContext:
    run_id: str
    baseline_id: BaselineId
    trace_id: str
    default_request_id: str


class EvaluationMessageNormalizer:
    def __init__(self, context: RuntimeLoggingContext) -> None:
        self.context = context

    def normalize(self, topic: str, payload: bytes | str) -> EvaluationRecord | None:
        raw_text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        data = json.loads(raw_text)
        schema = data.get("schema_version")
        if schema == ReliablePredicateResult.SCHEMA_VERSION:
            envelope = ReliablePredicateResult.model_validate(data)
            result = envelope.result
            bindings = {**result.binding_delta.validated, **result.binding_delta.introduced}
            source_id = result.provenance.source_ids[0] if result.provenance.source_ids else None
            sequence = None
            if source_id is not None:
                sequence_range = result.provenance.source_sequence_ranges.get(source_id)
                sequence = sequence_range[1] if sequence_range else None
            return PredicateObservation(
                run_id=self.context.run_id,
                baseline_id=self.context.baseline_id,
                trace_id=self.context.trace_id,
                request_id=result.request_id,
                hypothesis_id=str(result.hypothesis_id),
                sensor_id=source_id,
                provider_id=result.provenance.provider_id,
                event_time=result.event_time_interval.start,
                monotonic_timestamp_ns=perf_counter_ns(),
                observation_id=str(result.result_id),
                predicate_id=result.semantic_predicate.predicate_id,
                event_end_time=result.event_time_interval.end,
                bindings=bindings,
                confidence=result.confidence,
                evidence_refs=tuple(str(item) for item in result.artifact_ids),
                source_sequence=sequence,
                metadata={
                    "topic": topic,
                    "truth": result.truth.value,
                    "node_id": result.provenance.node_id,
                    "processing_started_at": result.processing_started_at.isoformat(),
                    "processing_completed_at": result.processing_completed_at.isoformat(),
                },
            )
        if schema == ProviderStatusEvent.SCHEMA_VERSION:
            event = ProviderStatusEvent.model_validate(data)
            return ProviderLifecycleEvent(
                run_id=self.context.run_id,
                baseline_id=self.context.baseline_id,
                trace_id=self.context.trace_id,
                request_id=self.context.default_request_id,
                sensor_id=event.node_id,
                provider_id=event.provider_id,
                event_time=event.emitted_at,
                wall_timestamp=event.emitted_at,
                monotonic_timestamp_ns=perf_counter_ns(),
                provider_instance_id=event.provider_instance_id,
                lifecycle_event=event.status.value,
                demand_ids=tuple(str(item) for item in event.active_lease_ids),
                node_id=event.node_id,
                metadata={
                    "topic": topic,
                    "session_id": event.session_id,
                    "container_id": event.container_id,
                    "adopted": event.adopted,
                    "reason": event.reason,
                },
            )
        if schema == ArtifactAnnouncement.SCHEMA_VERSION:
            event = ArtifactAnnouncement.model_validate(data)
            artifact = event.artifact
            return ArtifactEvent(
                run_id=self.context.run_id,
                baseline_id=self.context.baseline_id,
                trace_id=self.context.trace_id,
                request_id=self.context.default_request_id,
                sensor_id=event.node_id,
                provider_id=artifact.producer.provider_id,
                event_time=artifact.event_time_interval.start,
                wall_timestamp=event.emitted_at,
                monotonic_timestamp_ns=perf_counter_ns(),
                artifact_id=str(artifact.artifact_id),
                artifact_type=artifact.artifact_type,
                action="CREATE",
                node_id=artifact.location.node_id or event.node_id,
                bytes=artifact.bytes or 0,
                access_mode=",".join(item.value for item in artifact.access_modes),
                expires_at=artifact.expires_at,
                bindings=artifact.bindings,
                metadata={
                    "topic": topic,
                    "durable_local_write": event.durable_local_write,
                    "location_kind": artifact.location.kind.value,
                },
            )
        if schema == ReliableNodeHeartbeat.SCHEMA_VERSION:
            envelope = ReliableNodeHeartbeat.model_validate(data)
            heartbeat = envelope.heartbeat
            return ResourceSample(
                run_id=self.context.run_id,
                baseline_id=self.context.baseline_id,
                trace_id=self.context.trace_id,
                request_id=self.context.default_request_id,
                sensor_id=heartbeat.node_id,
                event_time=heartbeat.sent_at,
                wall_timestamp=heartbeat.sent_at,
                monotonic_timestamp_ns=perf_counter_ns(),
                node_id=heartbeat.node_id,
                memory_bytes=heartbeat.capacity.memory_free_mb * 1024 * 1024,
                gpu_memory_bytes=heartbeat.capacity.gpu_free_mb * 1024 * 1024,
                metadata={
                    "topic": topic,
                    "measurement_kind": "free_capacity_heartbeat",
                    "cpu_free_cores": heartbeat.capacity.cpu_free_cores,
                    "network_tx_available_mbps": heartbeat.capacity.network_tx_available_mbps,
                    "network_rx_available_mbps": heartbeat.capacity.network_rx_available_mbps,
                    "availability": heartbeat.availability.value,
                    "active_provider_instances": list(heartbeat.active_provider_instance_ids),
                    "active_demands": [str(item) for item in heartbeat.active_demand_ids],
                },
            )
        return None


class MqttEvaluationLogger:
    """Small Paho subscriber for the existing replay/MQTT deployment."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        client_id: str,
        store: JsonlEventStore,
        normalizer: EvaluationMessageNormalizer,
    ) -> None:
        import paho.mqtt.client as mqtt

        self.store = store
        self.normalizer = normalizer
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            clean_session=False,
        )
        self.host = host
        self.port = port
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        for topic, qos in (
            ("fable/v1/result/+/+", 1),
            ("fable/v1/status/+/provider", 1),
            ("fable/v1/artifact/+/announce", 1),
            ("fable/v1/status/+/heartbeat", 0),
        ):
            client.subscribe(topic, qos=qos)

    def _on_message(self, client, userdata, message) -> None:
        try:
            record = self.normalizer.normalize(message.topic, message.payload)
            if record is not None:
                self.store.append(record)
        except Exception as exc:  # Preserve the live run even if one record is malformed.
            path = self.store.root / "normalization_errors.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "topic": message.topic,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    + "\n"
                )

    def run_forever(self) -> None:
        self.client.connect(self.host, self.port, keepalive=60)
        self.client.loop_forever()
