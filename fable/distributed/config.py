"""YAML configuration loaders for deployment and node-local provider runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fable.planning.deployment import DeploymentGraph
from fable.common.schemas import ResourceReservation
from fable.common.time import EventTimeInterval
from fable.planning.models import ComputeCapacity, DeploymentNode, NetworkLink, SensorSource

from .models import ProviderRuntimeSpec, ResourceLimits, RuntimeMode


def _mqtt_topic_filter_matches(topic_filter: str, publish_topic: str) -> bool:
    """Match a concrete MQTT publish topic against a declared subscription."""

    if "+" in publish_topic or "#" in publish_topic:
        return topic_filter == publish_topic
    filter_levels = topic_filter.split("/")
    topic_levels = publish_topic.split("/")
    for index, level in enumerate(filter_levels):
        if level == "#":
            return index == len(filter_levels) - 1
        if index >= len(topic_levels):
            return False
        if level != "+" and level != topic_levels[index]:
            return False
    return len(filter_levels) == len(topic_levels)


class ProviderRuntimeResolver:
    def __init__(self, runtimes: dict[tuple[str, str], ProviderRuntimeSpec]) -> None:
        self._runtimes = dict(runtimes)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProviderRuntimeResolver":
        document = _load_yaml(path)
        runtimes: dict[tuple[str, str], ProviderRuntimeSpec] = {}
        nodes = document.get("nodes", {})
        for node_id, node_config in nodes.items():
            for provider_id, raw in node_config.get("providers", {}).items():
                payload = {
                    "provider_id": provider_id,
                    "provider_contract_version": raw.get("provider_contract_version", 1),
                    "node_id": node_id,
                    **raw,
                }
                runtimes[(node_id, provider_id)] = ProviderRuntimeSpec.model_validate(payload)
        return cls(runtimes)

    def resolve(
        self,
        *,
        node_id: str,
        provider_id: str,
        override: ProviderRuntimeSpec | None = None,
    ) -> ProviderRuntimeSpec:
        if override is not None:
            if override.node_id != node_id or override.provider_id != provider_id:
                raise ValueError("runtime override does not match requested node/provider")
            return override
        try:
            return self._runtimes[(node_id, provider_id)]
        except KeyError as exc:
            raise KeyError(f"no runtime configured for provider {provider_id} on node {node_id}") from exc

    def has(self, node_id: str, provider_id: str) -> bool:
        return (node_id, provider_id) in self._runtimes

    def worker_key(self, node_id: str, provider_id: str) -> str:
        """Return the physical worker identity for a logical provider capability."""
        runtime = self.resolve(node_id=node_id, provider_id=provider_id)
        return f"{node_id}/{runtime.worker_id or runtime.container_name or provider_id}"

    def supports_artifact_topic_transfer(
        self,
        *,
        source_node_id: str,
        source_provider_id: str,
        target_node_id: str,
        target_provider_id: str,
        data_type: str,
    ) -> bool:
        """Return whether a concrete MQTT artifact edge exists.

        Merely sharing a broker is not sufficient. Both runtime endpoints must
        independently declare the same typed artifact on the same exact topic.
        Node identity is not itself a transport boundary: generated evaluation
        nodes share the authenticated broker. Cross-node flow is executable
        only when both endpoints explicitly declare the same typed topic.
        """

        source = self.resolve(
            node_id=source_node_id, provider_id=source_provider_id
        )
        target = self.resolve(
            node_id=target_node_id, provider_id=target_provider_id
        )
        if source_node_id != target_node_id and not (
            source.artifact_broker_scope_id
            and source.artifact_broker_scope_id == target.artifact_broker_scope_id
        ):
            return False
        output_topic = source.artifact_topic_outputs.get(data_type)
        input_topic = target.artifact_topic_inputs.get(data_type)
        return bool(
            output_topic
            and input_topic
            and _mqtt_topic_filter_matches(input_topic, output_topic)
        )

    def capacity_group(
        self,
        node_id: str,
        provider_id: str,
        logical_reservation: ResourceReservation,
        fallback_owner_id: str,
    ) -> tuple[str, ResourceReservation]:
        """Map a logical provider reservation to a shared physical worker budget."""
        runtime = self.resolve(node_id=node_id, provider_id=provider_id)
        # A concrete container name is also a physical-worker identity.  Older
        # runtime manifests predate ``worker_id`` and commonly describe one
        # adopted worker (for example the identity service) through many
        # logical provider instances.  Charging each logical instance as a
        # separate process exhausts capacity even though they all attach to
        # the same container.
        worker_id = runtime.worker_id or runtime.container_name
        if not worker_id:
            # Preserve legacy one-instance/one-reservation semantics.
            return fallback_owner_id, logical_reservation
        limits = runtime.worker_resource_limits or ResourceLimits.from_reservation(logical_reservation)
        return (
            f"worker:{node_id}:{worker_id}",
            ResourceReservation(
                node_id=node_id,
                cpu_cores=limits.cpu_cores,
                memory_mb=limits.memory_mb,
                gpu_memory_mb=limits.gpu_memory_mb,
                network_bytes=0,
            ),
        )

    def allows_real_execution(self, node_id: str, provider_id: str) -> bool:
        return self.resolve(node_id=node_id, provider_id=provider_id).mode != RuntimeMode.REFERENCE

    @property
    def runtimes(self) -> tuple[ProviderRuntimeSpec, ...]:
        return tuple(self._runtimes[key] for key in sorted(self._runtimes))


def load_deployment_graph(path: str | Path) -> DeploymentGraph:
    document = _load_yaml(path)
    nodes = tuple(
        DeploymentNode(
            node_id=node_id,
            node_class=raw["node_class"],
            region=raw["region"],
            capacity=ComputeCapacity.model_validate(raw["capacity"]),
            resource_pool_id=raw.get("resource_pool_id"),
            capabilities=tuple(raw.get("capabilities", ())),
            policy_tags=tuple(raw.get("policy_tags", ())),
            available=bool(raw.get("available", True)),
        )
        for node_id, raw in sorted(document.get("nodes", {}).items())
    )
    sources = tuple(
        SensorSource(
            source_id=source_id,
            node_id=raw["node_id"],
            region=raw["region"],
            modalities=tuple(raw.get("modalities", ())),
            live_data_types=tuple(raw.get("live_data_types", ())),
            coverage_regions=tuple(raw.get("coverage_regions", ())),
            policy_tags=tuple(raw.get("policy_tags", ())),
            available=bool(raw.get("available", True)),
            raw_buffer_interval=(
                EventTimeInterval.model_validate(raw["raw_buffer_interval"])
                if raw.get("raw_buffer_interval") is not None
                else None
            ),
        )
        for source_id, raw in sorted(document.get("sources", {}).items())
    )
    links = tuple(
        NetworkLink.model_validate(item) for item in document.get("links", ())
    )
    resource_pools = {
        pool_id: ComputeCapacity.model_validate(raw["capacity"])
        for pool_id, raw in document.get("resource_pools", {}).items()
    }
    return DeploymentGraph(
        nodes=nodes, sources=sources, links=links,
        resource_pools=resource_pools or None,
    )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML document at {path} must be a mapping")
    return value
