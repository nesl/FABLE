"""YAML configuration loaders for deployment and node-local provider runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fable.planning.deployment import DeploymentGraph
from fable.planning.models import ComputeCapacity, DeploymentNode, NetworkLink, SensorSource

from .models import ProviderRuntimeSpec


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
        )
        for source_id, raw in sorted(document.get("sources", {}).items())
    )
    links = tuple(
        NetworkLink.model_validate(item) for item in document.get("links", ())
    )
    return DeploymentGraph(nodes=nodes, sources=sources, links=links)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML document at {path} must be a mapping")
    return value
