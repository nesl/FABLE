"""Load the documented deployment YAML into the refactored RuntimeState."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fable.planning import (
    LinkState,
    NodeState,
    RuntimeState,
    SourceState,
    load_provider_profiles,
)


def load_runtime_state(path: str | Path) -> RuntimeState:
    source = Path(path)
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if int(raw.get("version", 0)) != 1:
        raise ValueError("deployment version must be 1")
    nodes = {
        str(node_id): NodeState(
            str(node_id),
            str(spec.get("node_type", "unknown")),
            bool(spec.get("available", True)),
            float(spec.get("cpu_free", float("inf"))),
            float(spec.get("memory_mb_free", float("inf"))),
            float(spec.get("gpu_memory_mb_free", float("inf"))),
        )
        for node_id, spec in raw.get("nodes", {}).items()
    }
    sources = {
        str(source_id): SourceState(
            str(source_id),
            str(spec["node_id"]),
            str(spec["data_type"]),
            None if spec.get("site_id") is None else str(spec["site_id"]),
            int(spec.get("sample_bytes", 0)),
            bool(spec.get("available", True)),
        )
        for source_id, spec in raw.get("sources", {}).items()
    }
    links = tuple(
        LinkState(
            str(spec["source_node"]),
            str(spec["destination_node"]),
            float(spec.get("latency_ms", 0)),
            None
            if spec.get("bandwidth_mbps") is None
            else float(spec["bandwidth_mbps"]),
            bool(spec.get("available", True)),
        )
        for spec in raw.get("links", ())
    )
    unknown_owners = sorted(
        source.node_id for source in sources.values() if source.node_id not in nodes
    )
    if unknown_owners:
        raise ValueError(f"sources reference unknown nodes: {unknown_owners}")
    return RuntimeState(
        nodes=nodes,
        sources=sources,
        links=links,
        profiles=load_provider_profiles(),
    )
