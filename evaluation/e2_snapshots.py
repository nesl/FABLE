"""Immutable, replayable checkpoint snapshots for the redesigned E2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

from fable.common.schemas import ArtifactRef, PredicateDemand
from fable.planning.artifact_catalog import ArtifactCatalog
from fable.planning.models import ActiveProviderInstance, ComputeCapacity, PhysicalAlternativeGraph

from evaluation.baselines.models import BaselinePlanningCase


SCHEMA_VERSION = "fable.e2_checkpoint_snapshot.v1"


def export_checkpoint_snapshot(
    case: BaselinePlanningCase,
    path: str | Path,
    *,
    source_record_paths: Iterable[str] = (),
    reservations: Mapping[str, ComputeCapacity] | None = None,
    active_providers: Iterable[ActiveProviderInstance] = (),
    deployment_artifacts: Iterable[ArtifactRef] = (),
    capture_kind: str = "typed_runtime_export",
) -> dict[str, object]:
    """Persist the complete typed planning boundary, not a lossy log projection."""

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "trace_id": case.trace_id,
            "source_record_paths": sorted(set(source_record_paths)),
            "synthetic_expansion": False,
            "capture_kind": capture_kind,
        },
        "case": {
            "run_id": case.run_id,
            "trace_id": case.trace_id,
            "request_id": case.request_id,
            "event_family": case.event_family,
            "now": case.now.isoformat(),
            "placement_id": case.placement_id,
            "replay_supported_sensor_ids": list(case.replay_supported_sensor_ids),
            "graph_version": case.graph_version,
            "resource_epoch": case.resource_epoch,
            "semantic_epoch": case.semantic_epoch,
            "replan_trigger": case.replan_trigger,
            "frontier_demands": [
                item.model_dump(mode="json") for item in case.frontier_demands
            ],
            "all_task_demands": [
                item.model_dump(mode="json") for item in case.all_task_demands
            ],
            "frontier_graph": case.frontier_graph.model_dump(mode="json"),
            "whole_event_graph": case.whole_event_graph.model_dump(mode="json"),
        },
        "resource_reservations": {
            key: value.model_dump(mode="json")
            for key, value in sorted((reservations or {}).items())
        },
        "active_providers": [
            item.model_dump(mode="json") for item in active_providers
        ],
        "deployment_artifacts": [
            item.model_dump(mode="json") for item in deployment_artifacts
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_checkpoint_snapshot(path: str | Path) -> BaselinePlanningCase:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported E2 checkpoint snapshot schema")
    claimed = document.pop("sha256", None)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if claimed != actual:
        raise ValueError("E2 checkpoint snapshot checksum mismatch")
    raw = document["case"]
    return BaselinePlanningCase(
        run_id=raw["run_id"],
        trace_id=raw["trace_id"],
        request_id=raw["request_id"],
        event_family=raw["event_family"],
        frontier_demands=tuple(PredicateDemand.model_validate(x) for x in raw["frontier_demands"]),
        all_task_demands=tuple(PredicateDemand.model_validate(x) for x in raw["all_task_demands"]),
        frontier_graph=PhysicalAlternativeGraph.model_validate(raw["frontier_graph"]),
        whole_event_graph=PhysicalAlternativeGraph.model_validate(raw["whole_event_graph"]),
        now=datetime.fromisoformat(raw["now"]),
        placement_id=raw["placement_id"],
        replay_supported_sensor_ids=tuple(raw["replay_supported_sensor_ids"]),
        graph_version=raw["graph_version"],
        resource_epoch=raw["resource_epoch"],
        semantic_epoch=raw["semantic_epoch"],
        replan_trigger=raw["replan_trigger"],
    )


def load_checkpoint_snapshot_artifacts(path: str | Path) -> ArtifactCatalog:
    """Load the exact artifact identities referenced by a runtime frontier."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported E2 checkpoint snapshot schema")
    claimed = document.pop("sha256", None)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    if claimed != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        raise ValueError("E2 checkpoint snapshot checksum mismatch")
    return ArtifactCatalog(
        ArtifactRef.model_validate(item)
        for item in document.get("deployment_artifacts", ())
    )


def with_snapshot_identity(
    case: BaselinePlanningCase, *, run_id: str, request_id: str
) -> BaselinePlanningCase:
    """Change execution identity without recompiling immutable demand IDs."""

    return replace(case, run_id=run_id, request_id=request_id)
