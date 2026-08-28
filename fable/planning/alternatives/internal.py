"""Internal records and helpers shared by alternative-generation components."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from fable.common.enums import ArtifactAccessMode
from fable.common.ids import deterministic_id
from fable.planning.models import (
    AlternativeEdgeKind,
    AlternativeGraphEdge,
    DataTransfer,
    StepPlacement,
)


@dataclass(frozen=True)
class DataRef:
    source_ref: str
    data_type: str
    node_id: str
    bytes: int
    source_id: str | None = None
    artifact_id: UUID | None = None
    access_modes: tuple[ArtifactAccessMode, ...] = (
        ArtifactAccessMode.LOCAL,
        ArtifactAccessMode.TRANSFERRED,
        ArtifactAccessMode.REMOTE_REFERENCE,
    )


@dataclass(frozen=True)
class PlacementState:
    steps: tuple[StepPlacement, ...]
    transfers: tuple[DataTransfer, ...]
    outputs: tuple[tuple[str, DataRef], ...]
    resource_usage: tuple[tuple[str, float, int, int], ...]

    def output_map(self) -> dict[str, DataRef]:
        return dict(self.outputs)

    def usage_map(self) -> dict[str, tuple[float, int, int]]:
        return {
            node_id: (cpu, memory, gpu)
            for node_id, cpu, memory, gpu in self.resource_usage
        }


_SIZE_DEFAULTS: dict[str, int] = {
    "raw_video_frames.v1": 20_000_000,
    "camera_calibration.v1": 16_000,
    "route_graph.v1": 128_000,
    "detection_set.v1": 500_000,
    "track_set.v1": 250_000,
    "projected_track_set.v1": 300_000,
    "image_crop_set.v1": 5_000_000,
    "vehicle_reid_embedding_set.v1": 16_000,
    "canonical_entity_map.v1": 16_000,
    "pair_trajectory.v1": 128_000,
    "track_summary.v1": 96_000,
    "audio_segment.v1": 2_000_000,
    "audio_event_set.v1": 32_000,
    "predicate_match.v1": 4_000,
}


def estimated_size(data_type: str) -> int:
    return _SIZE_DEFAULTS.get(data_type, 64_000)


def deduplicate_states(states: Iterable[PlacementState]) -> list[PlacementState]:
    """Deduplicate without destroying the enumerator's feasibility ordering.

    Placement enumeration deliberately ranks input-local nodes first. Sorting
    the surviving states by an opaque deterministic hash before applying the
    beam cap made the cap effectively random: executable colocated pipelines
    could disappear while cross-worker variants survived. Input iteration is
    already deterministic, so insertion order is the correct stable order.
    """

    unique: dict[str, PlacementState] = {}
    for state in states:
        key = deterministic_id(
            "placement",
            {"steps": state.steps, "transfers": state.transfers},
            length=32,
        )
        unique.setdefault(key, state)
    return list(unique.values())


def add_edge(
    edges: dict[str, AlternativeGraphEdge],
    alternative_id: str,
    source_node_id: str,
    target_node_id: str,
    kind: AlternativeEdgeKind,
    data_type: str | None,
) -> None:
    edge_id = deterministic_id(
        "pae",
        {
            "alternative": alternative_id,
            "source": source_node_id,
            "target": target_node_id,
            "kind": kind,
            "data_type": data_type,
        },
    )
    edges[edge_id] = AlternativeGraphEdge(
        edge_id=edge_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        kind=kind,
        data_type=data_type,
    )


__all__ = [
    "DataRef",
    "PlacementState",
    "estimated_size",
    "deduplicate_states",
    "add_edge",
]
