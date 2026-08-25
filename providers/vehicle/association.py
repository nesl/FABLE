"""Cross-camera identity association over typed descriptor artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

from fable.common.ids import deterministic_id
from fable.common.time import EventTimeInterval

from .errors import ArtifactCompatibilityError, InvalidProviderInput
from .models import (
    DescriptorRecord,
    DescriptorSet,
    EntityAssociation,
    EntityAssociationSet,
)


@dataclass(frozen=True)
class RouteTimeGate:
    left_entity_id: str
    right_entity_id: str
    compatible: bool


class CrossSensorIdentityAssociator:
    provider_id = "cross_sensor_identity_association"
    provider_version = "1"

    def __init__(
        self,
        *,
        maximum_cosine_distance: float = 0.25,
        require_identity_calibration: bool = True,
    ) -> None:
        if not 0 <= maximum_cosine_distance <= 2:
            raise ValueError("cosine distance threshold must be in [0, 2]")
        self.maximum_cosine_distance = maximum_cosine_distance
        self.require_identity_calibration = require_identity_calibration

    def associate(
        self,
        left: DescriptorSet,
        right: DescriptorSet,
        *,
        route_time_compatibility: Mapping[tuple[str, str], bool] | None = None,
    ) -> EntityAssociationSet:
        if left.entity_kind != right.entity_kind:
            raise ArtifactCompatibilityError(
                "person and vehicle descriptor feature spaces cannot be associated"
            )
        if left.compatibility_key != right.compatibility_key:
            raise ArtifactCompatibilityError(
                "descriptor feature spaces differ; cross-camera association is invalid"
            )
        if self.require_identity_calibration and (
            not left.calibrated_for_identity or not right.calibrated_for_identity
        ):
            raise ArtifactCompatibilityError(
                "general visual embeddings are not canonical identity evidence without calibration"
            )
        if left.distance_metric != "cosine" or right.distance_metric != "cosine":
            raise InvalidProviderInput("this associator requires cosine descriptors")
        gate = route_time_compatibility or {}
        candidates: list[tuple[float, str, str, DescriptorRecord, DescriptorRecord]] = []
        for left_record in left.records:
            for right_record in right.records:
                if not gate.get(
                    (left_record.local_entity_id, right_record.local_entity_id), True
                ):
                    continue
                distance = cosine_distance(left_record.vector, right_record.vector)
                if distance <= self.maximum_cosine_distance:
                    candidates.append(
                        (
                            distance,
                            left_record.local_entity_id,
                            right_record.local_entity_id,
                            left_record,
                            right_record,
                        )
                    )
        # Deterministic one-to-one greedy matching after a global distance sort.
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        used_left: set[str] = set()
        used_right: set[str] = set()
        associations: list[EntityAssociation] = []
        for distance, left_id, right_id, _, _ in candidates:
            if left_id in used_left or right_id in used_right:
                continue
            used_left.add(left_id)
            used_right.add(right_id)
            canonical_id = deterministic_id(
                f"canonical_{left.entity_kind}",
                {
                    "feature_space": left.compatibility_key,
                    "left": [left.source_id, left_id],
                    "right": [right.source_id, right_id],
                },
                length=32,
            )
            associations.append(
                EntityAssociation(
                    left_local_entity_id=left_id,
                    right_local_entity_id=right_id,
                    canonical_entity_id=canonical_id,
                    distance=distance,
                    confidence=max(0.0, 1.0 - distance / max(self.maximum_cosine_distance, 1e-9)),
                    route_time_compatible=True,
                    association_basis="reid",
                    association_model_id=left.model_id,
                )
            )
        start = min(left.event_time_interval.start, right.event_time_interval.start)
        end = max(left.event_time_interval.end, right.event_time_interval.end)
        return EntityAssociationSet(
            left_source_id=left.source_id,
            right_source_id=right.source_id,
            event_time_interval=EventTimeInterval(start=start, end=end),
            feature_space_key=left.compatibility_key,
            entity_kind=left.entity_kind,
            associations=tuple(associations),
            unmatched_left=tuple(
                sorted(record.local_entity_id for record in left.records if record.local_entity_id not in used_left)
            ),
            unmatched_right=tuple(
                sorted(record.local_entity_id for record in right.records if record.local_entity_id not in used_right)
            ),
        )


def cosine_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ArtifactCompatibilityError("descriptor vectors must have the same non-zero dimension")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        raise InvalidProviderInput("descriptor vectors must be non-zero")
    similarity = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity
