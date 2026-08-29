"""Cross-sensor identity association and optional hosted-VLM fallback."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import sqrt
from typing import Callable, Mapping, Sequence

from .data_models import EmbeddingVector


@dataclass(frozen=True, slots=True)
class IdentityAssociation:
    left_object_id: str
    right_object_id: str
    cosine_similarity: float


class CrossSensorIdentityAssociationProvider:
    """Associate compatible ReID embeddings using global one-to-one matching.

    The provider refuses to compare different feature spaces (model ID/version),
    can restrict source pairs, and can reject observations too far apart in
    event time.  It returns only associations; the separate IdentityResolver is
    responsible for canonicalizing object identities.
    """

    provider_id = "cross_sensor_identity_association"
    provider_version = "2"

    def __init__(
        self,
        *,
        minimum_cosine_similarity: float = 0.75,
        maximum_time_gap_s: float | None = None,
        allowed_source_pairs: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        if not 0 <= minimum_cosine_similarity <= 1:
            raise ValueError("minimum_cosine_similarity must be in [0,1]")
        if maximum_time_gap_s is not None and maximum_time_gap_s < 0:
            raise ValueError("maximum_time_gap_s must be non-negative")
        self.minimum_cosine_similarity = float(minimum_cosine_similarity)
        self.maximum_time_gap_s = maximum_time_gap_s
        self.allowed_source_pairs = (
            {frozenset(pair) for pair in allowed_source_pairs}
            if allowed_source_pairs is not None
            else None
        )

    def associate_records(
        self,
        left: Sequence[EmbeddingVector],
        right: Sequence[EmbeddingVector],
    ) -> tuple[IdentityAssociation, ...]:
        candidates: list[tuple[float, str, str]] = []
        for a in left:
            for b in right:
                if a.object_id == b.object_id:
                    continue
                if a.model_id != b.model_id or a.model_version != b.model_version:
                    continue
                if a.source_id == b.source_id:
                    continue
                if self.allowed_source_pairs is not None and frozenset(
                    (a.source_id, b.source_id)
                ) not in self.allowed_source_pairs:
                    continue
                if self.maximum_time_gap_s is not None:
                    gap = abs((a.event_time - b.event_time).total_seconds())
                    if gap > self.maximum_time_gap_s:
                        continue
                score = _cosine(a.vector, b.vector)
                if score >= self.minimum_cosine_similarity:
                    candidates.append((score, a.object_id, b.object_id))

        # Global greedy one-to-one matching is deterministic and avoids the
        # left-order dependence of the previous implementation.
        candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
        used_left: set[str] = set()
        used_right: set[str] = set()
        output: list[IdentityAssociation] = []
        for score, left_id, right_id in candidates:
            if left_id in used_left or right_id in used_right:
                continue
            used_left.add(left_id)
            used_right.add(right_id)
            output.append(IdentityAssociation(left_id, right_id, score))
        return tuple(output)

    def associate(
        self,
        left: Sequence[EmbeddingVector],
        right: Sequence[EmbeddingVector],
    ) -> dict[str, str]:
        return {
            row.left_object_id: row.right_object_id
            for row in self.associate_records(left, right)
        }


class HostedVLMIdentityComparatorProvider:
    """Bounded hosted identity fallback with an injectable vision comparator."""

    provider_id = "hosted_vlm_identity_comparator"
    provider_version = "1"

    def __init__(
        self,
        comparator: Callable[[object, object], float],
        *,
        maximum_invocations: int = 4,
        threshold: float = 0.7,
    ) -> None:
        self.comparator = comparator
        self.maximum_invocations = maximum_invocations
        self.threshold = threshold
        self._calls = 0

    def compare(self, left_image: object, right_image: object) -> tuple[bool, float]:
        if self._calls >= self.maximum_invocations:
            raise RuntimeError("hosted VLM invocation limit reached")
        self._calls += 1
        score = max(0.0, min(1.0, float(self.comparator(left_image, right_image))))
        return score >= self.threshold, score


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)
