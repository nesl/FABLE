"""Temporal, angular, and zone-gated audio-visual association."""

from __future__ import annotations

from math import exp
from typing import Sequence

from fable.common.time import EventTimeInterval

from .models import (
    AudioEventObservation,
    AudioLocalization,
    AudioVisualAssociation,
    AudioVisualAssociationSet,
    InteractionPredicateObservation,
    VisualBearingCandidate,
    phase8_occurrence_id,
)


class AudioVisualAssociator:
    """Rank visual candidates without claiming canonical identity.

    An association is evidence that one visible entity is directionally and
    temporally compatible with an audio event.  It does not by itself prove
    that the entity caused the sound, and it never rewrites event bindings.
    """

    def __init__(
        self,
        *,
        maximum_angular_error_deg: float = 35.0,
        angular_scale_deg: float = 12.0,
        minimum_score: float = 0.25,
        provider_id: str = "audio_visual_association",
        provider_version: str = "1",
    ) -> None:
        if maximum_angular_error_deg <= 0 or angular_scale_deg <= 0:
            raise ValueError("audio-visual angular thresholds must be positive")
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum score must be in [0, 1]")
        self.maximum_angular_error_deg = maximum_angular_error_deg
        self.angular_scale_deg = angular_scale_deg
        self.minimum_score = minimum_score
        self.provider_id = provider_id
        self.provider_version = provider_version

    def associate(
        self,
        event: AudioEventObservation,
        localization: AudioLocalization,
        candidates: Sequence[VisualBearingCandidate],
    ) -> AudioVisualAssociationSet:
        if event.source_id != localization.source_id:
            raise ValueError("audio event and localization must share a source")
        overlap = event.event_time_interval.intersection(localization.event_time_interval)
        if overlap is None:
            raise ValueError("audio event and localization do not overlap in event time")
        associations: list[AudioVisualAssociation] = []
        for candidate in candidates:
            candidate_overlap = overlap.intersection(candidate.event_time_interval)
            if candidate_overlap is None:
                continue
            angular_error = _angular_distance(
                localization.azimuth_deg,
                candidate.azimuth_deg,
            )
            if angular_error > self.maximum_angular_error_deg:
                continue
            zone_compatible = (
                localization.zone_id is None
                or candidate.zone_id is None
                or localization.zone_id == candidate.zone_id
            )
            if not zone_compatible:
                continue
            angular_score = exp(-angular_error / self.angular_scale_deg)
            score = (
                angular_score
                * event.confidence
                * localization.confidence
                * candidate.confidence
            )
            score = max(0.0, min(1.0, score))
            if score < self.minimum_score:
                continue
            associations.append(
                AudioVisualAssociation(
                    audio_occurrence_id=event.occurrence_id,
                    local_entity_id=candidate.local_entity_id,
                    entity_type=candidate.entity_type,
                    score=score,
                    angular_error_deg=angular_error,
                    zone_compatible=zone_compatible,
                )
            )
        associations.sort(key=lambda item: (-item.score, item.local_entity_id))
        return AudioVisualAssociationSet(
            source_id=event.source_id,
            event_time_interval=overlap,
            audio_occurrence_id=event.occurrence_id,
            associations=tuple(associations),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
        )

    def best_predicate_observation(
        self,
        event: AudioEventObservation,
        localization: AudioLocalization,
        candidates: Sequence[VisualBearingCandidate],
        *,
        entity_role: str = "person",
    ) -> InteractionPredicateObservation | None:
        result = self.associate(event, localization, candidates)
        if not result.associations:
            return None
        best = result.associations[0]
        bindings = {
            "location": localization.zone_id or event.source_id,
            entity_role: best.local_entity_id,
        }
        return InteractionPredicateObservation(
            occurrence_id=phase8_occurrence_id(
                "AUDIO_VISUAL_ASSOCIATION",
                bindings,
                result.event_time_interval,
                self.provider_id,
            ),
            predicate_id="AUDIO_VISUAL_ASSOCIATION",
            truth=True,
            confidence=best.score,
            event_time_interval=result.event_time_interval,
            bindings=bindings,
            source_ids=(event.source_id,),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            supporting_artifact_types=(
                "audio_event_set.v1",
                "audio_localization.v1",
                "track_summary.v1",
            ),
            measurements={
                "audio_label": event.label,
                "angular_error_deg": best.angular_error_deg,
                "candidate_count": len(result.associations),
            },
        )


def _angular_distance(left: float, right: float) -> float:
    return abs(((left - right + 180.0) % 360.0) - 180.0)
