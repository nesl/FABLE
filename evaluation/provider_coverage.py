"""Fail-closed checks for providers selected by live evaluation plans."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from fable.distributed.config import ProviderRuntimeResolver
from fable.distributed.models import RuntimeMode
from fable.planning.models import PhysicalAlternativeGraph
from fable.planning.provider_registry import ProviderRegistry


# These are executable implementation symbols used by the adopted replay
# services. A catalog declaration or runtime YAML entry alone is insufficient.
IMPLEMENTATION_SYMBOLS = {
    "yolo_vehicle_fast_640": ("providers.vehicle.detector", "LegacyReplayYoloAdapter"),
    "yolo_full_context_960": ("providers.vehicle.detector", "LegacyReplayYoloAdapter"),
    "multi_object_tracker": ("providers.vehicle.tracker", "RoboflowTrackerAdapter"),
    "camera_projection": ("providers.vehicle.service", "VehicleReplayProcessor"),
    "route_map_matcher": ("providers.vehicle.geometry", "RouteMapMatcher"),
    "pass_reference_evaluator": ("providers.vehicle.geometry", "PassReferenceEvaluator"),
    "zone_membership_evaluator": ("providers.vehicle.geometry", "ZoneMembershipEvaluator"),
    "zone_transition_evaluator": ("providers.vehicle.geometry", "ZoneTransitionEvaluator"),
    "track_lifecycle_exit_evaluator": (
        "providers.vehicle.geometry",
        "TrackLifecycleExitEvaluator",
    ),
    "pairwise_distance_evaluator": ("providers.vehicle.geometry", "PairwiseDistanceEvaluator"),
    "motion_state_evaluator": ("providers.vehicle.geometry", "MotionStateEvaluator"),
    "follows_local_geometry": ("providers.vehicle.follows", "FollowsLocalGeometryEvaluator"),
    "audio_event_classifier": ("providers.multimodal.audio", "AudioEventClassifier"),
    "gcc_phat_audio_localizer": ("providers.multimodal.localization", "GccPhatAudioLocalizer"),
    "audio_visual_association": ("providers.multimodal.audiovisual", "AudioVisualAssociator"),
    "voice_activity_detector": ("providers.multimodal.conversation", "EnergyVoiceActivityDetector"),
    "speaker_embedding_provider": ("providers.multimodal.conversation", "SpectralSpeakerEmbeddingProvider"),
    "speaker_diarization_provider": ("providers.multimodal.conversation", "OnlineSpeakerDiarizer"),
    "conversation_provider": ("providers.multimodal.conversation", "ConversationEvaluator"),
    "person_proximity_provider": (
        "providers.multimodal.conversation",
        "PersonProximityEvaluator",
    ),
    "person_vehicle_relation_provider": ("providers.multimodal.person_vehicle", "PersonVehicleRelationEvaluator"),
    "package_detector": ("providers.multimodal.package_transfer", "PackageDetectionAdapter"),
    "object_transfer_reasoner": ("providers.multimodal.package_transfer", "TransferCustodyReasoner"),
    "historical_vehicle_interval_matcher": (
        "providers.vehicle.replay",
        "HistoricalVehicleIntervalMatcher",
    ),
    "hosted_vlm_identity_comparator": (
        "providers.vehicle.vlm_reid",
        "OpenAIVisionIdentityComparator",
    ),
}


@dataclass(frozen=True)
class ProviderCoverageIssue:
    provider_id: str
    node_id: str
    reason: str


def validate_live_provider_coverage(
    graph: PhysicalAlternativeGraph,
    *,
    registry: ProviderRegistry,
    runtimes: ProviderRuntimeResolver,
) -> tuple[ProviderCoverageIssue, ...]:
    issues: set[ProviderCoverageIssue] = set()
    for alternative in graph.alternatives:
        for placement in alternative.step_placements:
            provider_id = placement.provider_id
            node_id = placement.node_id
            if provider_id not in registry.providers:
                issues.add(ProviderCoverageIssue(provider_id, node_id, "not in provider catalog"))
                continue
            if not runtimes.has(node_id, provider_id):
                issues.add(ProviderCoverageIssue(provider_id, node_id, "no runtime mapping"))
                continue
            runtime = runtimes.resolve(node_id=node_id, provider_id=provider_id)
            if runtime.mode == RuntimeMode.REFERENCE:
                issues.add(ProviderCoverageIssue(provider_id, node_id, "reference runtime"))
                continue
            target = IMPLEMENTATION_SYMBOLS.get(provider_id)
            if target is None:
                issues.add(ProviderCoverageIssue(provider_id, node_id, "no audited implementation symbol"))
                continue
            module_name, symbol_name = target
            try:
                module = import_module(module_name)
                getattr(module, symbol_name)
            except Exception as exc:
                issues.add(
                    ProviderCoverageIssue(
                        provider_id,
                        node_id,
                        f"implementation import failed: {type(exc).__name__}: {exc}",
                    )
                )
    return tuple(sorted(issues, key=lambda item: (item.provider_id, item.node_id, item.reason)))
