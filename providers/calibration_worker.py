"""Provider-level JSON calibration worker included in analytics images."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter_ns

from providers.multimodal.audio import SpectralRuleAudioBackend
from providers.multimodal.audiovisual import AudioVisualAssociator
from providers.multimodal.conversation import (
    ConversationEvaluator,
    EnergyVoiceActivityDetector,
    OnlineSpeakerDiarizer,
    PersonProximityEvaluator,
    SpectralSpeakerEmbeddingProvider,
)
from providers.multimodal.localization import GccPhatAudioLocalizer
from providers.multimodal.models import (
    AudioEventObservation,
    AudioLocalization,
    AudioWindow,
    MicrophoneArrayGeometry,
    SpeakerTurnSet,
    SpeechSegment,
    VisualBearingCandidate,
)
from providers.multimodal.person_vehicle import PersonVehicleRelationEvaluator
from providers.multimodal.package_transfer import PackageDetectionAdapter
from providers.vehicle.geometry import PairwiseDistanceEvaluator
from providers.vehicle.follows import FollowsLocalGeometryEvaluator
from providers.vehicle.geometry import (
    MotionStateEvaluator,
    PassReferenceEvaluator,
    ReferenceLine,
    RouteMapMatcher,
    RoutePolyline,
    ZoneMembershipEvaluator,
    ZoneTransitionEvaluator,
)
from providers.vehicle.models import TrackObservation, TrackSet, VehicleZone
from providers.vehicle.replay import HistoricalVehicleIntervalMatcher
from providers.vehicle.detector import (
    DEFAULT_YOLO_VARIANTS,
    UltralyticsYoloDetector,
)
from providers.vehicle.models import DetectionFrame
from providers.vehicle.tracker import RoboflowTrackerAdapter

WORKER_OPERATIONS = {
    "audio_event_classifier": {
        "measurement_status": "IMPLEMENTATION_VALIDATION_ONLY",
        "input_classes": ("audio_segment.v1",),
        "reason": "spectral rule backend is not the final evaluation classifier",
    },
    "audio_visual_association": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": (
            "audio_event_set.v1+audio_localization.v1+visual_bearing_set.v1",
        ),
    },
    "conversation_provider": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": (
            "projected_track_set.v1+speaker_turn_set.v1",
            "projected_track_set.v1+speaker_turn_set.v1+transcript_event_set.v1",
        ),
    },
    "follows_local_geometry": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": (
            "projected_track_set.v1",
            "projected_track_set.v1+pair_trajectory.v1",
        ),
    },
    "gcc_phat_audio_localizer": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": (
            "audio_segment.v1+microphone_array_geometry.v1",
        ),
    },
    "motion_state_evaluator": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("projected_track_set.v1",),
    },
    "pairwise_distance_evaluator": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("projected_track_set.v1",),
    },
    "pass_reference_evaluator": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("projected_track_set.v1+route_graph.v1",),
    },
    "person_proximity_provider": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("track_set.v1",),
    },
    "person_vehicle_relation_provider": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("projected_track_set.v1",),
    },
    "historical_vehicle_interval_matcher": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("no_external_input", "track_set.v1"),
    },
    "route_map_matcher": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("projected_track_set.v1+route_graph.v1",),
    },
    "voice_activity_detector": {
        "measurement_status": "IMPLEMENTATION_VALIDATION_ONLY",
        "input_classes": ("audio_segment.v1",),
        "reason": "energy VAD is a smoke backend, not the configured WebRTC runtime",
    },
    "speaker_diarization_provider": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("speaker_embedding_set.v1+speech_segment_set.v1",),
    },
    "speaker_embedding_provider": {
        "measurement_status": "IMPLEMENTATION_VALIDATION_ONLY",
        "input_classes": ("audio_segment.v1+speech_segment_set.v1",),
        "reason": "spectral embedding backend is not the configured learned model",
    },
    "multi_object_tracker": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("detection_set.v1",),
    },
    "package_detector": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("raw_video_frames.v1",),
    },
    "yolo_full_context_960": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("raw_video_frames.v1",),
    },
    "yolo_vehicle_fast_640": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("raw_video_frames.v1",),
    },
    "zone_membership_evaluator": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("projected_track_set.v1+route_graph.v1",),
    },
    "zone_transition_evaluator": {
        "measurement_status": "MEASURED_PROVIDER",
        "input_classes": ("projected_track_set.v1+route_graph.v1",),
    },
}


def main() -> int:
    if "--capabilities" in sys.argv[1:]:
        print(json.dumps(worker_capabilities(), sort_keys=True))
        return 0
    if "--serve" in sys.argv[1:]:
        return _serve()
    request = json.load(sys.stdin)
    print(json.dumps(process_request(request), sort_keys=True))
    return 0


def worker_capabilities() -> dict:
    return {
        "schema_version": "fable.calibration_worker_capabilities.v1",
        "operations": WORKER_OPERATIONS,
    }


def process_request(request: dict) -> dict:
    if request.get("schema_version") != "fable.calibration_worker_request.v1":
        raise ValueError("calibration worker request schema mismatch")
    target = request["target"]
    fixture = request["fixture"]
    provider_id = target["provider_id"]
    capability = WORKER_OPERATIONS.get(provider_id)
    if capability is None:
        raise ValueError(
            f"provider calibration operation is not implemented: {provider_id}"
        )
    if target.get("tier") != "sensor":
        raise ValueError("provider calibration worker accepts only the sensor tier")
    if target.get("input_class") not in capability["input_classes"]:
        raise ValueError(
            "provider calibration input signature is not implemented: "
            f"{provider_id}/{target.get('input_class')}"
        )
    started = perf_counter_ns()
    if provider_id == "pairwise_distance_evaluator":
        response = _pairwise_distance(fixture)
    elif provider_id == "motion_state_evaluator":
        response = _motion_state(fixture)
    elif provider_id == "route_map_matcher":
        response = _route_match(fixture)
    elif provider_id == "zone_membership_evaluator":
        response = _zone_membership(fixture)
    elif provider_id == "follows_local_geometry":
        response = _follows_local(fixture)
    elif provider_id == "zone_transition_evaluator":
        response = _zone_transition(fixture)
    elif provider_id == "pass_reference_evaluator":
        response = _pass_reference(fixture)
    elif provider_id == "person_proximity_provider":
        response = _person_proximity(fixture)
    elif provider_id == "conversation_provider":
        response = _conversation(fixture)
    elif provider_id == "voice_activity_detector":
        response = _voice_activity(fixture)
    elif provider_id == "gcc_phat_audio_localizer":
        response = _audio_localization(fixture)
    elif provider_id == "audio_visual_association":
        response = _audio_visual_association(fixture)
    elif provider_id == "historical_vehicle_interval_matcher":
        response = _historical_match(fixture)
    elif provider_id == "person_vehicle_relation_provider":
        response = _person_vehicle_relation(fixture)
    elif provider_id == "speaker_diarization_provider":
        response = _speaker_diarization(fixture)
    elif provider_id == "speaker_embedding_provider":
        response = _speaker_embedding(fixture)
    elif provider_id in {"yolo_full_context_960", "yolo_vehicle_fast_640"}:
        response = _yolo_detection(provider_id, fixture)
    elif provider_id == "multi_object_tracker":
        response = _multi_object_tracking(fixture)
    elif provider_id == "package_detector":
        response = _package_detection(fixture)
    elif provider_id == "audio_event_classifier":
        response = _spectral_audio(fixture)
    else:  # guarded by WORKER_OPERATIONS above
        raise AssertionError(f"missing worker dispatch for {provider_id}")
    response["provider_execution_ms"] = (
        perf_counter_ns() - started
    ) / 1_000_000
    return {
        "schema_version": "fable.calibration_worker_response.v1",
        **response,
    }


def _serve() -> int:
    """Keep imports and provider state resident across warm JSONL requests."""

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            response = process_request(json.loads(line))
        except Exception as exc:
            response = {
                "schema_version": "fable.calibration_worker_response.v1",
                "successful": False,
                "quality_score": 0.0,
                "ambiguity_score": 2.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def _pairwise_distance(fixture: dict) -> dict:
    left = TrackObservation.model_validate(fixture["left"])
    right = TrackObservation.model_validate(fixture["right"])
    threshold = float(fixture["maximum_distance_m"])
    observed = PairwiseDistanceEvaluator().evaluate(
        left,
        right,
        maximum_distance_m=threshold,
    )
    expected = bool(fixture["expected_truth"])
    distance = float(observed.measurements["distance_m"])
    scale = max(threshold, 1e-9)
    ambiguity = max(0.0, 1.0 - min(1.0, abs(distance - threshold) / scale))
    return {
        "successful": observed.truth == expected,
        "quality_score": 1.0 if observed.truth == expected else 0.0,
        "ambiguity_score": ambiguity,
    }


def _spectral_audio(fixture: dict) -> dict:
    window = AudioWindow.model_validate(fixture["window"])
    expected_label = str(fixture["expected_label"])
    scores = SpectralRuleAudioBackend().score(window)
    predicted, confidence = max(scores.items(), key=lambda item: item[1])
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    return {
        "successful": predicted == expected_label,
        "quality_score": confidence if predicted == expected_label else 0.0,
        "ambiguity_score": 1.0 - max(0.0, min(1.0, margin)),
    }


def _motion_state(fixture: dict) -> dict:
    evaluator = MotionStateEvaluator(
        minimum_window_s=float(fixture.get("minimum_window_s", 0))
    )
    outputs = ()
    for raw in fixture["track_sets"]:
        outputs = evaluator.update(TrackSet.model_validate(raw))
    expected = str(fixture["expected_predicate_id"])
    matched = any(item.predicate_id == expected and item.truth for item in outputs)
    confidence = max(
        (item.confidence for item in outputs if item.predicate_id == expected),
        default=0.0,
    )
    return _classification_result(matched, confidence)


def _route_match(fixture: dict) -> dict:
    track = TrackObservation.model_validate(fixture["track"])
    routes = tuple(
        RoutePolyline.model_validate(item) for item in fixture["routes"]
    )
    matched = RouteMapMatcher().match(track, routes)
    correct = matched.route_id == str(fixture["expected_route_id"])
    distance = float(matched.attributes.get("route_distance_m", 0))
    ambiguity = min(2.0, distance / max(float(fixture.get("distance_scale_m", 10)), 1e-9))
    return {
        "successful": correct,
        "quality_score": matched.confidence if correct else 0.0,
        "ambiguity_score": ambiguity,
    }


def _zone_membership(fixture: dict) -> dict:
    observed = ZoneMembershipEvaluator().evaluate(
        TrackObservation.model_validate(fixture["track"]),
        VehicleZone.model_validate(fixture["zone"]),
    )
    expected = bool(fixture["expected_truth"])
    return _classification_result(observed.truth == expected, observed.confidence)


def _follows_local(fixture: dict) -> dict:
    evaluator = FollowsLocalGeometryEvaluator(
        maximum_gap_m=float(fixture.get("maximum_gap_m", 15)),
        minimum_duration_s=float(fixture.get("minimum_duration_s", 0)),
    )
    outputs = ()
    for raw in fixture["track_sets"]:
        outputs = evaluator.update(
            TrackSet.model_validate(raw),
            leader_id=str(fixture["leader_id"]),
            follower_id=str(fixture["follower_id"]),
        )
    matched = any(item.truth and item.predicate_id == "FOLLOWS" for item in outputs)
    confidence = max((item.confidence for item in outputs), default=0.0)
    return _classification_result(
        matched == bool(fixture.get("expected_truth", True)),
        confidence,
    )


def _zone_transition(fixture: dict) -> dict:
    evaluator = ZoneTransitionEvaluator()
    zone = VehicleZone.model_validate(fixture["zone"])
    outputs = []
    for raw in fixture["tracks"]:
        observed = evaluator.update(TrackObservation.model_validate(raw), zone)
        if observed is not None:
            outputs.append(observed)
    expected = str(fixture["expected_predicate_id"])
    matched = next(
        (item for item in outputs if item.predicate_id == expected and item.truth),
        None,
    )
    return _classification_result(
        matched is not None,
        matched.confidence if matched is not None else 0.0,
    )


def _pass_reference(fixture: dict) -> dict:
    evaluator = PassReferenceEvaluator()
    reference = ReferenceLine.model_validate(fixture["reference"])
    outputs = []
    for raw in fixture["tracks"]:
        observed = evaluator.update(
            TrackObservation.model_validate(raw),
            reference,
        )
        if observed is not None:
            outputs.append(observed)
    matched = next(
        (item for item in outputs if item.predicate_id == "PASSES" and item.truth),
        None,
    )
    return _classification_result(
        (matched is not None) == bool(fixture.get("expected_truth", True)),
        matched.confidence if matched is not None else 1.0,
    )


def _person_proximity(fixture: dict) -> dict:
    evaluator = PersonProximityEvaluator(
        maximum_normalized_gap=float(
            fixture.get("maximum_normalized_gap", 2.5)
        ),
        minimum_duration_seconds=float(
            fixture.get("minimum_duration_seconds", 0)
        ),
    )
    outputs = ()
    for raw in fixture["track_sets"]:
        outputs = evaluator.update(TrackSet.model_validate(raw))
    matched = next(
        (
            item
            for item in outputs
            if item.predicate_id == "PERSON_PROXIMITY" and item.truth
        ),
        None,
    )
    return _classification_result(
        (matched is not None) == bool(fixture.get("expected_truth", True)),
        matched.confidence if matched is not None else 1.0,
    )


def _conversation(fixture: dict) -> dict:
    observed = ConversationEvaluator(
        maximum_distance_m=float(fixture.get("maximum_distance_m", 2.5)),
        minimum_duration_seconds=float(
            fixture.get("minimum_duration_seconds", 0)
        ),
        minimum_speakers=int(fixture.get("minimum_speakers", 2)),
    ).evaluate(
        tuple(
            TrackSet.model_validate(item) for item in fixture["track_sets"]
        ),
        SpeakerTurnSet.model_validate(fixture["speaker_turn_set"]),
        required_terms=tuple(fixture.get("required_terms", ())),
    )
    matched = observed is not None
    return _classification_result(
        matched == bool(fixture.get("expected_truth", True)),
        observed.confidence if observed is not None else 1.0,
    )


def _voice_activity(fixture: dict) -> dict:
    segments = EnergyVoiceActivityDetector(
        rms_threshold=float(fixture.get("rms_threshold", 0.015))
    ).detect(AudioWindow.model_validate(fixture["window"]))
    matched = bool(segments)
    confidence = (
        max(item.speech_probability for item in segments) if segments else 1.0
    )
    return _classification_result(
        matched == bool(fixture["expected_speech"]),
        confidence,
    )


def _audio_localization(fixture: dict) -> dict:
    observed = GccPhatAudioLocalizer(
        MicrophoneArrayGeometry.model_validate(fixture["geometry"])
    ).localize(AudioWindow.model_validate(fixture["window"]))
    expected = float(fixture["expected_azimuth_deg"])
    tolerance = float(fixture.get("azimuth_tolerance_deg", 15.0))
    error = abs(((observed.azimuth_deg - expected + 180.0) % 360.0) - 180.0)
    return {
        "successful": error <= tolerance,
        "quality_score": observed.confidence if error <= tolerance else 0.0,
        "ambiguity_score": min(2.0, error / max(tolerance, 1e-9)),
    }


def _audio_visual_association(fixture: dict) -> dict:
    observed = AudioVisualAssociator().associate(
        AudioEventObservation.model_validate(fixture["event"]),
        AudioLocalization.model_validate(fixture["localization"]),
        tuple(
            VisualBearingCandidate.model_validate(item)
            for item in fixture["candidates"]
        ),
    )
    expected = str(fixture["expected_local_entity_id"])
    match = next(
        (
            item
            for item in observed.associations
            if item.local_entity_id == expected
        ),
        None,
    )
    return _classification_result(
        match is not None,
        match.score if match is not None else 0.0,
    )


def _historical_match(fixture: dict) -> dict:
    observed = HistoricalVehicleIntervalMatcher().match_many(
        tuple(
            TrackSet.model_validate(item) for item in fixture["track_sets"]
        ),
        entity_kind=str(fixture.get("entity_kind", "vehicle")),
        bound_entity_id=fixture.get("bound_entity_id"),
        maximum_candidates=int(fixture.get("maximum_candidates", 8)),
    )
    expected = str(fixture["expected_scoped_track_id"])
    match = next(
        (item for item in observed if item.scoped_track_id == expected),
        None,
    )
    return _classification_result(
        match is not None,
        match.confidence if match is not None else 0.0,
    )


def _person_vehicle_relation(fixture: dict) -> dict:
    evaluator = PersonVehicleRelationEvaluator(
        minimum_stop_seconds=float(fixture.get("minimum_stop_seconds", 0)),
        transition_window_seconds=float(
            fixture.get("transition_window_seconds", 8)
        ),
    )
    outputs = ()
    for raw in fixture["track_sets"]:
        outputs = evaluator.update(TrackSet.model_validate(raw))
    expected = str(fixture["expected_predicate_id"])
    match = next(
        (item for item in outputs if item.predicate_id == expected and item.truth),
        None,
    )
    return _classification_result(
        match is not None,
        match.confidence if match is not None else 0.0,
    )


def _speaker_diarization(fixture: dict) -> dict:
    observed = OnlineSpeakerDiarizer(
        maximum_cosine_distance=float(
            fixture.get("maximum_cosine_distance", 0.25)
        )
    ).diarize(
        tuple(
            SpeechSegment.model_validate(item) for item in fixture["segments"]
        )
    )
    expected = int(fixture["expected_speaker_count"])
    correct = observed.speaker_count == expected
    return _classification_result(correct, 1.0 if correct else 0.0)


def _speaker_embedding(fixture: dict) -> dict:
    segments = tuple(
        SpeechSegment.model_validate(item) for item in fixture["segments"]
    )
    observed = SpectralSpeakerEmbeddingProvider(
        dimension=int(fixture.get("dimension", 16))
    ).attach(AudioWindow.model_validate(fixture["window"]), segments)
    valid = bool(observed) and all(item.embedding for item in observed)
    return _classification_result(
        valid == bool(fixture.get("expected_embedding", True)),
        1.0 if valid else 0.0,
    )


def _load_fixture_image(fixture: dict):
    path = Path(str(fixture["image_path"])).resolve(strict=True)
    if not path.is_file():
        raise ValueError("calibration image path is not a regular file")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("media-backed calibration requires OpenCV") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("calibration image could not be decoded")
    return image


def _detector(provider_id: str, fixture: dict) -> UltralyticsYoloDetector:
    variant = DEFAULT_YOLO_VARIANTS[provider_id]
    if fixture.get("model_path"):
        variant = replace(variant, model_path=str(fixture["model_path"]))
    if fixture.get("device"):
        variant = replace(variant, device=str(fixture["device"]))
    return UltralyticsYoloDetector(variant)


def _detect(provider_id: str, fixture: dict) -> DetectionFrame:
    return _detector(provider_id, fixture).detect(
        _load_fixture_image(fixture),
        source_id=str(fixture.get("source_id", "calibration-camera")),
        event_time=datetime.fromisoformat(
            str(fixture.get("event_time", "2026-01-01T00:00:00+00:00"))
        ),
        frame_id=str(fixture.get("frame_id", "calibration-frame")),
    )


def _yolo_detection(provider_id: str, fixture: dict) -> dict:
    observed = _detect(provider_id, fixture)
    minimum = int(fixture.get("minimum_detection_count", 1))
    required_labels = {
        str(item).lower() for item in fixture.get("required_labels", ())
    }
    observed_labels = {item.class_name.lower() for item in observed.detections}
    correct = (
        len(observed.detections) >= minimum
        and required_labels.issubset(observed_labels)
    )
    confidence = max(
        (item.confidence for item in observed.detections),
        default=0.0,
    )
    return _classification_result(correct, confidence)


def _multi_object_tracking(fixture: dict) -> dict:
    tracker = RoboflowTrackerAdapter(
        algorithm=str(fixture.get("algorithm", "bytetrack")),
        frame_rate=float(fixture.get("frame_rate", 30)),
    )
    outputs = tuple(
        tracker.update(DetectionFrame.model_validate(item))
        for item in fixture["detection_frames"]
    )
    final_tracks = outputs[-1].tracks if outputs else ()
    minimum = int(fixture.get("minimum_track_count", 1))
    correct = len(final_tracks) >= minimum
    confidence = max((item.confidence for item in final_tracks), default=0.0)
    return _classification_result(correct, confidence)


def _package_detection(fixture: dict) -> dict:
    detections = _detect("yolo_full_context_960", fixture)
    selected = PackageDetectionAdapter(
        confidence_threshold=float(fixture.get("confidence_threshold", 0.25))
    ).filter(detections)
    minimum = int(fixture.get("minimum_package_count", 1))
    correct = len(selected.detections) >= minimum
    confidence = max(
        (item.confidence for item in selected.detections),
        default=0.0,
    )
    return _classification_result(correct, confidence)


def _classification_result(correct: bool, confidence: float) -> dict:
    confidence = max(0.0, min(1.0, float(confidence)))
    return {
        "successful": correct,
        "quality_score": confidence if correct else 0.0,
        "ambiguity_score": 1.0 - confidence,
    }


if __name__ == "__main__":
    raise SystemExit(main())
