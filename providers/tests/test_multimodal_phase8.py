from __future__ import annotations

from datetime import timedelta

import numpy as np

from fable.common.examples import BASE_TIME
from fable.common.time import EventTimeInterval
from providers.multimodal.audio import (
    AudioEventClassifier,
    AudioEventThreshold,
    DeterministicAudioEventBackend,
    YamNetBackend,
)
from providers.multimodal.audiovisual import AudioVisualAssociator
from providers.multimodal.conversation import ConversationEvaluator, OnlineSpeakerDiarizer
from providers.multimodal.localization import gcc_phat_delay
from providers.multimodal.models import (
    AudioEventObservation,
    AudioLocalization,
    AudioWindow,
    SpeechSegment,
    VisualBearingCandidate,
)
from providers.multimodal.package_transfer import PackageDetectionAdapter, TransferCustodyReasoner
from providers.multimodal.person_vehicle import PersonVehicleRelationEvaluator
from providers.vehicle.models import (
    BoundingBox,
    Detection,
    DetectionFrame,
    Point2D,
    TrackObservation,
    TrackSet,
    scoped_track_identity,
)


def window(second: float, *, amplitude: float = 0.1, channels: int = 2) -> AudioWindow:
    start = BASE_TIME + timedelta(seconds=second)
    samples = tuple(float(amplitude) for _ in range(160))
    return AudioWindow(
        source_id="mic_a",
        event_time_interval=EventTimeInterval(start=start, end=start + timedelta(milliseconds=10)),
        sample_rate_hz=16_000,
        channel_ids=tuple(f"ch{i}" for i in range(channels)),
        waveform=tuple(samples for _ in range(channels)),
    )


def track(local_id: int, class_name: str, second: float, x: float, y: float, *, speed: float | None = 0.0) -> TrackObservation:
    session = "phase8_session"
    source = "camera_a"
    return TrackObservation(
        local_track_id=local_id,
        scoped_track_id=scoped_track_identity(source, session, local_id),
        source_id=source,
        tracker_session_id=session,
        class_name=class_name,
        confidence=0.95,
        bbox=BoundingBox(x1=x, y1=y, x2=x + 1.0, y2=y + 1.0),
        event_time=BASE_TIME + timedelta(seconds=second),
        world_point=Point2D(x=x, y=y, coordinate_frame_id="world"),
        velocity_mps=speed,
    )


def track_set(second: float, *rows: TrackObservation) -> TrackSet:
    return TrackSet(
        source_id="camera_a",
        tracker_family="fake",
        tracker_version="1",
        tracker_session_id="phase8_session",
        event_time=BASE_TIME + timedelta(seconds=second),
        tracks=tuple(rows),
    )


def test_audio_event_classifier_aliases_debounce_and_refractory() -> None:
    classifier = AudioEventClassifier(
        DeterministicAudioEventBackend({"Gunshot, gunfire": 0.91, "Fire alarm": 0.8}),
        thresholds=(
            AudioEventThreshold("gunshot", 0.5, 1, 1.0),
            AudioEventThreshold("alarm", 0.5, 2, 0.0),
        ),
    )
    first = classifier.classify(window(0))
    assert [item.label for item in first] == ["gunshot"]
    second = classifier.classify(window(0.2))
    assert [item.label for item in second] == ["alarm"]
    # Gunshot is suppressed inside its refractory period.
    assert all(item.label != "gunshot" for item in second)
    later = classifier.classify(window(1.5))
    assert {item.label for item in later} == {"gunshot"}
    alarm_again = classifier.classify(window(1.7))
    assert {item.label for item in alarm_again} == {"alarm"}


def test_yamnet_adapter_uses_explicit_class_map() -> None:
    class FakeTensor:
        def numpy(self):
            return np.asarray([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)

    backend = YamNetBackend(
        model=lambda waveform: (FakeTensor(), None, None),
        class_names=("quiet", "Gunshot, gunfire"),
        model_version="fixture",
    )
    scores = backend.score(window(0, channels=1))
    assert scores["quiet"] == pytest_approx(0.3)
    assert scores["Gunshot, gunfire"] == pytest_approx(0.7)


def pytest_approx(value: float, tolerance: float = 1e-5):
    import pytest

    return pytest.approx(value, abs=tolerance)


def test_gcc_phat_detects_relative_impulse_delay() -> None:
    reference = np.zeros(256, dtype=np.float32)
    signal = np.zeros(256, dtype=np.float32)
    reference[80] = 1.0
    signal[84] = 1.0
    delay, quality = gcc_phat_delay(signal, reference, sample_rate_hz=16_000, interpolation=8)
    assert delay == pytest_approx(4 / 16_000, tolerance=2 / (16_000 * 8))
    assert quality > 0.1


def test_audio_visual_association_applies_time_angle_and_zone_gates() -> None:
    interval = EventTimeInterval(start=BASE_TIME, end=BASE_TIME + timedelta(seconds=1))
    event = AudioEventObservation(
        occurrence_id="audio-1",
        label="gunshot",
        confidence=0.9,
        event_time_interval=interval,
        source_id="mic_a",
        provider_id="audio_event_classifier",
        provider_version="1",
    )
    localization = AudioLocalization(
        localization_id="loc-1",
        source_id="mic_a",
        array_id="array",
        event_time_interval=interval,
        azimuth_deg=10.0,
        confidence=0.9,
        zone_id="front",
    )
    candidates = (
        VisualBearingCandidate(
            local_entity_id="person_near",
            entity_type="person",
            source_id="camera_a",
            event_time_interval=interval,
            azimuth_deg=12.0,
            zone_id="front",
            confidence=0.95,
        ),
        VisualBearingCandidate(
            local_entity_id="person_wrong_zone",
            entity_type="person",
            source_id="camera_a",
            event_time_interval=interval,
            azimuth_deg=11.0,
            zone_id="rear",
        ),
        VisualBearingCandidate(
            local_entity_id="person_far_angle",
            entity_type="person",
            source_id="camera_a",
            event_time_interval=interval,
            azimuth_deg=100.0,
            zone_id="front",
        ),
    )
    result = AudioVisualAssociator(minimum_score=0.1).associate(event, localization, candidates)
    assert [item.local_entity_id for item in result.associations] == ["person_near"]


def test_person_vehicle_disembarks_and_boards() -> None:
    evaluator = PersonVehicleRelationEvaluator(
        proximity_m=2.0,
        separation_m=4.0,
        vehicle_departure_m=2.0,
        minimum_stop_seconds=0.0,
    )
    vehicle0 = track(1, "car", 0, 0, 0, speed=0.0)
    assert evaluator.update(track_set(0, vehicle0)) == ()
    person1 = track(2, "person", 1, 1, 0, speed=0.5)
    vehicle1 = track(1, "car", 1, 0, 0, speed=0.0)
    assert evaluator.update(track_set(1, vehicle1, person1)) == ()
    person2 = track(2, "person", 2, 5, 0, speed=1.0)
    vehicle2 = track(1, "car", 2, 0, 0, speed=0.0)
    result = evaluator.update(track_set(2, vehicle2, person2))
    assert [item.predicate_id for item in result] == ["DISEMBARKS"]

    # Move the person back near the car, remove them, then move the car.
    person3 = track(3, "person", 3, 1, 0, speed=0.0)
    vehicle3 = track(1, "car", 3, 0, 0, speed=0.0)
    evaluator.update(track_set(3, vehicle3, person3))
    vehicle4 = track(1, "car", 4, 0, 0, speed=0.0)
    evaluator.update(track_set(4, vehicle4))
    vehicle5 = track(1, "car", 5, 3, 0, speed=3.0)
    result = evaluator.update(track_set(5, vehicle5))
    assert [item.predicate_id for item in result] == ["BOARDS"]


def test_conversation_uses_diarization_and_invokes_asr_only_for_content() -> None:
    diarizer = OnlineSpeakerDiarizer(maximum_cosine_distance=0.2)
    segments = (
        SpeechSegment(
            segment_id="s1",
            source_id="mic_a",
            event_time_interval=EventTimeInterval(start=BASE_TIME, end=BASE_TIME + timedelta(seconds=1)),
            speech_probability=0.9,
            embedding=(1.0, 0.0),
        ),
        SpeechSegment(
            segment_id="s2",
            source_id="mic_a",
            event_time_interval=EventTimeInterval(start=BASE_TIME + timedelta(seconds=1), end=BASE_TIME + timedelta(seconds=2)),
            speech_probability=0.9,
            embedding=(0.0, 1.0),
        ),
    )
    turns = diarizer.diarize(segments)
    history = (
        track_set(0, track(10, "person", 0, 0, 0), track(11, "person", 0, 1, 0)),
        track_set(2, track(10, "person", 2, 0, 0), track(11, "person", 2, 1, 0)),
    )
    evaluator = ConversationEvaluator(minimum_duration_seconds=1.0)

    class FakeAsr:
        provider_id = "fake_asr"
        provider_version = "1"

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, turns):
            self.calls += 1
            return "hand over the package"

    asr = FakeAsr()
    plain = evaluator.evaluate(history, turns, asr_provider=asr)
    assert plain is not None
    assert plain.measurements["asr_invoked"] is False
    assert asr.calls == 0
    content = evaluator.evaluate(
        history,
        turns,
        required_terms=("package",),
        asr_provider=asr,
    )
    assert content is not None
    assert content.measurements["asr_invoked"] is True
    assert asr.calls == 1


def test_package_detector_and_compact_custody_transfer() -> None:
    frame = DetectionFrame(
        source_id="camera_a",
        event_time=BASE_TIME,
        frame_id="f0",
        detector_id="full_context",
        detector_version="1",
        detections=(
            Detection(
                detection_id="bag",
                class_name="backpack",
                confidence=0.8,
                bbox=BoundingBox(x1=0, y1=0, x2=1, y2=1),
            ),
            Detection(
                detection_id="car",
                class_name="car",
                confidence=0.9,
                bbox=BoundingBox(x1=0, y1=0, x2=4, y2=2),
            ),
        ),
    )
    filtered = PackageDetectionAdapter().filter(frame)
    assert [item.detection_id for item in filtered.detections] == ["bag"]

    reasoner = TransferCustodyReasoner(
        maximum_holder_distance_m=2.0,
        minimum_stable_seconds=0.0,
    )
    package0 = track(20, "backpack", 0, 0.2, 0)
    source0 = track(21, "person", 0, 0, 0)
    destination0 = track(22, "person", 0, 10, 0)
    reasoner.update(track_set(0, package0, source0, destination0))
    state, events = reasoner.update(
        track_set(0.1, track(20, "backpack", 0.1, 0.2, 0), track(21, "person", 0.1, 0, 0), track(22, "person", 0.1, 10, 0))
    )
    assert len(state.records) == 1 and not events

    # The package becomes stably closest to the destination.
    reasoner.update(
        track_set(1, track(20, "backpack", 1, 10.2, 0), track(21, "person", 1, 0, 0), track(22, "person", 1, 10, 0))
    )
    state, events = reasoner.update(
        track_set(1.1, track(20, "backpack", 1.1, 10.2, 0), track(21, "person", 1.1, 0, 0), track(22, "person", 1.1, 10, 0))
    )
    assert [item.predicate_id for item in events] == ["TRANSFER"]
    assert state.records[0].previous_holder_id == scoped_track_identity("camera_a", "phase8_session", 21)
    assert state.records[0].holder_id == scoped_track_identity("camera_a", "phase8_session", 22)


def test_replay_processor_emits_audio_visual_association_from_buffered_context() -> None:
    from providers.multimodal.service import MultimodalReplayProcessor, MultimodalServiceConfig

    class FakeLocalizer:
        def localize(self, audio_window):
            return AudioLocalization(
                localization_id="loc-buffered",
                source_id=audio_window.source_id,
                array_id="array",
                event_time_interval=audio_window.event_time_interval,
                azimuth_deg=0.0,
                confidence=0.95,
                zone_id="front",
            )

    class ContextTracker:
        def update(self, detections):
            row = TrackObservation(
                local_track_id=30,
                scoped_track_id=scoped_track_identity("camera_a", "context", 30),
                source_id="camera_a",
                tracker_session_id="context",
                class_name="person",
                confidence=0.95,
                bbox=BoundingBox(x1=620, y1=100, x2=660, y2=200),
                event_time=detections.event_time,
            )
            return TrackSet(
                source_id="camera_a",
                tracker_family="fake",
                tracker_version="1",
                tracker_session_id="context",
                event_time=detections.event_time,
                tracks=(row,),
            )

    config = MultimodalServiceConfig(
        source_id="camera_a",
        raw_audio_topic="local://respeaker",
        yolo_topic="/yolo",
        audio_event_topic="/audio",
        localization_topic="/loc",
        speech_turn_topic="/turns",
        context_track_topic="/tracks",
        interaction_topic="/interactions",
        custody_topic="/custody",
        readiness_topic="/ready",
        visual_image_width_px=1280,
        camera_horizontal_fov_deg=90,
        visual_zone_id="front",
        association_time_tolerance_seconds=0.5,
    )
    processor = MultimodalReplayProcessor(
        config=config,
        context_tracker=ContextTracker(),  # type: ignore[arg-type]
        audio_classifier=AudioEventClassifier(
            DeterministicAudioEventBackend({"Gunshot, gunfire": 0.9}),
            thresholds=(AudioEventThreshold("gunshot", 0.5),),
        ),
        localizer=FakeLocalizer(),  # type: ignore[arg-type]
    )
    processor.process_audio_window(window(0, channels=2))
    output = processor.process_context_document(
        [{"class": "person", "conf": 0.95, "box": [640, 150, 40, 100], "t": BASE_TIME.timestamp()}]
    )
    associations = [item for item in output.interactions if item.predicate_id == "AUDIO_VISUAL_ASSOCIATION"]
    assert len(associations) == 1
    assert associations[0].bindings["person"] == scoped_track_identity("camera_a", "context", 30)
