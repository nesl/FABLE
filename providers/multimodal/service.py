"""Phase-8 multimodal processing core and optional MQTT bridge."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover - optional for unit tests
    mqtt = None  # type: ignore[assignment]

from fable.common.time import EventTimeInterval

from providers.vehicle.detector import LegacyReplayYoloAdapter
from providers.vehicle.models import TrackSet
from providers.vehicle.tracker import RoboflowTrackerAdapter

from .audiovisual import AudioVisualAssociator
from .audio import (
    AudioEventClassifier,
    SpectralRuleAudioBackend,
    audio_window_from_replay_payload,
)
from .conversation import (
    ConversationEvaluator,
    EnergyVoiceActivityDetector,
    OnlineSpeakerDiarizer,
    SpectralSpeakerEmbeddingProvider,
)
from .errors import InvalidAudioInput, OptionalDependencyError
from .localization import GccPhatAudioLocalizer
from .models import (
    AudioEventObservation,
    AudioLocalization,
    AudioWindow,
    CustodyState,
    InteractionPredicateObservation,
    MicrophoneArrayGeometry,
    SpeakerTurnSet,
    VisualBearingCandidate,
)
from .package_transfer import TransferCustodyReasoner
from .person_vehicle import PersonVehicleRelationEvaluator

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultimodalServiceConfig:
    source_id: str
    raw_audio_topic: str
    yolo_topic: str
    audio_event_topic: str
    localization_topic: str
    speech_turn_topic: str
    context_track_topic: str
    interaction_topic: str
    custody_topic: str
    readiness_topic: str
    sample_rate_hz: int = 16_000
    audio_channel_indices: tuple[int, ...] = (1, 2, 3, 4)
    evidence_horizon_seconds: float = 30.0
    visual_image_width_px: float = 1280.0
    camera_horizontal_fov_deg: float = 90.0
    visual_bearing_offset_deg: float = 0.0
    visual_zone_id: str | None = None
    association_time_tolerance_seconds: float = 0.5


@dataclass(frozen=True)
class AudioProcessingOutput:
    events: tuple[AudioEventObservation, ...]
    localizations: tuple[AudioLocalization, ...]
    turns: SpeakerTurnSet | None


@dataclass(frozen=True)
class ContextProcessingOutput:
    tracks: TrackSet
    interactions: tuple[InteractionPredicateObservation, ...]
    custody: CustodyState


class MultimodalReplayProcessor:
    """Pure core shared by the MQTT and local-IPC replay services."""

    def __init__(
        self,
        *,
        config: MultimodalServiceConfig,
        context_tracker: RoboflowTrackerAdapter,
        audio_classifier: AudioEventClassifier | None = None,
        localizer: GccPhatAudioLocalizer | None = None,
        vad: Any | None = None,
        speaker_embedder: Any | None = None,
        diarizer: OnlineSpeakerDiarizer | None = None,
        person_vehicle: PersonVehicleRelationEvaluator | None = None,
        custody: TransferCustodyReasoner | None = None,
        conversation: ConversationEvaluator | None = None,
        audiovisual: AudioVisualAssociator | None = None,
        context_adapter: LegacyReplayYoloAdapter | None = None,
    ) -> None:
        self.config = config
        self.context_tracker = context_tracker
        self.audio_classifier = audio_classifier or AudioEventClassifier(
            SpectralRuleAudioBackend()
        )
        self.localizer = localizer
        self.vad = vad or EnergyVoiceActivityDetector()
        self.speaker_embedder = speaker_embedder or SpectralSpeakerEmbeddingProvider()
        self.diarizer = diarizer or OnlineSpeakerDiarizer()
        self.person_vehicle = person_vehicle or PersonVehicleRelationEvaluator()
        self.custody = custody or TransferCustodyReasoner()
        self.conversation = conversation or ConversationEvaluator()
        self.audiovisual = audiovisual or AudioVisualAssociator()
        self.context_adapter = context_adapter or LegacyReplayYoloAdapter(
            detector_id="yolo_full_context_960",
            detector_version="legacy-replay-yolov8",
            class_allowlist=(),
        )
        self._audio_sequence = 0
        self._context_sequence = 0
        self._turns: list[Any] = []
        self._track_history: list[TrackSet] = []
        self._localized_audio_events: list[tuple[AudioEventObservation, AudioLocalization]] = []

    def process_audio_document(self, document: Any) -> AudioProcessingOutput:
        self._audio_sequence += 1
        window = audio_window_from_replay_payload(
            document,
            source_id=self.config.source_id,
            sample_rate_hz=self.config.sample_rate_hz,
            channel_indices=self.config.audio_channel_indices,
            source_sequence=self._audio_sequence,
        )
        return self.process_audio_window(window)

    def process_audio_window(self, window: AudioWindow) -> AudioProcessingOutput:
        events = list(self.audio_classifier.classify(window))
        localizations: list[AudioLocalization] = []
        if self.localizer is not None:
            try:
                localization = self.localizer.localize(window)
            except InvalidAudioInput as exc:
                # Localization is an optional enrichment. A geometrically
                # ambiguous window must not suppress a valid classifier event
                # needed by source-scoped AUDIO_EVENT demands.
                LOGGER.debug("audio localization unavailable: %s", exc)
            else:
                localizations.append(localization)
                events = [
                    event.model_copy(update={"localized_zone_id": localization.zone_id})
                    for event in events
                ]
                self._localized_audio_events.extend((event, localization) for event in events)
                self._trim_audio_context(window.event_time_interval.end)
        speech = self.vad.detect(window)
        if speech and self.speaker_embedder is not None:
            speech = self.speaker_embedder.attach(window, speech)
        turn_set = None
        if speech:
            current = self.diarizer.diarize(speech)
            self._turns.extend(current.turns)
            self._trim_turns(window.event_time_interval.end)
            turn_set = self._aggregate_turns(current)
        return AudioProcessingOutput(
            events=tuple(events),
            localizations=tuple(localizations),
            turns=turn_set,
        )

    def process_context_document(self, document: Any) -> ContextProcessingOutput:
        self._context_sequence += 1
        detections = self.context_adapter.parse(
            document,
            source_id=self.config.source_id,
            frame_id=f"{self.config.source_id}:context:{self._context_sequence}",
            source_sequence=self._context_sequence,
        )
        tracks = self.context_tracker.update(detections)
        interactions = list(self.person_vehicle.update(tracks))
        visual_candidates = self._visual_candidates(tracks)
        for event, localization in tuple(self._localized_audio_events):
            observation = self.audiovisual.best_predicate_observation(
                event, localization, visual_candidates
            )
            if observation is not None:
                interactions.append(observation)
        custody_state, transfer = self.custody.update(tracks)
        interactions.extend(transfer)
        self._track_history.append(tracks)
        self._trim_tracks(tracks.event_time)
        turns = self.latest_turn_set()
        if turns is not None:
            conversation = self.conversation.evaluate(self._track_history, turns)
            if conversation is not None:
                interactions.append(conversation)
        return ContextProcessingOutput(
            tracks=tracks,
            interactions=tuple(interactions),
            custody=custody_state,
        )

    def _visual_candidates(self, tracks: TrackSet) -> tuple[VisualBearingCandidate, ...]:
        width = max(float(self.config.visual_image_width_px), 1.0)
        fov = float(self.config.camera_horizontal_fov_deg)
        tolerance = timedelta(seconds=self.config.association_time_tolerance_seconds)
        interval = EventTimeInterval(
            start=tracks.event_time - tolerance,
            end=tracks.event_time + tolerance,
        )
        candidates = []
        for track in tracks.tracks:
            normalized_x = max(0.0, min(1.0, track.bbox.center[0] / width))
            azimuth = self.config.visual_bearing_offset_deg + (normalized_x - 0.5) * fov
            azimuth = ((azimuth + 180.0) % 360.0) - 180.0
            candidates.append(
                VisualBearingCandidate(
                    local_entity_id=track.scoped_track_id,
                    entity_type=track.class_name,
                    source_id=tracks.source_id,
                    event_time_interval=interval,
                    azimuth_deg=azimuth,
                    zone_id=self.config.visual_zone_id,
                    confidence=track.confidence,
                )
            )
        return tuple(candidates)

    def _trim_audio_context(self, now: Any) -> None:
        cutoff = now - timedelta(seconds=self.config.evidence_horizon_seconds)
        self._localized_audio_events = [
            item for item in self._localized_audio_events
            if item[0].event_time_interval.end >= cutoff
        ]

    def latest_turn_set(self) -> SpeakerTurnSet | None:
        if not self._turns:
            return None
        ordered = sorted(self._turns, key=lambda item: item.event_time_interval.start)
        return SpeakerTurnSet(
            source_id=self.config.source_id,
            event_time_interval=type(ordered[0].event_time_interval)(
                start=ordered[0].event_time_interval.start,
                end=max(item.event_time_interval.end for item in ordered),
            ),
            turns=tuple(ordered),
            speaker_count=len({item.speaker_id for item in ordered}),
            diarization_model_id=self.diarizer.model_id,
            diarization_model_version=self.diarizer.model_version,
        )

    def _aggregate_turns(self, current: SpeakerTurnSet) -> SpeakerTurnSet:
        return self.latest_turn_set() or current

    def _trim_turns(self, now: Any) -> None:
        cutoff = now - timedelta(seconds=self.config.evidence_horizon_seconds)
        self._turns = [
            item for item in self._turns if item.event_time_interval.end >= cutoff
        ]

    def _trim_tracks(self, now: Any) -> None:
        cutoff = now - timedelta(seconds=self.config.evidence_horizon_seconds)
        self._track_history = [item for item in self._track_history if item.event_time >= cutoff]


class MultimodalMqttService:
    """Network bridge for deployments that publish raw audio and YOLO JSON to MQTT."""

    def __init__(
        self,
        *,
        config: MultimodalServiceConfig,
        processor: MultimodalReplayProcessor,
        host: str,
        port: int,
        client: Any | None = None,
    ) -> None:
        if client is None and mqtt is None:
            raise OptionalDependencyError("paho-mqtt is required for the Phase-8 MQTT service")
        self.config = config
        self.processor = processor
        self.host = host
        self.port = port
        self.client = client or mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"fable-multimodal-{config.source_id}",
            clean_session=False,
        )
        self._stop = threading.Event()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client: Any, userdata: Any, flags: Any, reason_code: Any, properties: Any) -> None:
        if int(getattr(reason_code, "value", reason_code)) != 0:
            LOGGER.error("MQTT connection failed: %s", reason_code)
            return
        client.subscribe(self.config.raw_audio_topic, qos=0)
        client.subscribe(self.config.yolo_topic, qos=0)
        client.publish(
            self.config.readiness_topic,
            json.dumps({"ready": True, "source_id": self.config.source_id}),
            qos=1,
            retain=True,
        )

    def _on_message(self, client: Any, userdata: Any, message: Any) -> None:
        try:
            document = json.loads(message.payload.decode("utf-8"))
            if message.topic == self.config.raw_audio_topic:
                self._publish_audio(self.processor.process_audio_document(document))
            elif message.topic == self.config.yolo_topic:
                self._publish_context(self.processor.process_context_document(document))
        except Exception:
            LOGGER.exception("multimodal provider failed topic=%s", message.topic)

    def _publish_audio(self, output: AudioProcessingOutput) -> None:
        for event in output.events:
            self.client.publish(self.config.audio_event_topic, event.model_dump_json(), qos=0)
        for localization in output.localizations:
            self.client.publish(
                self.config.localization_topic,
                localization.model_dump_json(),
                qos=0,
            )
        if output.turns is not None:
            self.client.publish(
                self.config.speech_turn_topic,
                output.turns.model_dump_json(),
                qos=0,
            )

    def _publish_context(self, output: ContextProcessingOutput) -> None:
        self.client.publish(
            self.config.context_track_topic,
            output.tracks.model_dump_json(),
            qos=0,
        )
        for observation in output.interactions:
            self.client.publish(
                self.config.interaction_topic,
                observation.model_dump_json(),
                qos=0,
            )
        self.client.publish(
            self.config.custody_topic,
            output.custody.model_dump_json(),
            qos=0,
        )

    def run(self) -> None:
        self.client.connect(self.host, self.port, keepalive=60)
        self.client.loop_start()
        try:
            self._stop.wait()
        finally:
            self.client.publish(
                self.config.readiness_topic,
                json.dumps({"ready": False, "source_id": self.config.source_id}),
                qos=1,
                retain=True,
            )
            self.client.loop_stop()
            self.client.disconnect()

    def stop(self) -> None:
        self._stop.set()


def _geometry_from_json(path: str | None) -> GccPhatAudioLocalizer | None:
    if not path:
        return None
    document = json.loads(open(path, "r", encoding="utf-8").read())
    from .models import BearingZone

    geometry = MicrophoneArrayGeometry.model_validate(document["microphone_array"])
    zones = tuple(BearingZone.model_validate(item) for item in document.get("bearing_zones", ()))
    return GccPhatAudioLocalizer(geometry, bearing_zones=zones)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mqtt-host", default=os.getenv("MQTT_HOST", "mqtt"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--source-id", default=os.getenv("SOURCE_ID", "orin11"))
    parser.add_argument("--raw-audio-topic", default=os.getenv("RAW_AUDIO_TOPIC"))
    parser.add_argument("--yolo-topic", default=os.getenv("YOLO_TOPIC"))
    parser.add_argument("--geometry", default=os.getenv("FABLE_AUDIO_GEOMETRY"))
    parser.add_argument("--tracker", default=os.getenv("TRACKER_ALGORITHM", "bytetrack"))
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    source_id = args.source_id
    config = MultimodalServiceConfig(
        source_id=source_id,
        raw_audio_topic=args.raw_audio_topic or f"/{source_id}/fable/audio/raw",
        yolo_topic=args.yolo_topic or f"/{source_id}/analytics/yolo/bbox",
        audio_event_topic=f"/{source_id}/fable/audio/events",
        localization_topic=f"/{source_id}/fable/audio/localizations",
        speech_turn_topic=f"/{source_id}/fable/audio/speaker_turns",
        context_track_topic=f"/{source_id}/fable/context/tracks",
        interaction_topic=f"/{source_id}/fable/interactions/predicates",
        custody_topic=f"/{source_id}/fable/interactions/custody",
        readiness_topic=f"/readiness/{source_id}/fable_multimodal",
    )
    processor = MultimodalReplayProcessor(
        config=config,
        context_tracker=RoboflowTrackerAdapter(algorithm=args.tracker),
        localizer=_geometry_from_json(args.geometry),
    )
    service = MultimodalMqttService(
        config=config,
        processor=processor,
        host=args.mqtt_host,
        port=args.mqtt_port,
    )
    signal.signal(signal.SIGTERM, lambda *_: service.stop())
    signal.signal(signal.SIGINT, lambda *_: service.stop())
    service.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
