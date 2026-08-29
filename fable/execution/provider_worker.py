"""Long-running same-node provider workers.

A worker adapts one recovered FABLE provider implementation to the typed
``StreamBus``.  Provider implementations remain ordinary Python classes; they
do not need to inherit from a runtime framework base class.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Mapping, Sequence

from fable.providers.audio_classification import AudioEventClassifierProvider, YamNetBackend
from fable.providers.audio_localization import AudioVisualAssociationProvider, GccPhatAudioLocalizerProvider
from fable.providers.data_models import (
    AudioLocalization,
    AudioWindow,
    DiarizedSpeechWindow,
    DetectionFrame,
    EmbeddingVector,
    ImageCrop,
    MultichannelAudioWindow,
    SpeakerEmbedding,
    SpeechSegment,
    TrackFrame,
    VideoFrame,
    VisualBearing,
)
from fable.providers.identity import CrossSensorIdentityAssociationProvider, IdentityAssociation
from fable.providers.object_detection import (
    PackageDetectorProvider,
    YoloFullContext960Provider,
    YoloVehicleBalanced960Provider,
    YoloVehicleFast640Provider,
)
from fable.providers.predicate_implementations import (
    BoardsPersonVehicleProvider,
    ConversationAVProvider,
    DisembarksPersonVehicleProvider,
    EntersBasicProvider,
    ExitsBasicProvider,
    FollowsLocalGeometryProvider,
    MovingBasicProvider,
    NearGeometryProvider,
    PresentBasicProvider,
    TransferCustodyProvider,
)
from fable.providers.predicate_result import PredicateMatch
from fable.providers.provider_capabilities import load_provider_capabilities
from fable.providers.speech_processing import (
    KeywordOrASRProvider,
    SpeakerDiarizationProvider,
    SpeakerEmbeddingProvider,
    VoiceActivityDetectorProvider,
)
from fable.providers.tracking import MultiObjectTrackerProvider
from fable.providers.visual_features import (
    CameraProjectionProvider,
    OpenClipVisualDescriptorProvider,
    PersonReIDDescriptorProvider,
    TrackCropExtractorProvider,
    VehicleReIDDescriptorProvider,
)

from .plan_reconciler import ProviderInstanceKey, ProviderInstanceSpec
from .result_transport import ResultTransport
from .stream_bus import StreamBus, StreamKey, Subscription


ProviderFactory = Callable[[str], object]


class DefaultProviderFactory:
    """Create the recovered provider implementation for a provider ID.

    Heavy model imports remain lazy inside the provider classes.  Deployments
    that need custom checkpoints/backends can inject their own factory mapping
    into ``DataflowProviderRuntime`` instead of changing the semantic catalog.
    """

    def __call__(self, provider_id: str) -> object:
        constructors: dict[str, Callable[[], object]] = {
            "yolo_vehicle_fast_640": YoloVehicleFast640Provider,
            "yolo_vehicle_balanced_960": YoloVehicleBalanced960Provider,
            "yolo_full_context_960": YoloFullContext960Provider,
            "package_detector": PackageDetectorProvider,
            "multi_object_tracker": MultiObjectTrackerProvider,
            "present_basic": PresentBasicProvider,
            "enters_basic": EntersBasicProvider,
            "exits_basic": ExitsBasicProvider,
            "moving_basic": MovingBasicProvider,
            "near_geometry": NearGeometryProvider,
            "follows_local_geometry": FollowsLocalGeometryProvider,
            "boards_person_vehicle": BoardsPersonVehicleProvider,
            "disembarks_person_vehicle": DisembarksPersonVehicleProvider,
            "transfer_custody": TransferCustodyProvider,
            "conversation_av": ConversationAVProvider,
            "audio_event_classifier": lambda: AudioEventClassifierProvider(YamNetBackend()),
            "camera_projection": CameraProjectionProvider,
            "track_crop_extractor": TrackCropExtractorProvider,
            "vehicle_reid_descriptor": VehicleReIDDescriptorProvider,
            "person_reid_descriptor": PersonReIDDescriptorProvider,
            "openclip_visual_descriptor": OpenClipVisualDescriptorProvider,
            "cross_sensor_identity_association": CrossSensorIdentityAssociationProvider,
            "voice_activity_detector": VoiceActivityDetectorProvider,
            "speaker_embedding_provider": SpeakerEmbeddingProvider,
            "speaker_diarization_provider": SpeakerDiarizationProvider,
            "keyword_or_asr_provider": KeywordOrASRProvider,
            "gcc_phat_audio_localizer": GccPhatAudioLocalizerProvider,
            "audio_visual_association": AudioVisualAssociationProvider,
        }
        try:
            return constructors[provider_id]()
        except KeyError as exc:
            raise RuntimeError(
                f"provider {provider_id!r} has no live worker adapter; "
                "inject a deployment provider factory for it"
            ) from exc


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    key: ProviderInstanceKey
    ready: bool
    last_error: str | None = None


class ProviderWorker:
    """Connect one provider implementation to typed input/output streams."""

    def __init__(
        self,
        spec: ProviderInstanceSpec,
        provider: object,
        bus: StreamBus,
        result_transport: ResultTransport,
        *,
        provider_catalog: Mapping[str, object] | None = None,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self.bus = bus
        self.results = result_transport
        raw_catalog = provider_catalog if provider_catalog is not None else load_provider_capabilities()
        catalog = raw_catalog.get("providers", raw_catalog)  # type: ignore[union-attr]
        try:
            row = catalog[spec.key.provider_id]  # type: ignore[index]
        except KeyError as exc:
            raise RuntimeError(f"unknown provider {spec.key.provider_id!r}") from exc
        self.input_types = tuple(str(value) for value in row.get("inputs", ()))  # type: ignore[union-attr]
        outputs = tuple(str(value) for value in row.get("outputs", ()))  # type: ignore[union-attr]
        if spec.output_type:
            self.output_type = spec.output_type
        elif len(outputs) == 1:
            self.output_type = outputs[0]
        else:
            raise RuntimeError(f"provider {spec.key.provider_id!r} has ambiguous outputs")
        self._subscriptions: list[Subscription] = []
        self._latest: dict[str, dict[StreamKey, object]] = defaultdict(dict)
        self._last_signature: tuple | None = None
        self.ready = False
        self.last_error: str | None = None

    def start(self) -> None:
        if self.ready:
            return
        try:
            self._warmup()
            for data_type in sorted(set(self.input_types)):
                self._subscriptions.append(
                    self.bus.subscribe(
                        data_type,
                        self._on_input,
                        source_ids=self.spec.key.source_ids,
                    )
                )
            self.ready = True
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
            self.stop()
            raise

    def stop(self) -> None:
        for subscription in self._subscriptions:
            self.bus.unsubscribe(subscription)
        self._subscriptions.clear()
        self.ready = False
        stop = getattr(self.provider, "stop", None)
        close = getattr(self.provider, "close", None)
        if callable(stop):
            stop()
        elif callable(close):
            close()

    def status(self) -> WorkerStatus:
        return WorkerStatus(self.spec.key, self.ready, self.last_error)

    def _warmup(self) -> None:
        warmup = getattr(self.provider, "warmup", None)
        if callable(warmup):
            warmup()
            return
        backend = getattr(self.provider, "backend", None)
        backend_warmup = getattr(backend, "warmup", None)
        if callable(backend_warmup):
            backend_warmup()
            return
        # The recovered YOLO/YAMNet adapters intentionally load lazily.  A live
        # START acknowledgement should mean the model is usable, so force that
        # lazy load here when the implementation exposes it.
        loader = getattr(self.provider, "_load", None)
        if callable(loader):
            loader()
            return
        backend_loader = getattr(backend, "_load", None)
        if callable(backend_loader):
            backend_loader()

    def _on_input(self, key: StreamKey, value: object) -> None:
        if not self.ready:
            return
        self._latest[key.data_type][key] = value
        inputs = self._ready_inputs()
        if inputs is None:
            return
        signature = tuple((kind, tuple((stream, id(value)) for stream, value in rows)) for kind, rows in inputs)
        if signature == self._last_signature:
            return
        self._last_signature = signature
        try:
            output = self._process(inputs)
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
            raise
        self._publish_output(output)

    def _ready_inputs(self) -> tuple[tuple[str, tuple[tuple[StreamKey, object], ...]], ...] | None:
        if not self.input_types:
            return ()
        counts = Counter(self.input_types)
        assembled = []
        for data_type in self.input_types:
            if any(kind == data_type for kind, _ in assembled):
                continue
            rows = tuple(sorted(self._latest.get(data_type, {}).items(), key=lambda row: row[0]))
            required = counts[data_type]
            if len(rows) < required:
                return None
            if required == 1:
                rows = (rows[-1],)
            else:
                # Duplicate typed inputs (currently cross-sensor identity
                # association) must come from distinct source streams.
                distinct = []
                seen_sources: set[tuple[str, ...]] = set()
                for row in reversed(rows):
                    if row[0].source_ids in seen_sources:
                        continue
                    seen_sources.add(row[0].source_ids)
                    distinct.append(row)
                    if len(distinct) == required:
                        break
                if len(distinct) < required:
                    return None
                rows = tuple(reversed(distinct))
            assembled.append((data_type, rows))
        return tuple(assembled)

    def _process(self, inputs):
        provider_id = self.spec.key.provider_id
        values: dict[str, list[object]] = {
            kind: [value for _, value in rows] for kind, rows in inputs
        }
        first = lambda kind: values[kind][-1]

        if provider_id.startswith("yolo_") or provider_id == "package_detector":
            frame = first("video_frame")
            if not isinstance(frame, VideoFrame):
                raise TypeError("object detector expects VideoFrame")
            return self.provider.detect(
                frame.image,
                source_id=frame.source_id,
                event_time=frame.event_time,
                frame_id=frame.frame_id,
            )
        if provider_id == "multi_object_tracker":
            return self.provider.update(first("detections"))
        if provider_id in {"present_basic", "enters_basic", "exits_basic", "moving_basic", "boards_person_vehicle", "disembarks_person_vehicle", "transfer_custody"}:
            return self.provider.update(first("tracks"))
        if provider_id in {"near_geometry", "follows_local_geometry"}:
            return self.provider.evaluate(first("tracks"))
        if provider_id == "audio_event_classifier":
            return self.provider.classify(first("audio_window"))
        if provider_id == "track_crop_extractor":
            frame = first("video_frame")
            tracks = first("tracks")
            if not isinstance(frame, VideoFrame):
                raise TypeError("track crop extraction expects VideoFrame")
            return self.provider.extract(frame.image, tracks)
        if provider_id in {"vehicle_reid_descriptor", "person_reid_descriptor", "openclip_visual_descriptor"}:
            return self.provider.describe(first("track_crops"))
        if provider_id == "cross_sensor_identity_association":
            rows = values["identity_embeddings"]
            return self.provider.associate_records(rows[0], rows[1])
        if provider_id == "voice_activity_detector":
            return self.provider.detect(first("audio_window"))
        if provider_id == "speaker_embedding_provider":
            return self.provider.embed(first("audio_window"), first("speech_segments"))
        if provider_id == "speaker_diarization_provider":
            embeddings = first("speaker_embeddings")
            source_id = _source_id_from_embeddings(embeddings, self.spec.key.source_ids)
            return self.provider.diarize(source_id, embeddings)
        if provider_id == "keyword_or_asr_provider":
            # The recovered implementation consumes diarized speech.  The old
            # capability record called this input speech_segments; accept either
            # when an injected plan/provider supplies it.
            audio = first("audio_window")
            diarized = values.get("diarized_speech", values.get("speech_segments", []))[-1]
            return self.provider.transcribe(audio, diarized)
        if provider_id == "conversation_av":
            return self.provider.evaluate(first("tracks"), first("diarized_speech"))
        if provider_id == "gcc_phat_audio_localizer":
            return self.provider.localize(first("multichannel_audio"))
        if provider_id == "camera_projection":
            raise RuntimeError("camera_projection requires deployment homography configuration")
        if provider_id == "audio_visual_association":
            raise RuntimeError("audio_visual_association requires visual bearing inputs not yet exposed by the minimal live worker")
        if provider_id == "follows_cross_sensor":
            raise RuntimeError("cross-sensor follows requires projected/canonical tracks and is not a same-node live worker yet")
        raise RuntimeError(f"no live worker process adapter for provider {provider_id!r}")

    def _publish_output(self, output: object) -> None:
        if output is None:
            return
        if output == () and (
            self.output_type.startswith("predicate_match:")
            or self.output_type == "identity_associations"
        ):
            return
        values: Sequence[object]
        # Provider outputs that are logical collections (detections/tracks,
        # embedding batches, speech windows) remain one stream value.  Terminal
        # result tuples are emitted one record at a time to the controller.
        if isinstance(output, tuple) and output and all(
            isinstance(value, (PredicateMatch, IdentityAssociation)) for value in output
        ):
            values = output
        else:
            values = (output,)

        for value in values:
            if isinstance(value, PredicateMatch):
                self.results.send_predicate_match(value)
                self.bus.publish(StreamKey(self.output_type, self.spec.key.source_ids), value)
            elif isinstance(value, IdentityAssociation):
                self.results.send_identity_association(value)
                self.bus.publish(StreamKey(self.output_type, self.spec.key.source_ids), value)
            else:
                self.bus.publish(StreamKey(self.output_type, self.spec.key.source_ids), value)


def _source_id_from_embeddings(embeddings: object, fallback: tuple[str, ...]) -> str:
    if isinstance(embeddings, Sequence) and embeddings:
        value = embeddings[0]
        source_id = getattr(value, "source_id", None)
        if isinstance(source_id, str) and source_id:
            return source_id
    return fallback[0] if fallback else "unknown"
