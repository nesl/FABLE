#!/usr/bin/env python3
"""IoBT-Max bridge for the FABLE Phase-8 multimodal provider stack.

The bridge deliberately subscribes to ReSpeaker samples over the existing local
IPC path so raw audio does not need to be republished over MQTT.  YOLO context
continues to arrive through the replay stack's existing MQTT topic.
"""

from __future__ import annotations

import csv
import json
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append("/lib/iobtmax")

from iobt_max_service import iobt_max_service, state  # type: ignore

from providers.multimodal.audio import (
    AudioEventClassifier,
    SpectralRuleAudioBackend,
    YamNetBackend,
)
from providers.multimodal.service import (
    AudioProcessingOutput,
    ContextProcessingOutput,
    MultimodalReplayProcessor,
    MultimodalServiceConfig,
    _geometry_from_json,
)
from providers.vehicle.tracker import RoboflowTrackerAdapter


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_yamnet_classifier() -> AudioEventClassifier:
    """Build an explicitly configured YAMNet backend.

    A local/volume-mounted class-map file is required.  This avoids hidden
    network downloads and makes the model/version used by experiments visible.
    """

    handle = os.environ.get("FABLE_YAMNET_MODEL_HANDLE", "").strip()
    class_map = os.environ.get("FABLE_YAMNET_CLASS_MAP", "").strip()
    if not handle or not class_map:
        raise RuntimeError(
            "FABLE_AUDIO_BACKEND=yamnet requires FABLE_YAMNET_MODEL_HANDLE and "
            "FABLE_YAMNET_CLASS_MAP"
        )
    with Path(class_map).open("r", encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    labels = tuple(
        str(row.get("display_name") or row.get("label") or "").strip()
        for row in rows
    )
    if not labels or any(not item for item in labels):
        raise RuntimeError("YAMNet class map must provide display_name or label for every row")
    backend = YamNetBackend(
        model_handle=handle,
        class_names=labels,
        model_version=os.environ.get("FABLE_YAMNET_MODEL_VERSION", "configured"),
    )
    return AudioEventClassifier(backend)


def _build_classifier() -> tuple[AudioEventClassifier, str, bool]:
    backend_name = os.environ.get("FABLE_AUDIO_BACKEND", "spectral-rule").strip().lower()
    if backend_name == "yamnet":
        return _load_yamnet_classifier(), "yamnet", True
    if backend_name not in {"spectral-rule", "spectral", "baseline"}:
        raise RuntimeError(f"unsupported FABLE_AUDIO_BACKEND={backend_name!r}")
    # This baseline permits end-to-end deployment and failure testing, but its
    # output must not be treated as evaluation-quality gunshot/alarm labels.
    return AudioEventClassifier(SpectralRuleAudioBackend()), "spectral-rule-baseline", False


class fable_multimodal_stack(iobt_max_service):
    def __init__(self) -> None:
        self.name = "fable_multimodal"
        super().__init__(self.name)
        self.source_id = os.environ.get("SOURCE_ID", self.hostname)
        self.yolo_topic = os.environ.get(
            "YOLO_TOPIC", f"/{self.source_id}/analytics/yolo/bbox"
        )
        self.audio_event_topic = os.environ.get(
            "AUDIO_EVENT_TOPIC", f"/{self.source_id}/fable/audio/events"
        )
        self.localization_topic = os.environ.get(
            "AUDIO_LOCALIZATION_TOPIC", f"/{self.source_id}/fable/audio/localizations"
        )
        self.speech_turn_topic = os.environ.get(
            "SPEECH_TURN_TOPIC", f"/{self.source_id}/fable/audio/speaker_turns"
        )
        self.context_track_topic = os.environ.get(
            "CONTEXT_TRACK_TOPIC", f"/{self.source_id}/fable/context/tracks"
        )
        self.interaction_topic = os.environ.get(
            "INTERACTION_TOPIC", f"/{self.source_id}/fable/interactions/predicates"
        )
        self.custody_topic = os.environ.get(
            "CUSTODY_TOPIC", f"/{self.source_id}/fable/interactions/custody"
        )
        self.ready_topic = os.environ.get(
            "READINESS_TOPIC", f"/readiness/{self.source_id}/fable_multimodal"
        )
        self.sample_rate_hz = int(os.environ.get("AUDIO_SAMPLE_RATE_HZ", "16000"))
        self.audio_channel_indices = tuple(
            int(item.strip())
            for item in os.environ.get("AUDIO_CHANNEL_INDICES", "1,2,3,4").split(",")
            if item.strip()
        )
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue(
            maxsize=int(os.environ.get("FABLE_MULTIMODAL_QUEUE_SIZE", "512"))
        )
        self.local_subscription_ready = False
        self.frames_audio = 0
        self.frames_context = 0
        self.events_emitted = 0
        self.interactions_emitted = 0
        self.last_error: str | None = None
        self.start_wall_time = time.time()

        classifier, backend_label, evaluation_ready = _build_classifier()
        self.audio_backend_label = backend_label
        self.audio_backend_evaluation_ready = evaluation_ready
        config = MultimodalServiceConfig(
            source_id=self.source_id,
            raw_audio_topic="local://respeaker",
            yolo_topic=self.yolo_topic,
            audio_event_topic=self.audio_event_topic,
            localization_topic=self.localization_topic,
            speech_turn_topic=self.speech_turn_topic,
            context_track_topic=self.context_track_topic,
            interaction_topic=self.interaction_topic,
            custody_topic=self.custody_topic,
            readiness_topic=self.ready_topic,
            sample_rate_hz=self.sample_rate_hz,
            audio_channel_indices=self.audio_channel_indices,
            evidence_horizon_seconds=float(
                os.environ.get("FABLE_EVIDENCE_HORIZON_SECONDS", "30")
            ),
            visual_image_width_px=float(os.environ.get("FABLE_VISUAL_IMAGE_WIDTH_PX", "1280")),
            camera_horizontal_fov_deg=float(os.environ.get("FABLE_CAMERA_HORIZONTAL_FOV_DEG", "90")),
            visual_bearing_offset_deg=float(os.environ.get("FABLE_VISUAL_BEARING_OFFSET_DEG", "0")),
            visual_zone_id=os.environ.get("FABLE_VISUAL_ZONE_ID") or None,
            association_time_tolerance_seconds=float(
                os.environ.get("FABLE_AV_TIME_TOLERANCE_SECONDS", "0.5")
            ),
        )
        self.processor = MultimodalReplayProcessor(
            config=config,
            context_tracker=RoboflowTrackerAdapter(
                algorithm=os.environ.get("TRACKER_ALGORITHM", "bytetrack")
            ),
            audio_classifier=classifier,
            localizer=_geometry_from_json(os.environ.get("FABLE_AUDIO_GEOMETRY")),
        )

        self.subscribe("local", "respeaker", self._on_audio)
        self.subscribe("net", self.yolo_topic, self._on_yolo)
        self._publish_ready("startup_subscriptions_requested")

    def on_local_subscription_ready(self, topic: str, addr: str) -> None:
        if topic == "respeaker":
            self.local_subscription_ready = True
            self._publish_ready("local_respeaker_subscription_ready")

    def _enqueue(self, kind: str, payload: Any) -> None:
        try:
            self.queue.put_nowait((kind, payload))
        except queue.Full:
            self.last_error = f"queue_full_dropped_{kind}"

    def _on_audio(self, document: dict[str, Any]) -> None:
        self._enqueue("audio", document)

    def _on_yolo(self, topic: str, body: str) -> None:
        self._enqueue("context", body)

    def _publish_ready(self, reason: str) -> None:
        ready = bool(self.local_subscription_ready and self.last_error is None)
        payload = {
            "ready": ready,
            "service": "fable_multimodal",
            "source_id": self.source_id,
            "reason": reason,
            "audio_backend": self.audio_backend_label,
            "evaluation_ready_audio_model": self.audio_backend_evaluation_ready,
            "raw_audio_transport": "local_ipc",
            "expected_local_ipc": "/tmp/respeaker.ipc",
            "local_subscription_ready": self.local_subscription_ready,
            "frames_audio": self.frames_audio,
            "frames_context": self.frames_context,
            "events_emitted": self.events_emitted,
            "interactions_emitted": self.interactions_emitted,
            "queue_size": self.queue.qsize(),
            "last_error": self.last_error,
            "uptime_seconds": round(time.time() - self.start_wall_time, 3),
            "t": time.time(),
        }
        self.publish("net", self.ready_topic, json.dumps(payload))
        self.publish_readiness("fable_multimodal", **payload)

    def _publish_audio(self, output: AudioProcessingOutput) -> None:
        for event in output.events:
            self.publish("net", self.audio_event_topic, event.model_dump_json())
            self.events_emitted += 1
        for localization in output.localizations:
            self.publish("net", self.localization_topic, localization.model_dump_json())
        if output.turns is not None:
            self.publish("net", self.speech_turn_topic, output.turns.model_dump_json())

    def _publish_context(self, output: ContextProcessingOutput) -> None:
        self.publish("net", self.context_track_topic, output.tracks.model_dump_json())
        for item in output.interactions:
            self.publish("net", self.interaction_topic, item.model_dump_json())
            self.interactions_emitted += 1
        self.publish("net", self.custody_topic, output.custody.model_dump_json())

    def service_initialize(self) -> None:
        self._publish_ready("service_initialized")

    def service_stop(self) -> None:
        self._publish_ready("service_stopping")

    def service_initialize_collect(self) -> None:
        pass

    def service_stop_collect(self) -> None:
        pass

    def service_step(self) -> bool:
        last_status = 0.0
        while self.state != state.quit:
            try:
                kind, payload = self.queue.get(timeout=0.25)
            except queue.Empty:
                if time.time() - last_status >= 2.0:
                    self._publish_ready("idle")
                    last_status = time.time()
                continue
            try:
                if kind == "audio":
                    output = self.processor.process_audio_document(payload)
                    self.frames_audio += 1
                    self._publish_audio(output)
                else:
                    document = json.loads(payload) if isinstance(payload, str) else payload
                    output = self.processor.process_context_document(document)
                    self.frames_context += 1
                    self._publish_context(output)
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{kind}_processing_failed: {exc}"
                print(f"[FABLE multimodal] {self.last_error}", flush=True)
            if time.time() - last_status >= 2.0:
                self._publish_ready("active")
                last_status = time.time()
            time.sleep(0)
        return True


def main() -> None:
    node = fable_multimodal_stack()
    node.start()


if __name__ == "__main__":
    main()
