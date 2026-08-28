"""MQTT worker for site-local ReID over transferred, bounded image crops.

The worker accepts only ``bounded_reid_crop_set.v1`` payloads. It never accepts raw
frames, filenames, shell commands, or arbitrary artifact URIs. This keeps the
network alternative distinct from moving a camera stream off its sensor.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import signal
import threading
import uuid
from dataclasses import dataclass
from typing import Any

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None  # type: ignore[assignment]

from fable.common.time import EventTimeInterval

from .descriptors import FastReidEntityDescriptor
from .models import DescriptorSet

LOGGER = logging.getLogger(__name__)


def _decode_jpeg(raw: bytes) -> Any:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - container dependency
        raise RuntimeError("site ReID requires OpenCV and NumPy") from exc
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("crop is not a decodable JPEG")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@dataclass(frozen=True)
class SiteDescriptorConfig:
    input_topic: str = "/+/fable/identity/bounded-crops"
    output_topic: str = "/fable/identity/descriptors"
    readiness_topic: str = "/readiness/x86server/fable_reid_descriptor"
    maximum_crops_per_message: int = 2
    maximum_encoded_crop_bytes: int = 1_500_000


class CropDescriptorProcessor:
    def __init__(self, descriptor: Any, config: SiteDescriptorConfig | None = None) -> None:
        self.descriptor = descriptor
        self.config = config or SiteDescriptorConfig()

    def process(self, payload: dict[str, Any]) -> DescriptorSet:
        if payload.get("schema_version") != "bounded_reid_crop_set.v1":
            raise ValueError("site ReID accepts only bounded_reid_crop_set.v1")
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("crop set requires non-empty records")
        if len(records) > min(2, self.config.maximum_crops_per_message):
            raise ValueError("crop set exceeds bounded record count")
        crops = []
        image_by_entity: dict[str, str] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("crop record must be an object")
            entity_id = str(record.get("local_entity_id") or "")
            data_url = str(record.get("image_data_url") or "")
            if not entity_id or not data_url.startswith("data:image/jpeg;base64,"):
                raise ValueError("crop record requires an ID and inline JPEG data URL")
            encoded = data_url.partition(",")[2]
            if len(encoded) > (self.config.maximum_encoded_crop_bytes * 4 // 3 + 8):
                raise ValueError("encoded crop exceeds byte limit")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ValueError("invalid crop base64") from exc
            if len(raw) > self.config.maximum_encoded_crop_bytes:
                raise ValueError("decoded crop exceeds byte limit")
            crops.append((entity_id, _decode_jpeg(raw)))
            image_by_entity[entity_id] = data_url
        encoded = self.descriptor.encode(
            crops,
            source_id=str(payload["source_id"]),
            event_time_interval=EventTimeInterval.model_validate(payload["event_time_interval"]),
        )
        # The embedding alone is enough for local ReID, but an ambiguous exact
        # graph demand may escalate to the bounded VLM comparator. Preserve the
        # typed source crop through this artifact boundary so that escalation
        # has visual evidence. The input has already passed the strict JPEG and
        # size checks above; no path, URI, or arbitrary external payload is
        # introduced here.
        return encoded.model_copy(
            update={
                "records": tuple(
                    record.model_copy(
                        update={
                            "source_crop_data_urls": (
                                image_by_entity[record.local_entity_id],
                            )
                        }
                    )
                    for record in encoded.records
                )
            }
        )


def _descriptor_from_environment() -> FastReidEntityDescriptor:
    return FastReidEntityDescriptor(
        entity_kind="vehicle",
        config_path=os.environ.get(
            "VEHICLE_REID_CONFIG", "/app/reid/fastreid_veri_sbs_r50_ibn.yaml"
        ),
        model_path=os.environ.get("VEHICLE_REID_MODEL_PATH", "/models/reid/vehicle.pth"),
        model_id=os.environ.get("VEHICLE_REID_MODEL_ID", "fastreid:sbs_R50_ibn:vehicle"),
        model_version=os.environ["VEHICLE_REID_MODEL_VERSION"],
        preprocessing_id=os.environ.get(
            "VEHICLE_REID_PREPROCESSING_ID", "fastreid-veri-256x256-rgb"
        ),
        device=os.environ.get("VEHICLE_REID_DEVICE", "cuda:0"),
    )


def main() -> int:
    if mqtt is None:
        raise RuntimeError("paho-mqtt is required")
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    config = SiteDescriptorConfig()
    descriptor = _descriptor_from_environment()
    # FastReID is otherwise loaded lazily on the first crop.  Initialize it
    # before connecting so the worker cannot publish a false-positive ready
    # state when its model, configuration, or Python dependencies are broken.
    descriptor.warmup()
    processor = CropDescriptorProcessor(descriptor, config)
    stopped = threading.Event()
    worker_generation = str(uuid.uuid4())
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="fable-site-reid")

    def publish_ready(client):
        client.publish(
            config.readiness_topic,
            json.dumps({"ready": True, "worker_generation": worker_generation}),
            qos=1,
            retain=True,
        )

    def on_connect(client, _userdata, _flags, reason_code, _properties):
        if reason_code != 0:
            raise RuntimeError(f"MQTT connection failed: {reason_code}")
        # Calling subscribe only queues the request. Do not advertise readiness
        # until the broker acknowledges that the crop subscription is active;
        # otherwise a readiness-triggered crop replay can race ahead of SUBACK.
        client.subscribe(config.input_topic, qos=1)

    def on_subscribe(client, _userdata, _mid, _reason_codes, _properties):
        publish_ready(client)

    def on_message(client, _userdata, message):
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            result = processor.process(payload)
            client.publish(
                f"/{result.source_id}/fable/identity/descriptors",
                result.model_dump_json(),
                qos=1,
            )
        except Exception:
            LOGGER.exception("rejected site ReID crop message")

    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    for event in (signal.SIGINT, signal.SIGTERM):
        signal.signal(event, lambda *_: stopped.set())
    client.connect(os.environ.get("MQTT_HOST", "mqtt"), int(os.environ.get("MQTT_PORT", "1883")))
    client.loop_start()
    stopped.wait()
    client.loop_stop()
    client.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
