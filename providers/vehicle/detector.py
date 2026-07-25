"""YOLO detector variants and replay-payload adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Iterable

from fable.common.time import ensure_utc

from .errors import InvalidProviderInput, OptionalDependencyError
from .models import BoundingBox, Detection, DetectionFrame, Point2D


@dataclass(frozen=True)
class YoloVariant:
    provider_id: str
    model_path: str
    image_size: int
    maximum_rate_hz: float
    class_allowlist: tuple[str, ...]
    confidence_threshold: float
    device: str = "auto"
    model_version: str = "unresolved"

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.maximum_rate_hz <= 0:
            raise ValueError("maximum_rate_hz must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")


DEFAULT_YOLO_VARIANTS: dict[str, YoloVariant] = {
    "yolo_vehicle_fast_640": YoloVariant(
        provider_id="yolo_vehicle_fast_640",
        model_path="yolov8n.pt",
        image_size=640,
        maximum_rate_hz=10.0,
        class_allowlist=("car", "truck", "bus", "motorcycle"),
        confidence_threshold=0.30,
        model_version="yolov8n",
    ),
    "yolo_vehicle_balanced_960": YoloVariant(
        provider_id="yolo_vehicle_balanced_960",
        model_path="yolov8s.pt",
        image_size=960,
        maximum_rate_hz=5.0,
        class_allowlist=("car", "truck", "bus", "motorcycle"),
        confidence_threshold=0.25,
        model_version="yolov8s",
    ),
    "yolo_full_context_960": YoloVariant(
        provider_id="yolo_full_context_960",
        model_path="yolov8s.pt",
        image_size=960,
        maximum_rate_hz=2.0,
        class_allowlist=(),
        confidence_threshold=0.25,
        model_version="yolov8s",
    ),
}


class UltralyticsYoloDetector:
    """Lazy Ultralytics detector that emits the FABLE detection schema.

    The class is intentionally independent from the replay stack. It accepts an
    in-memory image and can therefore be used by live services or retrospective
    segment workers. The model is imported and loaded only when first used.
    """

    def __init__(self, variant: YoloVariant, *, model: Any | None = None) -> None:
        self.variant = variant
        self._model = model

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional heavy dependency
            raise OptionalDependencyError(
                "Ultralytics is required for UltralyticsYoloDetector; install the vehicle-models extra"
            ) from exc
        self._model = YOLO(self.variant.model_path)
        return self._model

    def detect(
        self,
        frame: Any,
        *,
        source_id: str,
        event_time: datetime,
        frame_id: str,
        source_sequence: int | None = None,
    ) -> DetectionFrame:
        model = self._load()
        kwargs: dict[str, Any] = {
            "verbose": False,
            "imgsz": self.variant.image_size,
            "conf": self.variant.confidence_threshold,
        }
        if self.variant.device != "auto":
            kwargs["device"] = self.variant.device
        results = model(frame, **kwargs)
        detections: list[Detection] = []
        allowlist = set(self.variant.class_allowlist)
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = _to_rows(getattr(boxes, "xyxy", ()))
            conf = _to_scalars(getattr(boxes, "conf", ()))
            cls = _to_scalars(getattr(boxes, "cls", ()))
            names = getattr(result, "names", {})
            for index, (coords, probability, class_index) in enumerate(zip(xyxy, conf, cls)):
                class_name = str(names[int(class_index)])
                if allowlist and class_name not in allowlist:
                    continue
                probability = float(probability)
                if probability < self.variant.confidence_threshold:
                    continue
                bbox = BoundingBox(
                    x1=float(coords[0]), y1=float(coords[1]), x2=float(coords[2]), y2=float(coords[3])
                )
                detection_id = _detection_id(source_id, frame_id, index, class_name, bbox)
                detections.append(
                    Detection(
                        detection_id=detection_id,
                        class_name=class_name,
                        confidence=probability,
                        bbox=bbox,
                    )
                )
        shape = getattr(frame, "shape", None)
        height = int(shape[0]) if shape is not None and len(shape) >= 2 else None
        width = int(shape[1]) if shape is not None and len(shape) >= 2 else None
        return DetectionFrame(
            source_id=source_id,
            event_time=ensure_utc(event_time),
            frame_id=frame_id,
            image_width=width,
            image_height=height,
            detector_id=self.variant.provider_id,
            detector_version=self.variant.model_version,
            detections=tuple(detections),
            source_sequence=source_sequence,
        )


class LegacyReplayYoloAdapter:
    """Parse the existing ``/analytics/yolo/bbox`` JSON list.

    The replay detector publishes center-form boxes and optional depth/world
    coordinates. This adapter preserves the event timestamp and converts the
    rows into a typed ``DetectionFrame`` without claiming track identity.
    """

    def __init__(
        self,
        *,
        detector_id: str = "yolo_vehicle_fast_640",
        detector_version: str = "legacy-replay-yolov8",
        class_allowlist: Iterable[str] = ("car", "truck", "bus", "motorcycle"),
        confidence_threshold: float = 0.0,
        coordinate_frame_id: str = "replay_world",
    ) -> None:
        self.detector_id = detector_id
        self.detector_version = detector_version
        self.class_allowlist = set(class_allowlist)
        self.confidence_threshold = confidence_threshold
        self.coordinate_frame_id = coordinate_frame_id

    def parse(
        self,
        document: Any,
        *,
        source_id: str,
        frame_id: str | None = None,
        source_sequence: int | None = None,
    ) -> DetectionFrame:
        rows = document if isinstance(document, list) else [document]
        rows = [row for row in rows if isinstance(row, dict)]
        if not rows:
            raise InvalidProviderInput("legacy YOLO payload contains no detection rows")
        event_times = [_parse_timestamp(row.get("t")) for row in rows]
        event_time = max(event_times)
        frame_id = frame_id or f"{source_id}:{event_time.isoformat()}"
        detections: list[Detection] = []
        for index, row in enumerate(rows):
            class_name = str(row.get("class") or row.get("label") or "object")
            confidence = float(row.get("conf", row.get("confidence", 1.0)))
            if self.class_allowlist and class_name not in self.class_allowlist:
                continue
            if confidence < self.confidence_threshold:
                continue
            raw_box = row.get("box")
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                raise InvalidProviderInput("legacy YOLO row must contain center-form box=[cx,cy,w,h]")
            bbox = BoundingBox.from_xywh(raw_box)
            world_point = None
            raw_world = row.get("world")
            if isinstance(raw_world, (list, tuple)) and len(raw_world) >= 2:
                world_point = Point2D(
                    x=float(raw_world[0]),
                    y=float(raw_world[1]),
                    coordinate_frame_id=self.coordinate_frame_id,
                )
            detection_id = str(
                row.get("id")
                or _detection_id(source_id, frame_id, index, class_name, bbox)
            )
            detections.append(
                Detection(
                    detection_id=detection_id,
                    class_name=class_name,
                    confidence=max(0.0, min(1.0, confidence)),
                    bbox=bbox,
                    world_point=world_point,
                    attributes={
                        "node": str(row.get("node") or ""),
                        "source_host": str(row.get("source_host") or ""),
                        "depth": float(row.get("depth", -1.0)),
                    },
                )
            )
        return DetectionFrame(
            source_id=source_id,
            event_time=event_time,
            frame_id=frame_id,
            detector_id=self.detector_id,
            detector_version=self.detector_version,
            detections=tuple(detections),
            source_sequence=source_sequence,
        )


def _detection_id(
    source_id: str,
    frame_id: str,
    index: int,
    class_name: str,
    bbox: BoundingBox,
) -> str:
    payload = f"{source_id}|{frame_id}|{index}|{class_name}|{bbox.model_dump_json()}"
    return sha256(payload.encode("utf-8")).hexdigest()[:32]


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        candidate = value.strip().replace("Z", "+00:00")
        try:
            return ensure_utc(datetime.fromisoformat(candidate))
        except ValueError:
            pass
    raise InvalidProviderInput(f"unsupported detection timestamp: {value!r}")


def _to_rows(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [[float(item) for item in row] for row in value]


def _to_scalars(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]
