"""Object-detection providers and model-specific variants."""
from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Iterable

from .data_models import BoundingBox, Detection, DetectionFrame

class OptionalProviderDependency(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class YoloConfig:
    provider_id: str
    model_path: str = "yolov8s.pt"
    model_version: str = "unresolved"
    image_size: int = 640
    confidence_threshold: float = 0.25
    class_allowlist: tuple[str, ...] = ()
    device: str = "auto"
    def __post_init__(self) -> None:
        if not self.provider_id or not self.model_path: raise ValueError("provider_id/model_path required")
        if self.image_size <= 0: raise ValueError("image_size must be positive")
        if not 0 <= self.confidence_threshold <= 1: raise ValueError("confidence_threshold must be in [0,1]")

class UltralyticsObjectDetectorProvider:
    """Lazy Ultralytics adapter preserving the model's native class labels."""
    def __init__(self, config: YoloConfig, *, model: Any | None = None) -> None:
        self.config, self._model = config, model
        self.provider_id = config.provider_id
        self.provider_version = config.model_version
    def _load(self) -> Any:
        if self._model is not None: return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise OptionalProviderDependency("Ultralytics is required for YOLO providers") from exc
        self._model = YOLO(self.config.model_path); return self._model
    def detect(self, image: Any, *, source_id: str, event_time: datetime, frame_id: str="") -> DetectionFrame:
        model = self._load(); kwargs={"verbose":False,"imgsz":self.config.image_size,"conf":self.config.confidence_threshold}
        if self.config.device != "auto": kwargs["device"] = self.config.device
        results = model(image, **kwargs); allow={x.lower() for x in self.config.class_allowlist}; out=[]; row_index=0
        for result in results:
            boxes=getattr(result,"boxes",None)
            if boxes is None: continue
            xyxy=_rows(getattr(boxes,"xyxy",())); confs=_values(getattr(boxes,"conf",())); classes=_values(getattr(boxes,"cls",())); names=getattr(result,"names",{})
            for coords, confidence, class_index in zip(xyxy, confs, classes):
                label=_class_name(names,int(class_index)); confidence=float(confidence)
                if allow and label.lower() not in allow: continue
                if confidence < self.config.confidence_threshold: continue
                out.append(Detection(label, confidence, BoundingBox(*map(float,coords[:4])), f"{source_id}:{frame_id or int(event_time.timestamp()*1000)}:{row_index}")); row_index += 1
        shape=getattr(image,"shape",None); h=int(shape[0]) if shape is not None and len(shape)>=2 else None; w=int(shape[1]) if shape is not None and len(shape)>=2 else None
        return DetectionFrame(source_id,event_time,tuple(out),frame_id,w,h)

class YoloVehicleFast640Provider(UltralyticsObjectDetectorProvider):
    def __init__(self, *, model: Any | None=None, model_path: str="yolov8s.pt", device: str="auto") -> None:
        super().__init__(YoloConfig("yolo_vehicle_fast_640",model_path,image_size=640,class_allowlist=("car","truck","bus","motorcycle"),device=device),model=model)

class YoloVehicleBalanced960Provider(UltralyticsObjectDetectorProvider):
    def __init__(self, *, model: Any | None=None, model_path: str="yolov8s.pt", device: str="auto") -> None:
        super().__init__(YoloConfig("yolo_vehicle_balanced_960",model_path,image_size=960,class_allowlist=("car","truck","bus","motorcycle"),device=device),model=model)

class YoloFullContext960Provider(UltralyticsObjectDetectorProvider):
    def __init__(self, *, model: Any | None=None, model_path: str="yolov8s.pt", device: str="auto") -> None:
        super().__init__(YoloConfig("yolo_full_context_960",model_path,image_size=960,device=device),model=model)

class PackageDetectorProvider(UltralyticsObjectDetectorProvider):
    """High-resolution package-like detector/filter.

    A deployment may provide custom weights; with generic COCO weights the
    default allowlist uses bag/luggage classes and therefore should be treated as
    an approximation rather than a universal package detector.
    """
    def __init__(self, *, model: Any | None=None, model_path: str="yolov8s.pt", labels: Iterable[str]=("backpack","handbag","suitcase"), device: str="auto") -> None:
        super().__init__(YoloConfig("package_detector",model_path,image_size=960,class_allowlist=tuple(labels),device=device),model=model)

def _class_name(names: Any,index:int)->str: return str(names[index])
def _rows(value: Any) -> list[list[float]]:
    if value is None: return []
    for method in ("detach","cpu"):
        if hasattr(value,method): value=getattr(value,method)()
    if hasattr(value,"numpy"): value=value.numpy()
    if hasattr(value,"tolist"): value=value.tolist()
    rows=list(value)
    if rows and not isinstance(rows[0],(list,tuple)): rows=[rows]
    return [[float(v) for v in row] for row in rows]
def _values(value: Any) -> list[float]:
    if value is None: return []
    for method in ("detach","cpu"):
        if hasattr(value,method): value=getattr(value,method)()
    if hasattr(value,"numpy"): value=value.numpy()
    if hasattr(value,"tolist"): value=value.tolist()
    return [float(v) for v in list(value)]
