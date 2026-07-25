"""Vehicle ReID and general visual descriptor providers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from fable.common.time import EventTimeInterval, ensure_utc

from .errors import InvalidProviderInput, OptionalDependencyError
from .models import DescriptorRecord, DescriptorSet


class TorchScriptVehicleReIDDescriptor:
    """Run a user-supplied TorchScript vehicle-ReID model over crops.

    FABLE does not ship or pretend to train a ReID checkpoint. The caller must
    provide an actual identity model and its preprocessing function. Metadata is
    carried in the artifact so incompatible feature spaces cannot be mixed.
    """

    provider_id = "vehicle_reid_descriptor"

    def __init__(
        self,
        *,
        model_path: str | Path,
        model_id: str,
        model_version: str,
        preprocessing_id: str,
        preprocess: Callable[[Any], Any],
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_id = model_id
        self.model_version = model_version
        self.preprocessing_id = preprocessing_id
        self.preprocess = preprocess
        self.device = device
        self._model = model

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OptionalDependencyError(
                "PyTorch is required for the TorchScript vehicle-ReID provider"
            ) from exc
        if not self.model_path.exists():
            raise InvalidProviderInput(f"ReID model does not exist: {self.model_path}")
        self._model = torch.jit.load(str(self.model_path), map_location=self.device)
        self._model.eval()
        return self._model

    def encode(
        self,
        crops: Iterable[tuple[str, Any]],
        *,
        source_id: str,
        event_time_interval: EventTimeInterval,
    ) -> DescriptorSet:
        rows = tuple(crops)
        if not rows:
            raise InvalidProviderInput("ReID descriptor provider requires at least one crop")
        model = self._load()
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("PyTorch is required for vehicle ReID") from exc
        tensors = [self.preprocess(image) for _, image in rows]
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            vectors = model(batch)
        vectors = _normalized_rows(vectors)
        return DescriptorSet(
            schema_version="vehicle_reid_embedding_set.v1",
            source_id=source_id,
            event_time_interval=event_time_interval,
            descriptor_kind="vehicle_reid",
            model_id=self.model_id,
            model_version=self.model_version,
            preprocessing_id=self.preprocessing_id,
            dimension=len(vectors[0]),
            normalization="l2",
            distance_metric="cosine",
            records=tuple(
                DescriptorRecord(local_entity_id=entity_id, vector=tuple(vector))
                for (entity_id, _), vector in zip(rows, vectors)
            ),
            calibrated_for_identity=True,
        )


class OpenClipVisualDescriptor:
    """General-purpose image embedding, deliberately distinct from ReID.

    OpenCLIP embeddings may be useful as a broad representation alternative,
    but are not accepted as canonical identity evidence unless a deployment
    explicitly calibrates that feature space for the identity task.
    """

    provider_id = "openclip_visual_descriptor"

    def __init__(
        self,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        model_version: str | None = None,
        device: str = "cpu",
        calibrated_for_identity: bool = False,
        model: Any | None = None,
        preprocess: Callable[[Any], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.model_version = model_version or pretrained
        self.device = device
        self.calibrated_for_identity = calibrated_for_identity
        self._model = model
        self._preprocess = preprocess

    def _load(self) -> tuple[Any, Callable[[Any], Any]]:
        if self._model is not None and self._preprocess is not None:
            return self._model, self._preprocess
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OptionalDependencyError(
                "open_clip_torch is required for the OpenCLIP descriptor provider"
            ) from exc
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        return model, preprocess

    def encode(
        self,
        crops: Iterable[tuple[str, Any]],
        *,
        source_id: str,
        event_time_interval: EventTimeInterval,
    ) -> DescriptorSet:
        rows = tuple(crops)
        if not rows:
            raise InvalidProviderInput("OpenCLIP descriptor provider requires at least one crop")
        model, preprocess = self._load()
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("PyTorch is required for OpenCLIP") from exc
        batch = torch.stack([preprocess(image) for _, image in rows]).to(self.device)
        with torch.inference_mode():
            vectors = model.encode_image(batch)
        vectors = _normalized_rows(vectors)
        return DescriptorSet(
            schema_version="general_visual_embedding_set.v1",
            source_id=source_id,
            event_time_interval=event_time_interval,
            descriptor_kind="general_visual",
            model_id=f"openclip:{self.model_name}",
            model_version=self.model_version,
            preprocessing_id=f"openclip:{self.model_name}:{self.pretrained}",
            dimension=len(vectors[0]),
            normalization="l2",
            distance_metric="cosine",
            records=tuple(
                DescriptorRecord(local_entity_id=entity_id, vector=tuple(vector))
                for (entity_id, _), vector in zip(rows, vectors)
            ),
            calibrated_for_identity=self.calibrated_for_identity,
        )


class DeterministicDescriptorProvider:
    """Small deterministic provider used only by tests and fake-data demos."""

    provider_id = "deterministic_descriptor_test_provider"

    def __init__(self, *, dimension: int = 4, calibrated_for_identity: bool = True) -> None:
        if dimension < 2:
            raise ValueError("descriptor dimension must be at least two")
        self.dimension = dimension
        self.calibrated_for_identity = calibrated_for_identity

    def encode_ids(
        self,
        entity_ids: Iterable[str],
        *,
        source_id: str,
        event_time: datetime,
    ) -> DescriptorSet:
        timestamp = ensure_utc(event_time)
        records = []
        for entity_id in entity_ids:
            seed = sum((index + 1) * ord(char) for index, char in enumerate(entity_id))
            raw = [float(((seed >> index) & 15) + 1) for index in range(self.dimension)]
            norm = sum(value * value for value in raw) ** 0.5
            records.append(
                DescriptorRecord(
                    local_entity_id=entity_id,
                    vector=tuple(value / norm for value in raw),
                )
            )
        return DescriptorSet(
            schema_version=(
                "vehicle_reid_embedding_set.v1"
                if self.calibrated_for_identity
                else "general_visual_embedding_set.v1"
            ),
            source_id=source_id,
            event_time_interval=EventTimeInterval(start=timestamp, end=timestamp),
            descriptor_kind=("vehicle_reid" if self.calibrated_for_identity else "general_visual"),
            model_id="deterministic-test-model",
            model_version="1",
            preprocessing_id="deterministic-v1",
            dimension=self.dimension,
            normalization="l2",
            distance_metric="cosine",
            records=tuple(records),
            calibrated_for_identity=self.calibrated_for_identity,
        )


def _normalized_rows(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    rows = value if value and isinstance(value[0], (list, tuple)) else [value]
    normalized: list[list[float]] = []
    for row in rows:
        vector = [float(item) for item in row]
        norm = sum(item * item for item in vector) ** 0.5
        if norm <= 0:
            raise InvalidProviderInput("descriptor model emitted a zero vector")
        normalized.append([item / norm for item in vector])
    return normalized
