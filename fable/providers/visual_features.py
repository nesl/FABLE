"""Visual projection, crop extraction, and descriptor providers.

ReID backends are real model adapters, but model files remain deployment assets.
Use ``scripts/provision_reid_models.py`` to download the pinned checkpoints from
``fable/providers/reid/models.json``.  Imports/model loading are lazy so the core
language/runtime can still be used without the heavyweight model stack.
"""
from __future__ import annotations

import os
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from .data_models import BoundingBox, Track, TrackFrame, ImageCrop, EmbeddingVector
from .object_detection import OptionalProviderDependency


class CameraProjectionProvider:
    provider_id = "camera_projection"
    provider_version = "1"

    def project(
        self,
        frame: TrackFrame,
        homography: Sequence[Sequence[float]],
        *,
        world_frame: str,
    ) -> TrackFrame:
        if len(homography) != 3 or any(len(row) != 3 for row in homography):
            raise ValueError("homography must be 3x3")
        h = [[float(v) for v in row] for row in homography]
        out = []
        for track in frame.tracks:
            x, y = track.bbox.center
            denom = h[2][0] * x + h[2][1] * y + h[2][2]
            if abs(denom) < 1e-12:
                continue
            wx = (h[0][0] * x + h[0][1] * y + h[0][2]) / denom
            wy = (h[1][0] * x + h[1][1] * y + h[1][2]) / denom
            out.append(
                Track(
                    track.object_id,
                    track.source_id,
                    track.class_name,
                    track.confidence,
                    track.bbox,
                    track.event_time,
                    (wx, wy),
                    world_frame,
                    track.velocity_xy_per_s,
                    track.age_frames,
                )
            )
        return TrackFrame(frame.source_id, frame.event_time, tuple(out))


class TrackCropExtractorProvider:
    provider_id = "track_crop_extractor"
    provider_version = "1"

    def extract(self, image: Any, tracks: TrackFrame) -> tuple[ImageCrop, ...]:
        out = []
        for track in tracks.tracks:
            x1, y1, x2, y2 = map(
                int, (track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2)
            )
            try:
                crop = image[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)].copy()
            except Exception:
                crop = (x1, y1, x2, y2)  # deterministic/non-array fallback for tests
            out.append(
                ImageCrop(
                    track.object_id,
                    track.source_id,
                    track.event_time,
                    crop,
                    track.confidence,
                )
            )
        return tuple(out)


class DescriptorBackend(Protocol):
    model_id: str
    model_version: str

    def embed(self, image: object) -> Sequence[float]: ...


class DeterministicDescriptorBackend:
    """Test-only descriptor backend; never advertise it as production ReID."""

    model_id = "deterministic_descriptor"
    model_version = "1"

    def embed(self, image: object) -> Sequence[float]:
        text = repr(image)
        vals = [
            float((sum(ord(c) for c in text[i::8]) % 997) / 997.0)
            for i in range(8)
        ]
        return _normalize(vals)


class _ReIDDescriptorProvider:
    provider_id = "reid_descriptor"
    provider_version = "1"

    def __init__(self, backend: DescriptorBackend) -> None:
        self.backend = backend
        self.provider_version = getattr(backend, "model_version", "1")

    def describe(self, crops: Sequence[ImageCrop]) -> tuple[EmbeddingVector, ...]:
        return tuple(
            EmbeddingVector(
                crop.object_id,
                crop.source_id,
                crop.event_time,
                tuple(float(v) for v in self.backend.embed(crop.image)),
                getattr(self.backend, "model_id", "descriptor"),
                self.provider_version,
            )
            for crop in crops
        )


class VehicleReIDDescriptorProvider(_ReIDDescriptorProvider):
    provider_id = "vehicle_reid_descriptor"

    def __init__(self, backend: DescriptorBackend | None = None) -> None:
        # Default is now the actual FastReID path. Tests may inject a deterministic
        # backend explicitly rather than silently getting fake identity evidence.
        super().__init__(backend or FastReIDDescriptorBackend())


class PersonReIDDescriptorProvider(_ReIDDescriptorProvider):
    provider_id = "person_reid_descriptor"

    def __init__(self, backend: DescriptorBackend | None = None) -> None:
        super().__init__(backend or TorchreidDescriptorBackend())


class FastReIDDescriptorBackend:
    """FastReID SBS-R50-IBN/VeRi inference backend.

    The implementation is adapted from the prior FABLE production provider.  It
    loads the FastReID config/checkpoint, builds the model directly, resizes crops
    to the configured inference shape, performs batched inference, and returns
    L2-normalized identity embeddings.
    """

    model_id = "fastreid:sbs_R50_ibn:vehicle"
    model_version = "fastreid-v0.1.1-veri-sbs-r50-ibn"

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        model_path: str | Path | None = None,
        device: str | None = None,
        model_id: str | None = None,
        model_version: str | None = None,
        extractor: Callable[[list[Any]], Any] | None = None,
    ) -> None:
        base = Path(__file__).with_name("reid")
        self.config_path = Path(
            config_path
            or os.environ.get("FABLE_VEHICLE_REID_CONFIG", "")
            or base / "fastreid_veri_sbs_r50_ibn.yaml"
        )
        self.model_path = Path(
            model_path
            or os.environ.get("FABLE_VEHICLE_REID_MODEL", "")
            or base / "models" / "vehicle.pth"
        )
        self.device = device or os.environ.get("FABLE_REID_DEVICE", "cuda:0")
        if model_id is not None:
            self.model_id = model_id
        if model_version is not None:
            self.model_version = model_version
        self._extractor = extractor

    def _load_extractor(self) -> Callable[[list[Any]], Any]:
        if self._extractor is not None:
            return self._extractor
        if not self.config_path.is_file():
            raise FileNotFoundError(f"FastReID config does not exist: {self.config_path}")
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"FastReID model does not exist: {self.model_path}; "
                "run scripts/provision_reid_models.py"
            )
        try:
            import cv2
            import numpy as np
            import torch
            from fastreid.config import get_cfg
            from fastreid.modeling.meta_arch import build_model
            from fastreid.utils.checkpoint import Checkpointer
        except ImportError as exc:  # pragma: no cover - optional heavy stack
            raise OptionalProviderDependency(
                "FastReID, PyTorch, NumPy, and OpenCV are required for vehicle ReID"
            ) from exc

        cfg = get_cfg()
        cfg.merge_from_file(str(self.config_path))
        cfg.MODEL.WEIGHTS = str(self.model_path)
        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.freeze()
        model = build_model(cfg)
        model.eval()
        Checkpointer(model).load(str(self.model_path))
        height, width = (int(value) for value in cfg.INPUT.SIZE_TEST)

        def extract(images: list[Any]) -> list[list[float]]:
            tensors = []
            for image in images:
                if not isinstance(image, np.ndarray) or image.ndim != 3:
                    raise ValueError("FastReID crops must be HxWxC NumPy images")
                resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
                tensors.append(
                    torch.from_numpy(
                        np.ascontiguousarray(resized.transpose(2, 0, 1))
                    ).float()
                )
            batch = torch.stack(tensors).to(self.device)
            with torch.inference_mode():
                vectors = model({"images": batch})
            if hasattr(vectors, "cpu"):
                vectors = vectors.cpu()
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()
            return [_normalize(row) for row in vectors]

        self._extractor = extract
        return extract

    def warmup(self) -> None:
        self._load_extractor()

    def embed(self, image: object) -> Sequence[float]:
        rows = self._load_extractor()([image])
        if len(rows) != 1:
            raise RuntimeError("FastReID returned an unexpected batch size")
        return _normalize(rows[0])


class TorchreidDescriptorBackend:
    """Torchreid OSNet-AIN/MSMT17 person-ReID inference backend."""

    model_id = "torchreid:osnet_ain_x1_0:person"
    model_version = "kaiyangzhou-osnet-a5c5cc0-msmt17"

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        architecture: str = "osnet_ain_x1_0",
        device: str | None = None,
        model_id: str | None = None,
        model_version: str | None = None,
        extractor: Callable[[list[Any]], Any] | None = None,
        input_color: str = "bgr",
    ) -> None:
        base = Path(__file__).with_name("reid")
        self.model_path = Path(
            model_path
            or os.environ.get("FABLE_PERSON_REID_MODEL", "")
            or base / "models" / "person.pth"
        )
        self.architecture = architecture
        self.device = device or os.environ.get("FABLE_REID_DEVICE", "cuda:0")
        self.input_color = input_color.lower()
        if self.input_color not in {"rgb", "bgr"}:
            raise ValueError("input_color must be 'rgb' or 'bgr'")
        if model_id is not None:
            self.model_id = model_id
        if model_version is not None:
            self.model_version = model_version
        self._extractor = extractor

    def _load_extractor(self) -> Callable[[list[Any]], Any]:
        if self._extractor is not None:
            return self._extractor
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Torchreid model does not exist: {self.model_path}; "
                "run scripts/provision_reid_models.py"
            )
        try:
            import numpy as np
            import torch
            import torchreid
            from PIL import Image
            from torchvision import transforms
        except ImportError as exc:  # pragma: no cover - optional heavy stack
            raise OptionalProviderDependency(
                "torchreid, PyTorch, torchvision, Pillow, and NumPy are required for person ReID"
            ) from exc

        model = torchreid.models.build_model(
            name=self.architecture,
            num_classes=1,
            loss="softmax",
            pretrained=False,
        )
        try:
            from torchreid.utils import load_pretrained_weights
            load_pretrained_weights(model, str(self.model_path))
        except Exception as exc:
            raise RuntimeError(f"failed to load Torchreid checkpoint {self.model_path}") from exc
        model.to(self.device)
        model.eval()
        transform = transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        def extract(images: list[Any]) -> list[list[float]]:
            tensors = []
            for image in images:
                if isinstance(image, Image.Image):
                    pil = image.convert("RGB")
                elif isinstance(image, np.ndarray) and image.ndim == 3:
                    array = image[..., ::-1] if self.input_color == "bgr" else image
                    pil = Image.fromarray(np.ascontiguousarray(array)).convert("RGB")
                else:
                    raise ValueError("Torchreid crops must be PIL or HxWxC NumPy images")
                tensors.append(transform(pil))
            batch = torch.stack(tensors).to(self.device)
            with torch.inference_mode():
                vectors = model(batch)
            vectors = vectors.detach().cpu().tolist()
            return [_normalize(row) for row in vectors]

        self._extractor = extract
        return extract

    def warmup(self) -> None:
        self._load_extractor()

    def embed(self, image: object) -> Sequence[float]:
        rows = self._load_extractor()([image])
        if len(rows) != 1:
            raise RuntimeError("Torchreid returned an unexpected batch size")
        return _normalize(rows[0])


class OpenClipDescriptorBackend:
    model_id = "openclip:ViT-B-32"
    model_version = "laion2b_s34b_b79k"

    def __init__(
        self,
        encoder: Callable[[object], Sequence[float]] | None = None,
        *,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ) -> None:
        self.encoder = encoder
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self._model = None
        self._preprocess = None

    def _load(self) -> None:
        if self.encoder is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover
            raise OptionalProviderDependency("open_clip_torch and PyTorch are required") from exc
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess

        def encode(image: object) -> Sequence[float]:
            tensor = preprocess(image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                vector = model.encode_image(tensor)[0].detach().cpu().tolist()
            return _normalize(vector)

        self.encoder = encode

    def embed(self, image: object) -> Sequence[float]:
        self._load()
        assert self.encoder is not None
        return _normalize(self.encoder(image))


class OpenClipVisualDescriptorProvider(_ReIDDescriptorProvider):
    provider_id = "openclip_visual_descriptor"

    def __init__(self, backend: DescriptorBackend | None = None) -> None:
        # OpenCLIP is a generic visual representation, not identity-calibrated by
        # default. It remains a separate provider alternative for later planning.
        super().__init__(backend or OpenClipDescriptorBackend())


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    vals = [float(v) for v in values]
    norm = sqrt(sum(v * v for v in vals))
    return tuple(vals if norm == 0 else [v / norm for v in vals])
