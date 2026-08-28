"""Capture and score blinded visual identity judgments for bounded E4 runs."""

from __future__ import annotations

import base64
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _decode_image(data_url: str) -> tuple[str, bytes]:
    header, separator, encoded = data_url.partition(",")
    if not separator or ";base64" not in header:
        raise ValueError("identity evidence must be a base64 image data URL")
    media_type = header.removeprefix("data:").split(";", 1)[0]
    suffixes = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    if media_type not in suffixes:
        raise ValueError(f"unsupported identity evidence media type: {media_type}")
    return suffixes[media_type], base64.b64decode(encoded, validate=True)


class IdentityEvidenceCapture:
    """Persist only evidence participating in emitted identity predictions."""

    def __init__(
        self,
        root: Path,
        *,
        experiment_id: str,
        baseline_id: str,
        maximum_predictions: int = 50,
    ) -> None:
        if maximum_predictions < 1:
            raise ValueError("maximum identity predictions must be positive")
        self.root = root
        self.experiment_id = experiment_id
        self.baseline_id = baseline_id
        self.maximum_predictions = maximum_predictions
        self.images = root / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.manifest = root / "identity_predictions.jsonl"
        self._evidence: dict[tuple[str, str], dict[str, Any]] = {}
        self._written: set[str] = set()
        self._referenced_images: set[str] = set()

    def observe_descriptor(self, document: dict[str, Any]) -> None:
        source_id = str(document.get("source_id") or "")
        entity_kind = str(document.get("entity_kind") or "")
        interval = document.get("event_time_interval") or {}
        if not source_id or entity_kind not in {"person", "vehicle"}:
            return
        for record in document.get("records") or ():
            if not isinstance(record, dict):
                continue
            local_id = str(record.get("local_entity_id") or "")
            urls = (
                record.get("source_context_image_data_urls")
                or record.get("source_crop_data_urls")
                or ()
            )
            if not local_id or not urls:
                continue
            try:
                suffix, content = _decode_image(str(urls[0]))
            except (ValueError, TypeError):
                continue
            digest = hashlib.sha256(content).hexdigest()
            image = self.images / f"{digest}{suffix}"
            if not image.exists():
                image.write_bytes(content)
            self._evidence[(source_id, local_id)] = {
                "path": str(image.resolve()),
                "sha256": digest,
                "entity_kind": entity_kind,
                "event_time_interval": interval,
            }

    def observe_associations(self, document: dict[str, Any]) -> int:
        if len(self._written) >= self.maximum_predictions:
            return 0
        left_source = str(document.get("left_source_id") or "")
        right_source = str(document.get("right_source_id") or "")
        entity_kind = str(document.get("entity_kind") or "")
        written = 0
        for association in document.get("associations") or ():
            if len(self._written) >= self.maximum_predictions:
                break
            if not isinstance(association, dict):
                continue
            left_id = str(association.get("left_local_entity_id") or "")
            right_id = str(association.get("right_local_entity_id") or "")
            left = self._evidence.get((left_source, left_id))
            right = self._evidence.get((right_source, right_id))
            if left is None or right is None:
                continue
            evidence_hashes = sorted((left["sha256"], right["sha256"]))
            pair_id = hashlib.sha256(
                json.dumps([entity_kind, *evidence_hashes]).encode("utf-8")
            ).hexdigest()
            prediction_id = hashlib.sha256(
                f"{self.experiment_id}|{self.baseline_id}|{pair_id}".encode()
            ).hexdigest()
            if prediction_id in self._written:
                continue
            row = {
                "schema_version": "fable.e4_identity_prediction.v1",
                "prediction_id": prediction_id,
                "pair_id": pair_id,
                "experiment_id": self.experiment_id,
                "baseline_id": self.baseline_id,
                "entity_kind": entity_kind,
                "predicted_same_identity": True,
                "association_basis": association.get("association_basis"),
                "association_model_id": association.get("association_model_id"),
                "prediction_confidence": association.get("confidence"),
                "left_source_id": left_source,
                "right_source_id": right_source,
                "left_local_entity_id": left_id,
                "right_local_entity_id": right_id,
                "left_image_path": left["path"],
                "right_image_path": right["path"],
                "left_sha256": left["sha256"],
                "right_sha256": right["sha256"],
                "left_event_time_interval": left["event_time_interval"],
                "right_event_time_interval": right["event_time_interval"],
            }
            with self.manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            self._written.add(prediction_id)
            self._referenced_images.update((left["path"], right["path"]))
            written += 1
        return written

    def finalize(self) -> None:
        """Discard descriptor frames that never supported an emitted prediction."""

        for image in self.images.iterdir():
            if image.is_file() and str(image.resolve()) not in self._referenced_images:
                image.unlink()


def summarize_judgments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["baseline_id"])].append(row)
    by_baseline = {}
    for baseline, values in sorted(groups.items()):
        determined = [row for row in values if row["judge_label"] != "UNDETERMINED"]
        agreed = [row for row in determined if row["agreement"]]
        by_baseline[baseline] = {
            "predictions": len(values),
            "determined": len(determined),
            "undetermined": len(values) - len(determined),
            "vlm_judged_binding_precision": (
                len(agreed) / len(determined) if determined else None
            ),
        }
    return {
        "schema_version": "fable.e4_vlm_judgment_summary.v1",
        "prediction_rows": len(rows),
        "unique_pairs": len({row["pair_id"] for row in rows}),
        "by_baseline": by_baseline,
        "validity_note": (
            "VLM-derived blinded reference judgments estimate binding precision; "
            "they are not human ground truth and do not by themselves measure recall."
        ),
    }
