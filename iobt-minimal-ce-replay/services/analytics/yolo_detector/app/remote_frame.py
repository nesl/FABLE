"""Decode raw camera frames transported over MQTT.

Physical replay publishes a typed JSON envelope while older bridges publish
either raw image bytes or a base64 string.  Keep that wire compatibility in a
small dependency-free module so the offload boundary can be tested without
loading CUDA/Ultralytics.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any


def decode_remote_frame(message: str | bytes | bytearray) -> dict[str, Any]:
    """Return image bytes plus replay provenance from a remote frame message."""

    raw = message.encode("utf-8") if isinstance(message, str) else bytes(message)
    try:
        document = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        document = None

    if isinstance(document, dict) and document.get("schema_version") == "fable.remote_camera_frame.v1":
        encoded = document.get("data")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("remote camera frame envelope requires non-empty base64 data")
        try:
            image = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("remote camera frame envelope contains invalid base64 data") from exc
        return {
            "image": image,
            "event_time": document.get("t"),
            "replay_id": document.get("replay_id"),
            "frame_number": document.get("frame_number"),
            "schema_version": document["schema_version"],
        }

    # Legacy MQTT publishers sent the image itself, either as base64 text or
    # as binary JPEG/PNG bytes.
    try:
        image = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        image = raw
    return {
        "image": image,
        "event_time": None,
        "replay_id": None,
        "frame_number": None,
        "schema_version": None,
    }
