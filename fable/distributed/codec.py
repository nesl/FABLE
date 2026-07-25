"""Deterministic JSON encoding and typed decoding for distributed messages."""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel

from fable.common.ids import canonical_json_bytes

T = TypeVar("T", bound=BaseModel)


def encode_model(model: BaseModel) -> bytes:
    return canonical_json_bytes(model)


def decode_model(payload: bytes | bytearray | str, model_type: type[T]) -> T:
    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload).decode("utf-8")
    else:
        raw = payload
    return model_type.model_validate(json.loads(raw))
