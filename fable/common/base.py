"""Shared Pydantic base classes and JSON-compatible value types."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import uuid
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, JsonValue, field_validator

JSONValue = JsonValue


class FableModel(BaseModel):
    """Strict base model used by all cross-module contracts.

    ``extra='forbid'`` is intentional: a cross-module producer must not smuggle
    provider, scheduler, or hypothesis-control fields through an otherwise
    unrelated contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
        populate_by_name=True,
    )


class FrozenFableModel(FableModel):
    """Immutable contract suitable for use inside plan labels and hashes."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
        populate_by_name=True,
        frozen=True,
    )


class VersionedModel(FableModel):
    """Base class for records carrying a globally visible schema version."""

    SCHEMA_VERSION: ClassVar[str]
    schema_version: str

    @field_validator("schema_version")
    @classmethod
    def _schema_version_must_match(cls, value: str) -> str:
        expected = getattr(cls, "SCHEMA_VERSION", None)
        if expected is not None and value != expected:
            raise ValueError(
                f"{cls.__name__} requires schema_version={expected!r}; got {value!r}"
            )
        return value


class FrozenVersionedModel(VersionedModel):
    """Immutable schema-versioned record used by deterministic planning state."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
        populate_by_name=True,
        frozen=True,
    )


def to_jsonable(value: Any) -> JSONValue:
    """Convert common Python/Pydantic values into deterministic JSON values."""

    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="python", exclude_none=True))
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Naive datetimes are not valid FABLE timestamps")
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        normalized = [to_jsonable(v) for v in value]
        return sorted(normalized, key=repr)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported canonical JSON value: {type(value)!r}")
