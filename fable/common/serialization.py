"""Version-aware serialization and compatibility checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .base import VersionedModel
from .schemas import (
    ArtifactRef,
    ExecutionPlan,
    FrontierSnapshot,
    Hypothesis,
    NodeHeartbeat,
    PhysicalPlanLabel,
    PredicateDemand,
    PredicateResult,
    ProviderContract,
    ProviderFamily,
    ProviderLease,
    SemanticCheckpoint,
    SemanticGraph,
)

T = TypeVar("T", bound=VersionedModel)
_VERSION_RE = re.compile(r"^(?P<name>[a-z0-9_.-]+)\.v(?P<major>[1-9][0-9]*)$")


@dataclass(frozen=True)
class ParsedSchemaVersion:
    name: str
    major: int


def parse_schema_version(value: str) -> ParsedSchemaVersion:
    match = _VERSION_RE.fullmatch(value)
    if not match:
        raise ValueError(f"invalid FABLE schema version: {value!r}")
    return ParsedSchemaVersion(name=match.group("name"), major=int(match.group("major")))


def schemas_compatible(producer: str, consumer: str) -> bool:
    """Phase-0 compatibility rule: same schema family and exact major version."""

    left = parse_schema_version(producer)
    right = parse_schema_version(consumer)
    return left == right


SCHEMA_REGISTRY: dict[str, type[VersionedModel]] = {
    cls.SCHEMA_VERSION: cls
    for cls in (
        SemanticGraph,
        Hypothesis,
        FrontierSnapshot,
        SemanticCheckpoint,
        PredicateDemand,
        ProviderContract,
        ProviderFamily,
        ArtifactRef,
        PhysicalPlanLabel,
        ExecutionPlan,
        PredicateResult,
        ProviderLease,
        NodeHeartbeat,
    )
}


def dump_model(model: BaseModel, *, indent: int | None = 2) -> str:
    return model.model_dump_json(indent=indent, exclude_none=True)


def load_versioned(data: str | bytes | dict[str, Any]) -> VersionedModel:
    if isinstance(data, (str, bytes)):
        raw = json.loads(data)
    else:
        raw = data
    if not isinstance(raw, dict):
        raise ValueError("versioned payload must be a JSON object")
    version = raw.get("schema_version")
    if not isinstance(version, str):
        raise ValueError("versioned payload is missing schema_version")
    model = SCHEMA_REGISTRY.get(version)
    if model is None:
        raise ValueError(f"unsupported schema_version: {version}")
    return model.model_validate(raw)


def write_fixture(path: str | Path, model: BaseModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_model(model) + "\n", encoding="utf-8")
