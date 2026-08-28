"""Portable resource/performance profile records for provider planning."""

from __future__ import annotations

import json

import yaml
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from pydantic import Field

from fable.common.base import FableModel
from fable.planning.models import ProviderResourceProfile


class ProviderProfileRecord(FableModel):
    schema_version: str = "fable.provider_profile.v1"
    provider_id: str = Field(min_length=1)
    node_class: str = Field(min_length=1)
    cold_start_samples_ms: tuple[float, ...] = ()
    warm_execution_samples_ms: tuple[float, ...] = ()
    cpu_cores: float = Field(default=0.1, ge=0)
    memory_mb: int = Field(default=64, ge=0)
    gpu_memory_mb: int = Field(default=0, ge=0)
    quality_score: float = Field(default=1.0, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_planner_profile(self) -> ProviderResourceProfile:
        startup = int(round(median(self.cold_start_samples_ms))) if self.cold_start_samples_ms else 0
        execution = int(round(median(self.warm_execution_samples_ms))) if self.warm_execution_samples_ms else 0
        return ProviderResourceProfile(
            provider_id=self.provider_id,
            node_class=self.node_class,
            startup_ms=startup,
            execution_ms=execution,
            cpu_cores=self.cpu_cores,
            memory_mb=self.memory_mb,
            gpu_memory_mb=self.gpu_memory_mb,
            quality_score=self.quality_score,
        )


def _load_mapping_or_list(path: str | Path) -> object:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def load_profile_records(path: str | Path) -> tuple[ProviderProfileRecord, ...]:
    document = _load_mapping_or_list(path)
    rows = document if isinstance(document, list) else document.get("profiles", [])
    return tuple(ProviderProfileRecord.model_validate(row) for row in rows)


def save_profile_records(path: str | Path, records: Iterable[ProviderProfileRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": "fable.provider_profiles.v1",
                "profiles": [record.model_dump(mode="json") for record in records],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_planner_profiles(path: str | Path) -> tuple[ProviderResourceProfile, ...]:
    """Load planner-facing resource profiles from JSON or YAML data."""

    document = _load_mapping_or_list(path)
    rows = document if isinstance(document, list) else (document or {}).get("profiles", [])
    return tuple(ProviderResourceProfile.model_validate(row) for row in rows)


def default_planner_profiles() -> tuple[ProviderResourceProfile, ...]:
    """Load packaged deterministic fallback profiles used when measurements are absent."""

    return load_planner_profiles(Path(__file__).with_name("default_provider_profiles.yaml"))
