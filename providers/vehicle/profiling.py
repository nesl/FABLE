"""Measured provider profiles used by physical planning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Any, Callable, Iterable

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


@dataclass
class ProviderProfiler:
    provider_id: str
    node_class: str
    cold_start_samples_ms: list[float] = field(default_factory=list)
    warm_execution_samples_ms: list[float] = field(default_factory=list)

    def time_cold_start(self, factory: Callable[[], Any]) -> Any:
        start = perf_counter_ns()
        instance = factory()
        self.cold_start_samples_ms.append((perf_counter_ns() - start) / 1_000_000.0)
        return instance

    def time_warm_call(self, call: Callable[[], Any]) -> Any:
        start = perf_counter_ns()
        result = call()
        self.warm_execution_samples_ms.append((perf_counter_ns() - start) / 1_000_000.0)
        return result

    def record(
        self,
        *,
        cpu_cores: float,
        memory_mb: int,
        gpu_memory_mb: int = 0,
        quality_score: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> ProviderProfileRecord:
        return ProviderProfileRecord(
            provider_id=self.provider_id,
            node_class=self.node_class,
            cold_start_samples_ms=tuple(self.cold_start_samples_ms),
            warm_execution_samples_ms=tuple(self.warm_execution_samples_ms),
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            quality_score=quality_score,
            metadata=metadata or {},
        )


def load_profile_records(path: str | Path) -> tuple[ProviderProfileRecord, ...]:
    target = Path(path)
    document = json.loads(target.read_text(encoding="utf-8"))
    rows = document if isinstance(document, list) else document.get("profiles", [])
    return tuple(ProviderProfileRecord.model_validate(row) for row in rows)


def save_profile_records(path: str | Path, records: Iterable[ProviderProfileRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {"schema_version": "fable.provider_profiles.v1", "profiles": [record.model_dump(mode="json") for record in records]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
