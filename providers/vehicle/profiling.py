"""Measured provider profiles used by physical planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any, Callable

from fable.catalog.profiles import ProviderProfileRecord, load_profile_records, save_profile_records


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
