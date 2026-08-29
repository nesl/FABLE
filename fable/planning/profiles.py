"""Load small provider performance/resource profiles for physical planning."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

from .runtime_state import ProviderProfile


def load_provider_profiles(path: str | Path | None = None) -> dict[tuple[str, str], ProviderProfile]:
    source = Path(path) if path is not None else Path(__file__).parents[1] / "providers" / "provider_profiles.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("profiles"), dict):
        raise ValueError(f"{source}: invalid provider profile document")
    result: dict[tuple[str, str], ProviderProfile] = {}
    for provider_id, spec in raw["profiles"].items():
        if not isinstance(spec, dict):
            raise ValueError(f"{source}.profiles.{provider_id}: expected mapping")
        node_type = str(spec.get("node_type", "*"))
        result[(provider_id, node_type)] = ProviderProfile(
            provider_id=provider_id,
            node_type=node_type,
            startup_ms=float(spec.get("startup_ms", 0)),
            execution_ms=float(spec.get("execution_ms", 0)),
            cpu=float(spec.get("cpu", 0)),
            memory_mb=float(spec.get("memory_mb", 0)),
            gpu_memory_mb=float(spec.get("gpu_memory_mb", 0)),
            output_bytes=int(spec.get("output_bytes", 0)),
            quality=float(spec.get("quality", 1.0)),
        )
    return result
