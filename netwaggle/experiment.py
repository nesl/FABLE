"""Minimal, replay-relative network experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class NetworkCondition:
    starts_at_seconds: float
    affected_link_name: str
    remains_for_seconds: float | None = None
    connection_status: str | None = None
    bandwidth_mbps: float | None = None
    latency_ms: float | None = None
    jitter_ms: float | None = None
    packet_loss_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.starts_at_seconds < 0:
            raise ValueError("starts_at_seconds must be nonnegative")
        if self.remains_for_seconds is not None and self.remains_for_seconds <= 0:
            raise ValueError("remains_for_seconds must be positive")
        if not self.affected_link_name:
            raise ValueError("affected_link_name must be non-empty")
        if self.connection_status not in {None, "connected", "disconnected"}:
            raise ValueError("connection_status must be connected or disconnected")
        for name in ("bandwidth_mbps", "latency_ms", "jitter_ms"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.packet_loss_fraction is not None and not 0 <= self.packet_loss_fraction <= 1:
            raise ValueError("packet_loss_fraction must be between 0 and 1")
        if all(getattr(self, name) is None for name in _CONDITION_FIELDS):
            raise ValueError("network condition must set at least one condition field")

    def values(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _CONDITION_FIELDS if getattr(self, name) is not None}


@dataclass(frozen=True, slots=True)
class NetworkExperiment:
    topology: Path
    conditions: tuple[NetworkCondition, ...]
    event: Path | None = None
    replay_manifest: Path | None = None
    timeout_seconds: float = 120
    expected_initial_node: str | None = None
    expected_disconnected_node: str | None = None
    require_completion: bool = False


_CONDITION_FIELDS = (
    "connection_status", "bandwidth_mbps", "latency_ms", "jitter_ms",
    "packet_loss_fraction",
)
_ENTRY_FIELDS = {"starts_at_seconds", "remains_for_seconds", "affected_link_name", *_CONDITION_FIELDS}
_EXPERIMENT_FIELDS = {
    "topology", "conditions", "event", "replay_manifest", "timeout_seconds",
    "expected_initial_node", "expected_disconnected_node", "require_completion",
}


def load_network_experiment(path: str | Path) -> NetworkExperiment:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not {"topology", "conditions"} <= set(raw):
        raise ValueError("experiment requires topology and conditions")
    unknown = set(raw) - _EXPERIMENT_FIELDS
    if unknown:
        raise ValueError(f"experiment has unknown fields: {sorted(unknown)}")
    rows = raw["conditions"]
    if not isinstance(rows, list):
        raise ValueError("conditions must be a list")
    conditions = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"condition {index} must be a mapping")
        unknown = set(row) - _ENTRY_FIELDS
        if unknown:
            raise ValueError(f"condition {index} has unknown fields: {sorted(unknown)}")
        conditions.append(NetworkCondition(**row))
    def resolved_path(value: object | None) -> Path | None:
        if value is None:
            return None
        result = Path(str(value))
        return (result if result.is_absolute() else Path.cwd() / result).resolve()

    timeout = float(raw.get("timeout_seconds", 120))
    if timeout <= 0:
        raise ValueError("timeout_seconds must be positive")
    return NetworkExperiment(
        resolved_path(raw["topology"]), tuple(conditions),
        resolved_path(raw.get("event")), resolved_path(raw.get("replay_manifest")),
        timeout, raw.get("expected_initial_node"),
        raw.get("expected_disconnected_node"), bool(raw.get("require_completion", False)),
    )
