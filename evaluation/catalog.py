"""Ground-truth experiment catalog and spatial/replay eligibility derivation."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from pydantic import Field, field_validator, model_validator

from fable.common.base import FableModel
from fable.common.time import ensure_utc

EST = timezone(timedelta(hours=-5), name="EST")


class SpatialTopologyStatus(str):
    pass


class GroundTruthExperiment(FableModel):
    experiment_id: str = Field(min_length=1)
    campaign_year: int
    date: str = Field(min_length=1)
    recording_start: datetime
    recording_end: datetime
    duration_seconds: int = Field(ge=0)
    ce_variant: str = Field(min_length=1)
    logical_stages: str = ""
    key_bindings: str = ""
    experiment_block: str = ""
    scenario_id: str = ""
    source_event_name: str = ""
    attempt_type: str = "standard"
    route_id: str = ""
    vehicle_ids: tuple[str, ...] = ()
    vehicle_count: int | None = Field(default=None, ge=0)
    relevant_nodes: tuple[str, ...] = ()
    quality_status: str = ""
    recommended_for_use: bool = False
    notes: str = ""
    source_file: str = ""
    source_row: int | None = Field(default=None, ge=0)
    timezone_basis: str = ""

    # Derived evaluation metadata.
    spatial_topology_known: bool = False
    spatial_coordination_eligible: bool = False
    topology_deployment_ids: tuple[str, ...] = ()
    topology_layout_ambiguous: bool = False
    replay_sensor_scope: str = "ORIN_FIXED_ONLY"
    replay_supported_sensor_ids: tuple[str, ...] = ()
    unavailable_mobile_sensor_ids: tuple[str, ...] = ()
    spatial_notes: tuple[str, ...] = ()

    @field_validator("recording_start", "recording_end")
    @classmethod
    def _normalize_times(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _validate_interval(self) -> "GroundTruthExperiment":
        if self.recording_end < self.recording_start:
            raise ValueError("experiment end precedes start")
        return self


class ExperimentCatalogSummary(FableModel):
    total_experiments: int
    recommended_experiments: int
    by_year: dict[int, int]
    by_variant: dict[str, int]
    spatial_eligible_by_year: dict[int, int]
    topology_unknown_count: int
    mobile_replay_deferred_count: int


class SensorTopologyMetadata:
    """Read-only view over the qualitative transition-model deployment metadata."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        fixed = payload.get("fixed_sensors", {})
        self.fixed_sensor_ids = tuple(sorted(str(item) for item in fixed))
        deployments = payload.get("mobile_deployments", {})
        self.mobile_by_deployment: dict[str, tuple[str, ...]] = {
            str(deployment_id): tuple(sorted(str(item) for item in data.get("nodes", {})))
            for deployment_id, data in deployments.items()
            if isinstance(data, dict)
        }

    @classmethod
    def load(cls, path: str | Path) -> "SensorTopologyMetadata":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


class ExperimentCatalog:
    def __init__(self, experiments: Iterable[GroundTruthExperiment]) -> None:
        self.experiments = tuple(experiments)
        self.by_id = {item.experiment_id: item for item in self.experiments}
        if len(self.by_id) != len(self.experiments):
            raise ValueError("duplicate experiment IDs in ground-truth catalog")

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        transition_model_path: str | Path | None = None,
    ) -> "ExperimentCatalog":
        topology = (
            SensorTopologyMetadata.load(transition_model_path)
            if transition_model_path is not None
            else None
        )
        rows: list[GroundTruthExperiment] = []
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                rows.append(_parse_row(raw, topology=topology))
        return cls(rows)

    def recommended(self) -> tuple[GroundTruthExperiment, ...]:
        return tuple(item for item in self.experiments if item.recommended_for_use)

    def spatial_eligible(self) -> tuple[GroundTruthExperiment, ...]:
        return tuple(item for item in self.experiments if item.spatial_coordination_eligible)

    def summary(self) -> ExperimentCatalogSummary:
        by_year = Counter(item.campaign_year for item in self.experiments)
        by_variant = Counter(item.ce_variant for item in self.experiments)
        spatial = Counter(
            item.campaign_year
            for item in self.experiments
            if item.spatial_coordination_eligible
        )
        return ExperimentCatalogSummary(
            total_experiments=len(self.experiments),
            recommended_experiments=sum(item.recommended_for_use for item in self.experiments),
            by_year=dict(sorted(by_year.items())),
            by_variant=dict(sorted(by_variant.items())),
            spatial_eligible_by_year=dict(sorted(spatial.items())),
            topology_unknown_count=sum(not item.spatial_topology_known for item in self.experiments),
            mobile_replay_deferred_count=sum(bool(item.unavailable_mobile_sensor_ids) for item in self.experiments),
        )


def _parse_row(
    row: dict[str, str],
    *,
    topology: SensorTopologyMetadata | None,
) -> GroundTruthExperiment:
    year = int(row["campaign_year"])
    start = _parse_est(row["recording_start_est"])
    duration_seconds = int(float(row.get("duration_seconds") or 0))
    end_text = row.get("recording_end_est", "").strip()
    end = _parse_est(end_text) if end_text else start + timedelta(seconds=duration_seconds)
    deployments, ambiguous = _deployment_ids(year, row.get("ce_variant", ""))
    known = year in (2024, 2025) and bool(deployments) and topology is not None

    fixed = topology.fixed_sensor_ids if known and topology is not None else ()
    mobile: set[str] = set()
    if known and topology is not None:
        for deployment_id in deployments:
            mobile.update(topology.mobile_by_deployment.get(deployment_id, ()))

    notes: list[str] = []
    if year == 2026:
        notes.append(
            "Sensor locations are not available for the 2026 campaign; exclude this trace "
            "from topology-based spatial-coordination metrics."
        )
    if mobile:
        notes.append(
            "The topology contains deployment-local mobile sensors, but the current replay "
            "stack exposes fixed Orin devices only. Mobile candidates are recorded as "
            "unavailable rather than counted as failed activations."
        )
    if ambiguous:
        notes.append(
            "The 2024 temporal_ce2 source contains two mobile layouts; fixed-Orin spatial "
            "rules remain usable, while mobile-node results require a run-specific layout override."
        )

    relevant = tuple(_split_tokens(row.get("relevant_nodes", "")))
    return GroundTruthExperiment(
        experiment_id=row["experiment_id"],
        campaign_year=year,
        date=row.get("date", ""),
        recording_start=start,
        recording_end=end,
        duration_seconds=duration_seconds,
        ce_variant=row.get("ce_variant", ""),
        logical_stages=row.get("logical_stages", ""),
        key_bindings=row.get("key_bindings", ""),
        experiment_block=row.get("experiment_block", ""),
        scenario_id=row.get("scenario_id", ""),
        source_event_name=row.get("source_event_name", ""),
        attempt_type=row.get("attempt_type", "standard"),
        route_id=row.get("route_id", ""),
        vehicle_ids=tuple(_split_tokens(row.get("vehicle_ids", ""))),
        vehicle_count=(int(row["vehicle_count"]) if row.get("vehicle_count", "").strip() else None),
        relevant_nodes=relevant,
        quality_status=row.get("quality_status", ""),
        recommended_for_use=_parse_bool(row.get("recommended_for_use", "")),
        notes=row.get("notes", ""),
        source_file=row.get("source_file", ""),
        source_row=(int(row["source_row"]) if row.get("source_row", "").strip() else None),
        timezone_basis=row.get("timezone_basis", ""),
        spatial_topology_known=known,
        spatial_coordination_eligible=known and _parse_bool(row.get("recommended_for_use", "")),
        topology_deployment_ids=deployments,
        topology_layout_ambiguous=ambiguous,
        replay_sensor_scope="ORIN_FIXED_ONLY",
        replay_supported_sensor_ids=fixed,
        unavailable_mobile_sensor_ids=tuple(sorted(mobile)),
        spatial_notes=tuple(notes),
    )


def _deployment_ids(year: int, variant: str) -> tuple[tuple[str, ...], bool]:
    key = variant.strip().lower()
    if year == 2024:
        if key == "vehicle convergence":
            return ("2024_spatial_ce1",), False
        if key == "route convoy":
            return ("2024_temporal_ce1",), False
        if key == "two-vehicle chase":
            return ("2024_temporal_ce2_page5", "2024_temporal_ce2_page6"), True
    if year == 2025:
        if key == "talking/rendezvous":
            return ("2025_package_exchange",), False
        if key in {"robbery with alarm", "two-visit stalking"}:
            return ("2025_robbery",), False
    return (), False


def _parse_est(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("ground-truth timestamp cannot be empty")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=EST)
    return parsed.astimezone(timezone.utc)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _split_tokens(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[;,]", value) if item.strip()]
