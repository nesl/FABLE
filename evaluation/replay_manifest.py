"""Join experiment ground truth to replay scenarios without assuming mobile replay support."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import Field, field_validator

from fable.common.base import FableModel
from fable.common.time import ensure_utc

from .catalog import GroundTruthExperiment


class ReplayScenario(FableModel):
    scenario_id: str = Field(min_length=1)
    start_datetime: datetime
    observed_start_datetime: datetime | None = None
    observed_end_datetime: datetime | None = None
    nodes: tuple[str, ...] = ()
    zed_nodes: tuple[str, ...] = ()
    respeaker_nodes: tuple[str, ...] = ()
    source_root: str = ""

    @field_validator("start_datetime", "observed_start_datetime", "observed_end_datetime")
    @classmethod
    def _normalize_times(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            # Replay scanner timestamps may be naive local experiment timestamps.
            return value.replace(tzinfo=GroundTruthTimezone.EST).astimezone(GroundTruthTimezone.UTC)
        return ensure_utc(value)


class GroundTruthTimezone:
    from datetime import timezone

    EST = timezone(timedelta(hours=-5))
    UTC = timezone.utc


class ReplayRunManifest(FableModel):
    experiment_id: str
    replay_scenario_id: str | None = None
    matched_by: str = "UNMATCHED"
    start_delta_seconds: float | None = None
    available_orin_nodes: tuple[str, ...] = ()
    mobile_nodes_deferred: tuple[str, ...] = ()
    spatial_coordination_eligible: bool = False
    topology_deployment_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def match_replay_scenario(
    experiment: GroundTruthExperiment,
    scenarios: tuple[ReplayScenario, ...],
    *,
    tolerance_seconds: float = 180.0,
) -> ReplayRunManifest:
    candidates: list[tuple[float, ReplayScenario]] = []
    for scenario in scenarios:
        delta = abs((scenario.start_datetime - experiment.recording_start).total_seconds())
        if delta <= tolerance_seconds:
            candidates.append((delta, scenario))
    warnings = list(experiment.spatial_notes)
    if not candidates:
        return ReplayRunManifest(
            experiment_id=experiment.experiment_id,
            available_orin_nodes=(),
            mobile_nodes_deferred=experiment.unavailable_mobile_sensor_ids,
            spatial_coordination_eligible=experiment.spatial_coordination_eligible,
            topology_deployment_ids=experiment.topology_deployment_ids,
            warnings=tuple((*warnings, "No replay scenario matched the ground-truth start time.")),
        )
    delta, scenario = min(candidates, key=lambda item: (item[0], item[1].scenario_id))
    orin_nodes = tuple(
        sorted(
            {
                normalized
                for value in (*scenario.nodes, *scenario.zed_nodes, *scenario.respeaker_nodes)
                if (normalized := normalize_orin(value)) is not None
            }
        )
    )
    return ReplayRunManifest(
        experiment_id=experiment.experiment_id,
        replay_scenario_id=scenario.scenario_id,
        matched_by="NEAREST_START_TIME",
        start_delta_seconds=delta,
        available_orin_nodes=orin_nodes,
        mobile_nodes_deferred=experiment.unavailable_mobile_sensor_ids,
        spatial_coordination_eligible=experiment.spatial_coordination_eligible,
        topology_deployment_ids=experiment.topology_deployment_ids,
        warnings=tuple(warnings),
    )


def normalize_orin(value: str) -> str | None:
    match = re.search(r"orin[_-]?(\d+)", value.lower())
    return None if match is None else f"orin_{int(match.group(1))}"
