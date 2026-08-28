"""Immutable E1 workload/opportunity characterization manifests."""

from __future__ import annotations

from pydantic import Field

from fable.common.base import FrozenFableModel
from fable.common.ids import deterministic_id


class PlannedCharacterizationRun(FrozenFableModel):
    schema_version: str = "fable.planned_characterization_run.v1"
    run_id: str
    experiment_id: str = Field(min_length=1)
    event_family: str = Field(min_length=1)
    campaign_year: int
    characterize_semantic_structure: bool = True
    characterize_physical_alternatives: bool = True
    characterize_replayability: bool = True
    characterize_spatial_opportunity: bool


def build(catalog) -> tuple[PlannedCharacterizationRun, ...]:
    rows = []
    for experiment in catalog.recommended():
        identity = {
            "experiment_id": experiment.experiment_id,
            "event_family": experiment.ce_variant,
            "campaign_year": experiment.campaign_year,
        }
        rows.append(
            PlannedCharacterizationRun(
                run_id=deterministic_id(
                    "characterization_run", identity, length=32
                ),
                experiment_id=experiment.experiment_id,
                event_family=experiment.ce_variant,
                campaign_year=experiment.campaign_year,
                characterize_spatial_opportunity=(
                    experiment.spatial_coordination_eligible
                ),
            )
        )
    return tuple(sorted(rows, key=lambda item: item.run_id))
