from __future__ import annotations

from itertools import product

from pydantic import Field

from fable.common.base import FableModel
from evaluation.catalog import ExperimentCatalog
from evaluation.schemas import BaselineId, EvaluationMode
from .specs import ExperimentQuestion, default_experiment_specs


class PlannedRun(FableModel):
    question: ExperimentQuestion
    experiment_id: str
    baseline_id: BaselineId
    mode: EvaluationMode
    spatial_metrics_enabled: bool = False
    warnings: tuple[str, ...] = ()


def build_run_matrix(
    catalog: ExperimentCatalog,
    question: ExperimentQuestion,
) -> tuple[PlannedRun, ...]:
    spec = next(item for item in default_experiment_specs() if item.question == question)
    rows = []
    for experiment in catalog.recommended():
        normalized = _event_family(experiment.ce_variant)
        if spec.workload_families != ("synthetic_mixed",) and normalized not in set(spec.workload_families):
            continue
        spatial = (
            question == ExperimentQuestion.RQ3_SPATIAL_COORDINATION
            and experiment.spatial_coordination_eligible
        )
        if question == ExperimentQuestion.RQ3_SPATIAL_COORDINATION and not spatial:
            continue
        for baseline, mode in product(spec.baselines, spec.modes):
            rows.append(
                PlannedRun(
                    question=question,
                    experiment_id=experiment.experiment_id,
                    baseline_id=baseline,
                    mode=mode,
                    spatial_metrics_enabled=spatial,
                    warnings=experiment.spatial_notes,
                )
            )
    return tuple(rows)


def _event_family(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    aliases = {
        "two_visit_stalking": "repeated_visit_stalking",
        "three_visit_stalking": "repeated_visit_stalking",
        "cross_sensor_robbery": "robbery_with_alarm",
    }
    return aliases.get(normalized, normalized)
