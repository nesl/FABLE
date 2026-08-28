from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterable

from pydantic import Field

from fable.common.base import FableModel
from fable.common.ids import deterministic_id
from evaluation.catalog import ExperimentCatalog
from evaluation.schemas import BaselineId, EvaluationMode
from .specs import ExperimentQuestion, default_experiment_specs


class PlannedRun(FableModel):
    """Immutable experiment intent plus its concrete execution envelope.

    The question/experiment/baseline/mode fields are the portable matrix
    identity.  The remaining fields freeze how that intent is executed.  They
    deliberately live on the same typed record because unattended runners must
    not recover network, replay, topology, or retention behavior from ambient
    process state.
    """

    schema_version: str = "fable.planned_run.v1"
    run_id: str = ""
    question: ExperimentQuestion
    experiment_id: str
    baseline_id: BaselineId
    mode: EvaluationMode
    network_profile_id: str = "good_network"
    network_profile_path: str = "netwaggle/configs/profiles/good_network.json"
    repetition: int = Field(default=1, ge=1)
    random_seed: int = 0
    playback_mode: str = "realtime"
    provider_profile_version: str = "unversioned"
    disturbance_profile_id: str | None = None
    condition_trace_id: str = ""
    condition_trace_path: str = ""
    ce_start_offset_seconds: float = Field(default=0.0, ge=0)
    provider_execution_mode: str = "real"
    vlm_mode: str = "replayed_response"
    retrospective_policy_id: str = ""
    spatial_metrics_enabled: bool = False
    campaign_year: int = 0
    replay_supported_sensor_ids: tuple[str, ...] = ()
    unavailable_mobile_sensor_ids: tuple[str, ...] = ()
    topology_deployment_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def build_run_matrix(
    catalog: ExperimentCatalog,
    question: ExperimentQuestion,
    *,
    network_profiles: tuple[str, ...] = ("good_network",),
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
        for baseline, mode, network_profile in product(
            spec.baselines, spec.modes, network_profiles
        ):
            identity = {
                "question": question.value,
                "experiment_id": experiment.experiment_id,
                "baseline_id": baseline.value,
                "mode": mode.value,
                "repetition": 1,
                "random_seed": 0,
                "network_profile_id": network_profile,
            }
            rows.append(
                PlannedRun(
                    run_id=deterministic_id("eval_run", identity, length=32),
                    question=question,
                    experiment_id=experiment.experiment_id,
                    baseline_id=baseline,
                    mode=mode,
                    network_profile_id=network_profile,
                    network_profile_path=(
                        f"netwaggle/configs/profiles/{network_profile}.json"
                    ),
                    spatial_metrics_enabled=spatial,
                    campaign_year=experiment.campaign_year,
                    replay_supported_sensor_ids=experiment.replay_supported_sensor_ids,
                    unavailable_mobile_sensor_ids=experiment.unavailable_mobile_sensor_ids,
                    topology_deployment_ids=experiment.topology_deployment_ids,
                    warnings=experiment.spatial_notes,
                )
            )
    return tuple(rows)


def write_planned_runs(
    runs: Iterable[PlannedRun], path: str | Path
) -> Path:
    """Write a deterministic JSONL execution manifest.

    Fail closed on duplicate or missing run identifiers so a resumable
    campaign cannot silently conflate cells.
    """

    output = Path(path)
    rows = tuple(runs)
    run_ids = [item.run_id for item in rows]
    if any(not value for value in run_ids):
        raise ValueError("every planned run must have a non-empty run_id")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("planned run manifest contains duplicate run_id values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(item.model_dump_json() + "\n" for item in rows),
        encoding="utf-8",
    )
    return output


def _event_family(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    aliases = {
        "two_visit_stalking": "repeated_visit_stalking",
        "three_visit_stalking": "repeated_visit_stalking",
        "cross_sensor_robbery": "robbery_with_alarm",
    }
    return aliases.get(normalized, normalized)
