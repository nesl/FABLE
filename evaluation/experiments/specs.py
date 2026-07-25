"""Machine-readable experiment matrix for the internal evaluation scope."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from fable.common.base import FableModel
from evaluation.schemas import BaselineId, EvaluationMode


class ExperimentQuestion(StrEnum):
    RQ1_END_TO_END = "RQ1_END_TO_END"
    RQ2_PLANNING = "RQ2_PLANNING"
    RQ3_OPERATING_ADAPTATION = "RQ3_OPERATING_ADAPTATION"
    RQ3_SPATIAL_COORDINATION = "RQ3_SPATIAL_COORDINATION"
    RQ3_CONTINUATION = "RQ3_CONTINUATION"
    RQ4_SCALING = "RQ4_SCALING"


class ExperimentSpec(FableModel):
    question: ExperimentQuestion
    modes: tuple[EvaluationMode, ...]
    baselines: tuple[BaselineId, ...]
    workload_families: tuple[str, ...]
    primary_metrics: tuple[str, ...]
    spatial_campaign_years: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()


def default_experiment_specs() -> tuple[ExperimentSpec, ...]:
    return (
        ExperimentSpec(
            question=ExperimentQuestion.RQ1_END_TO_END,
            modes=(EvaluationMode.COMMON_PERCEPTION, EvaluationMode.FULL_STACK),
            baselines=(
                BaselineId.B0_ALWAYS_ON,
                BaselineId.B1_HANDWRITTEN_STATIC,
                BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
                BaselineId.FABLE,
            ),
            workload_families=(
                "route_convoy",
                "vehicle_convergence",
                "talking_rendezvous",
                "robbery_with_alarm",
                "repeated_visit_stalking",
            ),
            primary_metrics=(
                "event_f1",
                "timely_recall",
                "detection_delay",
                "role_binding_accuracy",
                "gpu_seconds",
                "cpu_seconds",
                "transmitted_bytes",
                "provider_active_time",
            ),
        ),
        ExperimentSpec(
            question=ExperimentQuestion.RQ2_PLANNING,
            modes=(EvaluationMode.FULL_STACK,),
            baselines=(
                BaselineId.B2_STATIC_WHOLE_EVENT,
                BaselineId.B4_GREEDY_FRONTIER,
                BaselineId.FABLE,
                BaselineId.O1_EXHAUSTIVE_ORACLE,
            ),
            workload_families=("route_convoy", "robbery_with_alarm", "package_exchange"),
            primary_metrics=(
                "feasible_plan_rate",
                "deadline_miss_rate",
                "planning_latency",
                "oracle_cost_gap",
                "transfer_bytes_by_representation",
                "continuation_reuse",
            ),
        ),
        ExperimentSpec(
            question=ExperimentQuestion.RQ3_OPERATING_ADAPTATION,
            modes=(EvaluationMode.FULL_STACK,),
            baselines=(
                BaselineId.B2_STATIC_WHOLE_EVENT,
                BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
                BaselineId.FABLE,
            ),
            workload_families=("route_convoy", "robbery_with_alarm"),
            primary_metrics=(
                "timely_recall",
                "adaptation_latency",
                "processing_blackout",
                "state_bytes_moved",
                "redundant_work",
            ),
        ),
        ExperimentSpec(
            question=ExperimentQuestion.RQ3_SPATIAL_COORDINATION,
            modes=(EvaluationMode.FULL_STACK,),
            baselines=(
                BaselineId.SPATIAL_BROADCAST,
                BaselineId.SPATIAL_TOPOLOGY_SHORTLIST,
                BaselineId.SPATIAL_RESOURCE_ONLY,
                BaselineId.SPATIAL_FABLE,
                BaselineId.SPATIAL_ORACLE,
            ),
            workload_families=("route_convoy", "two_vehicle_chase", "robbery_with_alarm"),
            primary_metrics=(
                "timely_handoff_recall",
                "weighted_activation_fanout_reduction",
                "sensor_targeting_precision",
                "lookout_lead_time",
                "false_wakeup_rate",
            ),
            spatial_campaign_years=(2024, 2025),
            notes=(
                "2026 is excluded because sensor locations are unavailable.",
                "Mobile topology nodes are logged but excluded until replay containers support them.",
            ),
        ),
        ExperimentSpec(
            question=ExperimentQuestion.RQ3_CONTINUATION,
            modes=(EvaluationMode.FULL_STACK,),
            baselines=(BaselineId.B1_HANDWRITTEN_STATIC, BaselineId.FABLE),
            workload_families=("route_convoy", "repeated_visit_stalking", "package_exchange"),
            primary_metrics=(
                "continuation_success_rate",
                "identity_switch_rate",
                "state_transfer_bytes",
                "retrospective_recovery_rate",
                "buffer_expiration_rate",
            ),
        ),
        ExperimentSpec(
            question=ExperimentQuestion.RQ4_SCALING,
            modes=(EvaluationMode.COMMON_PERCEPTION, EvaluationMode.FULL_STACK),
            baselines=(BaselineId.B0_ALWAYS_ON, BaselineId.FABLE),
            workload_families=("synthetic_mixed",),
            primary_metrics=(
                "maximum_sustainable_workload",
                "timely_recall",
                "p95_control_plane_latency",
                "memory_per_hypothesis",
                "provider_reuse_rate",
                "deadline_miss_rate",
            ),
        ),
    )
