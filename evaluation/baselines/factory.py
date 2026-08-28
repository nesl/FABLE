from __future__ import annotations

from pathlib import Path

from fable.planning.beam_search import BoundedLabelPlanner

from evaluation.schemas import BaselineId

from .policies import (
    AlwaysOnPolicy,
    ProduceAllPolicy,
    ExhaustiveOraclePolicy,
    FablePolicy,
    GreedyFrontierPolicy,
    HandwrittenStaticPolicy,
    StaticWholeEventPolicy,
    TaskResourceAdaptivePolicy,
)
from .static_registry import StaticPipelineRegistry


def build_baseline_policy(
    baseline_id: BaselineId,
    *,
    planner: BoundedLabelPlanner,
    static_registry_path: str | Path = "evaluation/manifests/baselines/static_pipelines.yaml",
):
    if baseline_id == BaselineId.B0_ALWAYS_ON:
        return AlwaysOnPolicy()
    if baseline_id == BaselineId.B0_PRODUCE_ALL:
        return ProduceAllPolicy()
    if baseline_id in {
        BaselineId.B1_HANDWRITTEN_STATIC,
        BaselineId.B1_STATIC_WHOLE_EVENT,
    }:
        return HandwrittenStaticPolicy(StaticPipelineRegistry.load(static_registry_path))
    if baseline_id == BaselineId.B2_STATIC_WHOLE_EVENT:
        return StaticWholeEventPolicy(planner)
    if baseline_id == BaselineId.B3_TASK_RESOURCE_ADAPTIVE:
        return TaskResourceAdaptivePolicy(planner)
    if baseline_id == BaselineId.B4_GREEDY_FRONTIER:
        return GreedyFrontierPolicy()
    if baseline_id == BaselineId.FABLE:
        return FablePolicy(planner)
    if baseline_id == BaselineId.O1_EXHAUSTIVE_ORACLE:
        return ExhaustiveOraclePolicy(planner)
    raise ValueError(f"unsupported controlled baseline: {baseline_id}")
