"""Clean evaluation policies implemented against the refactored planner API."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol

import yaml

from fable.planning import (
    ExecutionPlan,
    LinkState,
    PhysicalPlanner,
    PlanStep,
    RuntimeState,
)
from fable.runtime import ActiveFrontier


class EvaluationPolicy(Protocol):
    policy_id: str

    def plan(
        self, frontier: ActiveFrontier, runtime: RuntimeState, *, now: datetime
    ) -> ExecutionPlan: ...


@dataclass
class FablePolicy:
    policy_id: str = "FABLE"
    planner: PhysicalPlanner | None = None

    def plan(self, frontier, runtime, *, now):
        return (self.planner or PhysicalPlanner()).plan(frontier, runtime, now=now)


@dataclass
class GreedyFrontierPolicy:
    """B4: plan only the current frontier with a width-one joint search."""

    policy_id: str = "B4_GREEDY"
    planner: PhysicalPlanner | None = None

    def plan(self, frontier, runtime, *, now):
        planner = self.planner or PhysicalPlanner(beam_width=1)
        return planner.plan(frontier, runtime, now=now)


@dataclass
class ResourceOnlyPolicy:
    """B3: adapt to compute availability while ignoring network variation.

    This is an explicit ablation of FABLE's measured-link input, not a renamed
    invocation of the full planner. Current node capacity and running-provider
    state remain visible; available links are normalized to zero latency and
    unbounded throughput.
    """

    policy_id: str = "B3_RESOURCE"
    planner: PhysicalPlanner | None = None

    def plan(self, frontier, runtime, *, now):
        normalized = replace(
            runtime,
            links=tuple(
                LinkState(
                    row.source_node,
                    row.destination_node,
                    0.0,
                    float("inf"),
                    row.available,
                    row.measured_at,
                    "B3_IGNORED",
                    "B3_IGNORED",
                )
                for row in runtime.links
            ),
        )
        return (self.planner or PhysicalPlanner()).plan(frontier, normalized, now=now)


class StaticPlacementPolicy:
    """B1: one exact provider placement fixed for the complete CE run."""

    policy_id = "B1_STATIC"

    def __init__(self, event_name: str, placement_path: str | Path) -> None:
        raw = yaml.safe_load(Path(placement_path).read_text(encoding="utf-8")) or {}
        event = (raw.get("events") or {}).get(event_name)
        if not isinstance(event, dict):
            raise ValueError(f"no static B1 placement for event {event_name!r}")
        self.event_name = event_name
        self.steps = tuple(
            PlanStep(
                str(row["provider_id"]),
                str(row["node_id"]),
                tuple(str(value) for value in row.get("source_ids", ())),
                str(row["output_type"]),
            )
            for row in event.get("steps", ())
        )
        if not self.steps:
            raise ValueError(f"static B1 placement for {event_name!r} is empty")

    def plan(self, frontier, runtime, *, now):
        del frontier, now
        for step in self.steps:
            if step.node_id not in runtime.nodes:
                raise RuntimeError(f"B1 placement node {step.node_id!r} is absent")
            if not runtime.nodes[step.node_id].available:
                raise RuntimeError(f"B1 placement node {step.node_id!r} is unavailable")
            missing = sorted(set(step.source_ids) - set(runtime.sources))
            if missing:
                raise RuntimeError(f"B1 placement references missing sources: {missing}")
        return ExecutionPlan(
            self.steps,
            {"static-whole-event": tuple(step.provider_id for step in self.steps)},
            0.0,
            0,
            len(self.steps),
            0.0,
            1.0,
        )


def resolve_policy(
    policy_id: str,
    *,
    event_name: str | None = None,
    static_placements: str | Path | None = None,
) -> EvaluationPolicy:
    if policy_id == "FABLE":
        return FablePolicy()
    if policy_id == "B4_GREEDY":
        return GreedyFrontierPolicy()
    if policy_id == "B3_RESOURCE":
        return ResourceOnlyPolicy()
    if policy_id == "B1_STATIC":
        if event_name is None or static_placements is None:
            raise ValueError("B1_STATIC requires event_name and static_placements")
        return StaticPlacementPolicy(event_name, static_placements)
    raise ValueError(f"unknown evaluation policy {policy_id!r}")
