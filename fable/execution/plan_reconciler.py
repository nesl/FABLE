"""Reconcile a selected ExecutionPlan with currently running providers.

The reconciler is intentionally pure: it does not start processes, contact
nodes, or inspect CE state.  It only computes deterministic START / KEEP / STOP
sets from planner output and runtime state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fable.planning.physical_planner import ExecutionPlan, PlanStep
from fable.planning.runtime_state import RunningProvider
from fable.providers.provider_capabilities import load_provider_capabilities


@dataclass(frozen=True, slots=True, order=True)
class ProviderInstanceKey:
    provider_id: str
    node_id: str
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ids", tuple(sorted(self.source_ids)))

    @classmethod
    def from_running(cls, row: RunningProvider) -> "ProviderInstanceKey":
        return cls(row.provider_id, row.node_id, row.source_ids)

    @classmethod
    def from_step(cls, step: PlanStep) -> "ProviderInstanceKey":
        return cls(step.provider_id, step.node_id, step.source_ids)


@dataclass(frozen=True, slots=True)
class ProviderInstanceSpec:
    key: ProviderInstanceKey
    output_type: str = ""

    @classmethod
    def from_step(cls, step: PlanStep) -> "ProviderInstanceSpec":
        return cls(ProviderInstanceKey.from_step(step), step.output_type)


@dataclass(frozen=True, slots=True)
class ReconcileActions:
    start: tuple[ProviderInstanceSpec, ...] = ()
    keep: tuple[ProviderInstanceSpec, ...] = ()
    stop: tuple[ProviderInstanceKey, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.start or self.stop)


def reconcile_plan(
    running: Iterable[RunningProvider],
    plan: ExecutionPlan,
) -> ReconcileActions:
    """Return START / KEEP / STOP actions for ``plan``.

    Provider identity is deliberately small: provider implementation, node, and
    source set.  This mirrors the planner's sharing key and avoids a second
    lifecycle-specific identity system.
    """

    current = {ProviderInstanceKey.from_running(row) for row in running}
    desired_specs = {
        ProviderInstanceKey.from_step(step): ProviderInstanceSpec.from_step(step)
        for step in plan.steps
    }
    desired = set(desired_specs)

    start_keys = _ordered_start_keys(plan, desired - current)
    stop_keys = _ordered_stop_keys(current - desired)
    return ReconcileActions(
        start=tuple(desired_specs[key] for key in start_keys),
        keep=tuple(desired_specs[key] for key in sorted(desired & current)),
        stop=tuple(stop_keys),
    )


def _ordered_start_keys(
    plan: ExecutionPlan,
    keys: set[ProviderInstanceKey],
) -> tuple[ProviderInstanceKey, ...]:
    """Start downstream consumers before upstream producers.

    This avoids a newly-started source/detector immediately emitting values
    before the tracker/predicate workers that consume them have subscribed.
    """
    if not keys:
        return ()
    catalog_raw = load_provider_capabilities()
    catalog = catalog_raw.get("providers", catalog_raw)
    step_by_key = {ProviderInstanceKey.from_step(step): step for step in plan.steps}
    edges: dict[ProviderInstanceKey, set[ProviderInstanceKey]] = {key: set() for key in keys}
    all_keys = set(step_by_key)
    for upstream in all_keys:
        output_type = step_by_key[upstream].output_type
        for downstream in all_keys:
            if upstream == downstream or upstream.node_id != downstream.node_id:
                continue
            row = catalog.get(downstream.provider_id, {})
            if output_type not in row.get("inputs", ()):  # type: ignore[union-attr]
                continue
            if not set(upstream.source_ids).issubset(set(downstream.source_ids)):
                continue
            if upstream in keys and downstream in keys:
                edges[upstream].add(downstream)

    depth_cache: dict[ProviderInstanceKey, int] = {}
    def depth(key: ProviderInstanceKey, visiting: set[ProviderInstanceKey] | None = None) -> int:
        if key in depth_cache:
            return depth_cache[key]
        visiting = set() if visiting is None else visiting
        if key in visiting:
            return 0
        visiting.add(key)
        value = 0 if not edges.get(key) else 1 + max(depth(child, visiting) for child in edges[key])
        visiting.remove(key)
        depth_cache[key] = value
        return value

    # Terminal/downstream stages have depth 0 and start first.
    return tuple(sorted(keys, key=lambda key: (depth(key), key)))


def _ordered_stop_keys(keys: set[ProviderInstanceKey]) -> tuple[ProviderInstanceKey, ...]:
    """Stop likely upstream producers first; deterministic ordering otherwise."""
    if not keys:
        return ()
    catalog_raw = load_provider_capabilities()
    catalog = catalog_raw.get("providers", catalog_raw)
    raw_types = {"video_frame", "audio_window", "multichannel_audio"}
    def rank(key: ProviderInstanceKey) -> tuple[int, ProviderInstanceKey]:
        row = catalog.get(key.provider_id, {})
        inputs = set(row.get("inputs", ()))  # type: ignore[union-attr]
        return (0 if inputs.intersection(raw_types) else 1, key)
    return tuple(sorted(keys, key=rank))
