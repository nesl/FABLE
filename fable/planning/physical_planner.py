"""Small physical planner for frontier-driven provider execution.

Planning has three phases:

1. derive/coalesce physical requirements from the semantic frontier,
2. dynamically discover provider recipes by typed backward search,
3. enumerate feasible source/node placements and choose a joint plan with a
   bounded lexicographic beam search.

The provider catalog stays ordinary YAML; no persistent graph data structure is
introduced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from math import inf
from typing import Any, Mapping, Sequence

from fable.providers.provider_capabilities import load_provider_capabilities
from fable.runtime.frontier import ActiveFrontier, FrontierItem

from .provider_search import ProviderRecipe, ProviderSearcher, RawInput, recipe_signature
from .runtime_state import ProviderProfile, RuntimeState, SourceState, transfer_time_ms


@dataclass(frozen=True, slots=True)
class PhysicalRequirement:
    requirement_id: str
    item: FrontierItem
    site_id: str
    consumers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlanStep:
    provider_id: str
    node_id: str
    source_ids: tuple[str, ...]
    output_type: str

    @property
    def key(self) -> tuple[str, str, tuple[str, ...]]:
        return self.provider_id, self.node_id, tuple(sorted(self.source_ids))


@dataclass(frozen=True, slots=True)
class PlanAlternative:
    requirement_id: str
    recipe: ProviderRecipe
    steps: tuple[PlanStep, ...]
    source_ids: tuple[str, ...]
    predicted_completion_ms: float
    transfer_bytes: int
    new_provider_count: int
    peak_resource_fraction: float
    quality: float
    expiry_slack_ms: float


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    steps: tuple[PlanStep, ...]
    covers: Mapping[str, tuple[str, ...]]
    predicted_completion_ms: float
    transfer_bytes: int
    new_provider_count: int
    peak_resource_fraction: float
    quality: float

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted({step.provider_id for step in self.steps}))


@dataclass(frozen=True, slots=True)
class _PlacedValue:
    node_id: str
    source_ids: tuple[str, ...]
    output_type: str
    output_bytes: int
    ready_ms: float
    transfer_bytes: int
    steps: tuple[PlanStep, ...]
    quality: float


@dataclass(slots=True)
class _BeamState:
    steps: dict[tuple[str, str, tuple[str, ...]], PlanStep] = field(default_factory=dict)
    covers: dict[str, tuple[str, ...]] = field(default_factory=dict)
    completion_ms: float = 0.0
    transfer_bytes: int = 0
    minimum_expiry_slack_ms: float = inf
    quality: float = 1.0


class PhysicalPlanner:
    def __init__(
        self,
        *,
        provider_catalog: Mapping[str, Any] | None = None,
        beam_width: int = 8,
        max_recipes_per_requirement: int = 32,
        max_placements_per_recipe: int = 128,
        allow_cross_node_intermediates: bool = False,
    ) -> None:
        self.catalog = provider_catalog if provider_catalog is not None else load_provider_capabilities()
        self.searcher = ProviderSearcher(
            self.catalog,
            max_recipes=max_recipes_per_requirement,
        )
        self.beam_width = beam_width
        self.max_placements_per_recipe = max_placements_per_recipe
        self.allow_cross_node_intermediates = allow_cross_node_intermediates

    def plan(
        self,
        frontier: ActiveFrontier,
        runtime: RuntimeState,
        *,
        now: datetime,
    ) -> ExecutionPlan:
        requirements = coalesce_frontier(frontier, runtime)
        if not requirements:
            return ExecutionPlan((), {}, 0.0, 0, 0, 0.0, 1.0)

        alternatives: dict[str, tuple[PlanAlternative, ...]] = {}
        for requirement in requirements:
            recipes = self.searcher.recipes_for_frontier_item(requirement.item)
            rows: list[PlanAlternative] = []
            for recipe in recipes:
                rows.extend(
                    self._alternatives_for_recipe(requirement, recipe, runtime, now=now)
                )
                if len(rows) >= self.max_placements_per_recipe:
                    break
            rows = sorted(rows, key=_alternative_rank)
            if not rows:
                raise RuntimeError(
                    f"no feasible physical implementation for {requirement.item.predicate!r} "
                    f"at site {requirement.site_id!r}"
                )
            alternatives[requirement.requirement_id] = tuple(rows[: self.max_placements_per_recipe])

        ordered = sorted(
            requirements,
            key=lambda row: (
                row.item.expires_at or datetime.max.replace(tzinfo=now.tzinfo),
                row.requirement_id,
            ),
        )
        beam = [_BeamState()]
        for requirement in ordered:
            expanded: list[_BeamState] = []
            for partial in beam:
                for alternative in alternatives[requirement.requirement_id]:
                    candidate = self._merge_beam(partial, alternative, runtime)
                    if candidate is not None:
                        expanded.append(candidate)
            beam = _dedupe_beam(expanded)
            beam.sort(key=lambda row: self._beam_rank(row, runtime))
            beam = beam[: self.beam_width]
            if not beam:
                raise RuntimeError("no feasible joint execution plan")

        winner = beam[0]
        steps = tuple(sorted(winner.steps.values(), key=lambda step: step.key))
        new_provider_count, peak = _plan_resource_metrics(steps, runtime)
        return ExecutionPlan(
            steps=steps,
            covers=dict(winner.covers),
            predicted_completion_ms=winner.completion_ms,
            transfer_bytes=winner.transfer_bytes,
            new_provider_count=new_provider_count,
            peak_resource_fraction=peak,
            quality=winner.quality,
        )

    def _alternatives_for_recipe(
        self,
        requirement: PhysicalRequirement,
        recipe: ProviderRecipe,
        runtime: RuntimeState,
        *,
        now: datetime,
    ) -> list[PlanAlternative]:
        placed = self._place(recipe, requirement.site_id, runtime, limit=self.max_placements_per_recipe)
        output: list[PlanAlternative] = []
        for row in placed:
            steps = _unique_steps(row.steps)
            if not _resources_feasible(steps, runtime):
                continue
            new_count, peak = _plan_resource_metrics(steps, runtime)
            expiry_slack = inf
            if requirement.item.expires_at is not None:
                useful_ms = (requirement.item.expires_at - now).total_seconds() * 1000.0
                expiry_slack = useful_ms - row.ready_ms
                if expiry_slack < 0:
                    continue
            output.append(
                PlanAlternative(
                    requirement_id=requirement.requirement_id,
                    recipe=recipe,
                    steps=steps,
                    source_ids=row.source_ids,
                    predicted_completion_ms=row.ready_ms,
                    transfer_bytes=row.transfer_bytes,
                    new_provider_count=new_count,
                    peak_resource_fraction=peak,
                    quality=row.quality,
                    expiry_slack_ms=expiry_slack,
                )
            )
        return output

    def _place(
        self,
        recipe: ProviderRecipe | RawInput,
        site_id: str,
        runtime: RuntimeState,
        *,
        limit: int,
    ) -> list[_PlacedValue]:
        if isinstance(recipe, RawInput):
            return [
                _PlacedValue(
                    source.node_id,
                    (source.source_id,),
                    source.data_type,
                    source.sample_bytes,
                    0.0,
                    0,
                    (),
                    1.0,
                )
                for source in runtime.sources_of_type(recipe.data_type, site_id=site_id)
            ][:limit]

        child_sets = [self._place(child, site_id, runtime, limit=limit) for child in recipe.inputs]
        if any(not rows for rows in child_sets):
            return []
        combinations = product(*child_sets) if child_sets else [()]
        output: list[_PlacedValue] = []
        for children in combinations:
            source_ids = tuple(sorted({sid for child in children for sid in child.source_ids}))
            for node in sorted(runtime.nodes.values(), key=lambda value: value.node_id):
                if not node.available:
                    continue
                profile = runtime.profile_for(recipe.provider_id, node.node_type)
                if profile is None:
                    continue
                transfers = sum(child.transfer_bytes for child in children)
                ready_inputs = 0.0
                transfer_failed = False
                for child in children:
                    if child.node_id == node.node_id:
                        ready_inputs = max(ready_inputs, child.ready_ms)
                        continue
                    # Raw sensor inputs can be transferred to the compute node.
                    # Provider-produced intermediates remain co-located by
                    # default because M12 does not yet define an arbitrary
                    # intermediate-data transport protocol.
                    if child.steps and not self.allow_cross_node_intermediates:
                        transfer_failed = True
                        break
                    path = runtime.path_metrics(child.node_id, node.node_id)
                    transfer_ms = transfer_time_ms(child.output_bytes, path)
                    if transfer_ms is None:
                        transfer_failed = True
                        break
                    transfers += child.output_bytes
                    ready_inputs = max(ready_inputs, child.ready_ms + transfer_ms)
                if transfer_failed:
                    continue
                startup = 0.0 if runtime.running_key(recipe.provider_id, node.node_id, source_ids) else profile.startup_ms
                ready = ready_inputs + startup + profile.execution_ms
                steps = tuple(step for child in children for step in child.steps) + (
                    PlanStep(recipe.provider_id, node.node_id, source_ids, recipe.output_type),
                )
                steps = _unique_steps(steps)
                if not _resources_feasible(steps, runtime):
                    continue
                quality = min([profile.quality, *(child.quality for child in children)])
                output.append(
                    _PlacedValue(
                        node.node_id,
                        source_ids,
                        recipe.output_type,
                        profile.output_bytes,
                        ready,
                        transfers,
                        steps,
                        quality,
                    )
                )
                if len(output) >= limit:
                    return output
        return output

    def _merge_beam(
        self,
        partial: _BeamState,
        alternative: PlanAlternative,
        runtime: RuntimeState,
    ) -> _BeamState | None:
        steps = dict(partial.steps)
        for step in alternative.steps:
            steps[step.key] = step
        merged_steps = tuple(steps.values())
        if not _resources_feasible(merged_steps, runtime):
            return None
        return _BeamState(
            steps=steps,
            covers={**partial.covers, alternative.requirement_id: alternative.source_ids},
            completion_ms=max(partial.completion_ms, alternative.predicted_completion_ms),
            transfer_bytes=partial.transfer_bytes + alternative.transfer_bytes,
            minimum_expiry_slack_ms=min(partial.minimum_expiry_slack_ms, alternative.expiry_slack_ms),
            quality=min(partial.quality, alternative.quality),
        )

    def _beam_rank(self, state: _BeamState, runtime: RuntimeState) -> tuple:
        steps = tuple(state.steps.values())
        new_count, peak = _plan_resource_metrics(steps, runtime)
        slack_rank = -state.minimum_expiry_slack_ms if state.minimum_expiry_slack_ms != inf else -inf
        stable = tuple(sorted(step.key for step in steps))
        return (
            state.completion_ms,
            slack_rank,
            new_count,
            state.transfer_bytes,
            peak,
            -state.quality,
            stable,
        )


def coalesce_frontier(frontier: ActiveFrontier, runtime: RuntimeState) -> tuple[PhysicalRequirement, ...]:
    """Coalesce physically equivalent semantic needs while preserving consumers.

    Discovery and continuation needs are expanded per deployment site so new CE
    instances remain discoverable everywhere.  If a bound object has known
    source provenance, continuation work is scoped to that object's current site.
    """

    groups: dict[tuple, dict[str, Any]] = {}

    def add(item: FrontierItem, consumer: str) -> None:
        sites = _sites_for_item(item, runtime)
        for site in sites:
            key = (
                site,
                item.predicate,
                tuple(sorted(item.classes.items())),
                tuple(sorted((name, _freeze(value)) for name, value in item.parameters.items())),
            )
            row = groups.setdefault(key, {"item": item, "consumers": []})
            row["consumers"].append(consumer)
            existing = row["item"]
            if existing.expires_at is None or (
                item.expires_at is not None and item.expires_at < existing.expires_at
            ):
                row["item"] = item

    for item in frontier.discovery:
        add(item, "discovery")
    for instance_id, items in frontier.continuation.items():
        for item in items:
            add(item, instance_id)

    output: list[PhysicalRequirement] = []
    for index, (key, row) in enumerate(sorted(groups.items(), key=lambda pair: repr(pair[0]))):
        site = str(key[0])
        output.append(
            PhysicalRequirement(
                requirement_id=f"R{index}:{row['item'].predicate}@{site}",
                item=row["item"],
                site_id=site,
                consumers=tuple(sorted(set(row["consumers"]))),
            )
        )
    return tuple(output)


def _sites_for_item(item: FrontierItem, runtime: RuntimeState) -> tuple[str, ...]:
    source_ids = {
        runtime.object_sources[value]
        for value in item.arguments.values()
        if isinstance(value, str) and value in runtime.object_sources
    }
    if source_ids:
        sites = {
            runtime.sources[source_id].site
            for source_id in source_ids
            if source_id in runtime.sources
        }
        if sites:
            return tuple(sorted(sites))
    return runtime.sites()


def _unique_steps(steps: Sequence[PlanStep]) -> tuple[PlanStep, ...]:
    by_key = {step.key: step for step in steps}
    return tuple(sorted(by_key.values(), key=lambda step: step.key))


def _resources_feasible(steps: Sequence[PlanStep], runtime: RuntimeState) -> bool:
    used: dict[str, list[float]] = {}
    for step in _unique_steps(steps):
        node = runtime.nodes.get(step.node_id)
        if node is None or not node.available:
            return False
        profile = runtime.profile_for(step.provider_id, node.node_type)
        if profile is None:
            return False
        if runtime.running_key(step.provider_id, step.node_id, step.source_ids):
            continue
        row = used.setdefault(step.node_id, [0.0, 0.0, 0.0])
        row[0] += profile.cpu
        row[1] += profile.memory_mb
        row[2] += profile.gpu_memory_mb
    for node_id, (cpu, memory, gpu) in used.items():
        node = runtime.nodes[node_id]
        if cpu > node.cpu_free or memory > node.memory_mb_free or gpu > node.gpu_memory_mb_free:
            return False
    return True


def _plan_resource_metrics(steps: Sequence[PlanStep], runtime: RuntimeState) -> tuple[int, float]:
    new_count = 0
    used: dict[str, list[float]] = {}
    for step in _unique_steps(steps):
        node = runtime.nodes[step.node_id]
        profile = runtime.profile_for(step.provider_id, node.node_type)
        if profile is None or runtime.running_key(step.provider_id, step.node_id, step.source_ids):
            continue
        new_count += 1
        row = used.setdefault(step.node_id, [0.0, 0.0, 0.0])
        row[0] += profile.cpu
        row[1] += profile.memory_mb
        row[2] += profile.gpu_memory_mb
    peak = 0.0
    for node_id, (cpu, memory, gpu) in used.items():
        node = runtime.nodes[node_id]
        ratios = []
        if node.cpu_free not in {0, inf}:
            ratios.append(cpu / node.cpu_free)
        if node.memory_mb_free not in {0, inf}:
            ratios.append(memory / node.memory_mb_free)
        if node.gpu_memory_mb_free not in {0, inf}:
            ratios.append(gpu / node.gpu_memory_mb_free)
        peak = max(peak, *(ratios or [0.0]))
    return new_count, peak


def _alternative_rank(row: PlanAlternative) -> tuple:
    slack = -row.expiry_slack_ms if row.expiry_slack_ms != inf else -inf
    return (
        row.predicted_completion_ms,
        slack,
        row.new_provider_count,
        row.transfer_bytes,
        row.peak_resource_fraction,
        -row.quality,
        recipe_signature(row.recipe),
    )


def _dedupe_beam(rows: Sequence[_BeamState]) -> list[_BeamState]:
    by_key: dict[tuple, _BeamState] = {}
    for row in rows:
        key = tuple(sorted(row.steps))
        current = by_key.get(key)
        if current is None or row.completion_ms < current.completion_ms:
            by_key[key] = row
    return list(by_key.values())


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value
