"""Architecture-level facade for physical planning.

Input is a set of provider-independent ``PredicateDemand`` objects. Output is
the generated physical alternatives, beam-search trace, and selected concrete
``ExecutionPlan``. This module changes no search policy; it makes the existing
alternative-generation then search pipeline obvious to readers and callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from fable.common.schemas import ExecutionPlan, PredicateDemand

from .alternative_graph import PhysicalAlternativeGraphBuilder
from .beam_search import BoundedLabelPlanner
from .models import PhysicalAlternativeGraph
from .search_models import PlanSearchResult


@dataclass(frozen=True)
class PlanningResult:
    """Alternative set, search result, and selected execution plan."""

    alternatives: PhysicalAlternativeGraph
    search: PlanSearchResult

    @property
    def execution_plan(self) -> ExecutionPlan | None:
        return self.search.execution_plan


class PhysicalPlanner:
    """Generate physical implementations and select one feasible plan."""

    def __init__(
        self,
        *,
        alternative_generator: PhysicalAlternativeGraphBuilder,
        plan_search: BoundedLabelPlanner,
    ) -> None:
        self.alternative_generator = alternative_generator
        self.plan_search = plan_search

    def plan(
        self,
        demands: Iterable[PredicateDemand],
        *,
        now: datetime | None = None,
        required_checkpoint_consumers: Iterable[str] = (),
    ) -> PlanningResult:
        """Enumerate alternatives, search them, and return the selected plan."""

        materialized_demands = tuple(demands)
        alternatives = self.alternative_generator.build(materialized_demands, now=now)
        search = self.plan_search.search(
            alternatives,
            materialized_demands,
            now=now,
            required_checkpoint_consumers=required_checkpoint_consumers,
        )
        return PlanningResult(alternatives=alternatives, search=search)


__all__ = ["PhysicalPlanner", "PlanningResult"]
