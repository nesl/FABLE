"""Planning-policy seam used by the deployed closed-loop controller.

The controller owns semantic progress and execution. A policy may only constrain
which already-grounded physical alternatives are eligible for the current
checkpoint. Evaluation baselines implement this protocol outside FABLE core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from fable.common.schemas import Hypothesis, PredicateDemand
from fable.planning import ArtifactCatalog, DemandCompileContext
from fable.planning.deployment import DeploymentGraph
from fable.planning.models import PhysicalAlternativeGraph
from fable.planning.provider_registry import ProviderRegistry
from fable.semantic.compiled import CompiledSemanticGraph


@dataclass(frozen=True)
class ControllerPlanningContext:
    request_id: str
    trace_id: str
    placement_id: str
    family_id: str
    hypothesis_id: UUID
    semantic_epoch: int
    resource_epoch: int
    checkpoint_id: UUID
    hypothesis: Hypothesis
    semantic_graph: CompiledSemanticGraph
    demand_context: DemandCompileContext
    frontier_demands: tuple[PredicateDemand, ...]
    frontier_graph: PhysicalAlternativeGraph
    deployment: DeploymentGraph
    provider_registry: ProviderRegistry
    artifact_catalog: ArtifactCatalog
    runtime_provider_keys: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class ControllerPlanningDecision:
    """Policy constraint applied before the normal FABLE feasibility/search path.

    ``allowed_alternative_ids=None`` means no policy-side restriction: the normal
    bounded FABLE planner selects among every feasible current-frontier option.
    An empty tuple deliberately makes the checkpoint infeasible.
    """

    policy_id: str
    allowed_alternative_ids: tuple[str, ...] | None = None
    reason: str = ""
    frozen: bool = False


class ControllerPlanningPolicy(Protocol):
    policy_id: str

    def select(self, context: ControllerPlanningContext) -> ControllerPlanningDecision: ...


class FableControllerPlanningPolicy:
    policy_id = "FABLE"

    def select(self, context: ControllerPlanningContext) -> ControllerPlanningDecision:
        return ControllerPlanningDecision(
            policy_id=self.policy_id,
            allowed_alternative_ids=None,
            reason="Use normal grounded-frontier bounded FABLE planning.",
        )
