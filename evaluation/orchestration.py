"""Policy selection at the planning/orchestration boundary.

The distributed orchestrator consumes complete ``PlanCandidate`` objects.  It
must not reinterpret them as planning inputs.  This coordinator therefore sits
immediately before candidate construction and controls when a baseline policy
is allowed to make a new decision.
"""

from __future__ import annotations

from enum import StrEnum

from evaluation.baselines.models import (
    BaselineDecision,
    BaselinePlanningCase,
    TaskResourcePlanningCase,
)
from evaluation.baselines.policies import BaselinePolicy
from evaluation.schemas import BaselineId


class PlanningTrigger(StrEnum):
    ADMISSION = "ADMISSION"
    RESOURCE_EPOCH = "RESOURCE_EPOCH"
    SEMANTIC_FRONTIER = "SEMANTIC_FRONTIER"


class ControlledPlanningCoordinator:
    """Apply one controlled policy at its declared decision cadence."""

    _ADMISSION_ONLY = {
        BaselineId.B1_STATIC_WHOLE_EVENT,
        BaselineId.B0_ALWAYS_ON,
        BaselineId.B1_HANDWRITTEN_STATIC,
        BaselineId.B2_STATIC_WHOLE_EVENT,
    }
    _RESOURCE_ADAPTIVE = {BaselineId.B3_TASK_RESOURCE_ADAPTIVE}
    _SEMANTIC_ADAPTIVE = {
        # B0's authored all-node pipeline is fixed with respect to resources.
        # Binding-dependent
        # demands (notably SAME_ENTITY) only become executable after semantic
        # progression, so refresh its union at those frontiers only.
        BaselineId.B0_PRODUCE_ALL,
    }
    _FRONTIER_ADAPTIVE = {
        BaselineId.B2_FRONTIER_FIXED_REALIZATION,
        BaselineId.B4_GREEDY_FRONTIER,
        BaselineId.FABLE,
        BaselineId.FABLE_NO_SHARING,
    }

    def __init__(self, policy: BaselinePolicy) -> None:
        if policy.baseline_id == BaselineId.O1_EXHAUSTIVE_ORACLE:
            raise ValueError("the exhaustive oracle is offline-only")
        if policy.baseline_id not in (
            self._ADMISSION_ONLY
            | self._RESOURCE_ADAPTIVE
            | self._SEMANTIC_ADAPTIVE
            | self._FRONTIER_ADAPTIVE
        ):
            raise ValueError(f"unsupported live planning policy: {policy.baseline_id}")
        self.policy = policy
        self._latest: dict[str, BaselineDecision] = {}

    @property
    def baseline_id(self) -> BaselineId:
        return self.policy.baseline_id

    def decide(
        self,
        case: BaselinePlanningCase,
        *,
        trigger: PlanningTrigger,
    ) -> BaselineDecision:
        """Return the current decision, recomputing only when policy permits."""

        previous = self._latest.get(case.request_id)
        if trigger != PlanningTrigger.ADMISSION and previous is None:
            raise ValueError(
                f"{case.request_id!r} has no admission-time planning decision"
            )

        should_plan = (
            trigger == PlanningTrigger.ADMISSION
            or (
                self.baseline_id in self._RESOURCE_ADAPTIVE
                and trigger == PlanningTrigger.RESOURCE_EPOCH
                and previous is not None
                and case.resource_epoch != previous.resource_epoch
            )
            or (
                self.baseline_id in self._SEMANTIC_ADAPTIVE
                and trigger == PlanningTrigger.SEMANTIC_FRONTIER
                and previous is not None
                and case.semantic_epoch != previous.semantic_epoch
            )
            or (
                self.baseline_id in self._FRONTIER_ADAPTIVE
                and trigger
                in {
                    PlanningTrigger.RESOURCE_EPOCH,
                    PlanningTrigger.SEMANTIC_FRONTIER,
                }
                and previous is not None
                and (
                    case.resource_epoch != previous.resource_epoch
                    or case.semantic_epoch != previous.semantic_epoch
                )
            )
        )
        if should_plan:
            policy_case = (
                TaskResourcePlanningCase.from_case(case)
                if self.baseline_id == BaselineId.B3_TASK_RESOURCE_ADAPTIVE
                else case
            )
            decision = self.policy.plan(policy_case)  # type: ignore[arg-type]
            if decision.baseline_id != self.baseline_id:
                raise ValueError("policy returned a decision for a different baseline")
            if decision.request_id != case.request_id:
                raise ValueError("policy returned a decision for a different request")
            self._latest[case.request_id] = decision
            return decision
        assert previous is not None
        return previous

    def forget(self, request_id: str) -> None:
        """Release coordinator state after a request reaches a terminal state."""

        self._latest.pop(request_id, None)

    def has_decision(self, request_id: str) -> bool:
        """Report whether the request still has live planning state."""

        return request_id in self._latest
