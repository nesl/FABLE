"""Safe checkpoint-advisor interface.

An LLM may order already-authored, already-feasible branches.  It cannot create
new graph nodes, satisfy predicates, override source coverage, or select an
otherwise infeasible provider plan.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from fable.common.time import ensure_utc, utc_now

from .models import (
    BranchPriorityAdjustment,
    CheckpointAdvisorHint,
    CheckpointAdvisorRequest,
    ValidatedCheckpointHint,
)


class CheckpointAdvisor(Protocol):
    def advise(self, request: CheckpointAdvisorRequest) -> CheckpointAdvisorHint:
        """Return a typed hint; the deterministic validator remains authoritative."""


class CheckpointHintValidator:
    def validate(
        self,
        *,
        request: CheckpointAdvisorRequest,
        hint: CheckpointAdvisorHint,
        now: datetime | None = None,
    ) -> ValidatedCheckpointHint:
        observed_now = ensure_utc(now or utc_now())
        if hint.expires_at <= observed_now:
            raise ValueError("checkpoint hint has expired")
        eligible = set(request.eligible_branch_ids)
        replayable = set(request.replayable_branch_ids)
        ignored: list[str] = []
        branch_order = []
        for branch_id in hint.ordered_branch_ids:
            if branch_id not in eligible:
                ignored.append(f"unknown or inactive branch ignored: {branch_id}")
                continue
            if replayable and branch_id not in replayable:
                ignored.append(
                    f"live/non-replayable branch cannot be deferred by LLM ordering: {branch_id}"
                )
                continue
            branch_order.append(branch_id)
        adjustments: list[BranchPriorityAdjustment] = []
        for adjustment in hint.priority_adjustments:
            if adjustment.branch_id not in eligible:
                ignored.append(
                    f"adjustment for unknown or inactive branch ignored: {adjustment.branch_id}"
                )
                continue
            if replayable and adjustment.branch_id not in replayable:
                ignored.append(
                    f"adjustment for live/non-replayable branch ignored: {adjustment.branch_id}"
                )
                continue
            adjustments.append(adjustment)
        return ValidatedCheckpointHint(
            hint=hint,
            accepted_branch_order=tuple(branch_order),
            accepted_adjustments=tuple(adjustments),
            ignored_reasons=tuple(ignored),
        )


class NoOpCheckpointAdvisor:
    """Default implementation used when no LLM endpoint is configured."""

    def advise(self, request: CheckpointAdvisorRequest) -> CheckpointAdvisorHint:
        from datetime import timedelta

        return CheckpointAdvisorHint(
            ordered_branch_ids=(),
            priority_adjustments=(),
            explanation="No LLM checkpoint advisor is configured.",
            evidence_refs=(),
            expires_at=request.requested_at + timedelta(seconds=30),
        )
