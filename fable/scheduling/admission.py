"""Coordinate whether ordered plan candidates can be admitted now.

Ordering decides which candidate is considered first; lifecycle previews
sharing and incremental reservations; CapacityLedger provides the hard capacity
answer. Actual workers are started later by distributed NodeAgent execution.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from pydantic import Field

from fable.common.base import FableModel
from fable.common.time import ensure_utc, utc_now

from .lifecycle import ProviderLifecycleError, ProviderLifecycleManager
from .models import (
    AdmissionBatchResult,
    AdmissionDecision,
    AdmissionRecord,
    PlanCandidate,
)
from .order import MultiTenantOrderer


class AdmissionConfig(FableModel):
    near_expiry_horizon_ms: int = Field(default=5_000, ge=0)
    defer_on_capacity: bool = True


class MultiTenantScheduler:
    """Admits checkpoint plans without conflating them with event hypotheses."""

    def __init__(
        self,
        *,
        lifecycle: ProviderLifecycleManager,
        config: AdmissionConfig | None = None,
        orderer: MultiTenantOrderer | None = None,
    ) -> None:
        self.lifecycle = lifecycle
        self.config = config or AdmissionConfig()
        self.orderer = orderer or MultiTenantOrderer(
            near_expiry_horizon_ms=self.config.near_expiry_horizon_ms
        )

    def admit(
        self,
        candidates: Iterable[PlanCandidate],
        *,
        now: datetime | None = None,
    ) -> AdmissionBatchResult:
        """Order candidates and atomically attach leases for those feasible now."""

        observed_now = ensure_utc(now or utc_now())
        # Ordering is policy; capacity remains a hard gate evaluated later for
        # each candidate against reservations admitted earlier in this batch.
        ordered = self.orderer.order(tuple(candidates), now=observed_now)
        records: list[AdmissionRecord] = []
        admitted_plan_ids = []
        resource_pressure = False
        admitted_obligation_keys: set[tuple] = set()

        for order_rank, candidate in enumerate(ordered):
            # Urgency is recorded even for rejected work so operators can
            # reconstruct why ordering and admission differed.
            urgency = self.orderer.evidence_urgency(candidate, now=observed_now)
            base = {
                "candidate_id": candidate.candidate_id or "",
                "evidence_urgency": urgency,
                "latest_start": candidate.latest_start,
                "order_rank": order_rank,
            }
            obligation_key = (
                candidate.request_id,
                tuple(sorted(str(item) for item in candidate.hypothesis_ids)),
                tuple(sorted(str(item) for item in candidate.plan.demand_ids)),
                str(candidate.plan.checkpoint_id),
            )
            # Primary and fallback candidates cover the same semantic
            # obligation. At most one may consume capacity in a batch.
            if obligation_key in admitted_obligation_keys:
                records.append(
                    AdmissionRecord(
                        **base,
                        decision=AdmissionDecision.DEFERRED,
                        reason="a higher-ranked primary/fallback plan already covers these demands",
                    )
                )
                continue
            if candidate.earliest_deadline <= observed_now:
                records.append(
                    AdmissionRecord(
                        **base,
                        decision=AdmissionDecision.EXPIRED,
                        reason="latest useful completion has passed",
                    )
                )
                continue
            if candidate.latest_start < observed_now:
                records.append(
                    AdmissionRecord(
                        **base,
                        decision=AdmissionDecision.REJECTED,
                        reason="predicted checkpoint completion cannot meet the latest useful completion",
                    )
                )
                continue

            # Preview accounts for provider sharing: an already-running
            # compatible instance contributes no new compute reservation.
            incremental = self.lifecycle.preview_incremental_reservations(candidate)
            feasible, reason = self.lifecycle.capacity.can_reserve(incremental)
            if not feasible:
                resource_pressure = True
                decision = (
                    AdmissionDecision.DEFERRED
                    if self.config.defer_on_capacity
                    else AdmissionDecision.REJECTED
                )
                records.append(
                    AdmissionRecord(
                        **base,
                        decision=decision,
                        reason=reason,
                        incremental_reservations=tuple(item[1] for item in incremental),
                    )
                )
                continue

            try:
                # `attach_candidate` rechecks capacity and performs rollback on
                # partial failure; the preview above is advisory for reporting.
                attached = self.lifecycle.attach_candidate(candidate, now=observed_now)
            except ProviderLifecycleError as exc:
                records.append(
                    AdmissionRecord(
                        **base,
                        decision=AdmissionDecision.REJECTED,
                        reason=str(exc),
                    )
                )
                continue
            admitted_plan_ids.append(candidate.plan.plan_id)
            admitted_obligation_keys.add(obligation_key)
            records.append(
                AdmissionRecord(
                    **base,
                    decision=AdmissionDecision.ADMITTED,
                    reason="capacity reserved and provider leases attached",
                    plan_id=candidate.plan.plan_id,
                    lease_ids=attached.lease_ids,
                    created_provider_instance_ids=attached.created_provider_instance_ids,
                    reused_provider_instance_ids=attached.reused_provider_instance_ids,
                    incremental_reservations=attached.incremental_reservations,
                )
            )

        # The batch result is an immutable audit record. Physical worker start
        # happens later after the orchestrator dispatches activation commands.
        return AdmissionBatchResult(
            ordered_candidate_ids=tuple(candidate.candidate_id or "" for candidate in ordered),
            records=tuple(records),
            resource_pressure=resource_pressure,
            admitted_plan_ids=tuple(admitted_plan_ids),
        )
