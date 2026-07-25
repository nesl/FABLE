"""Deterministic multi-tenant admission ordering."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from datetime import datetime, timedelta

from fable.common.time import ensure_utc, utc_now
from fable.planning.models import ExternalInputKind

from .models import EvidenceUrgency, PlanCandidate, TaskPriorityClass


class MultiTenantOrderer:
    """Priority classes with persistent round-robin fairness among tasks."""

    def __init__(self, *, near_expiry_horizon_ms: int = 5_000) -> None:
        self.near_expiry_horizon_ms = near_expiry_horizon_ms
        self._last_served_task: dict[TaskPriorityClass, str] = {}

    def evidence_urgency(
        self,
        candidate: PlanCandidate,
        *,
        now: datetime | None = None,
    ) -> EvidenceUrgency:
        observed_now = ensure_utc(now or utc_now())
        if candidate.task_policy.historical_priority_override:
            return EvidenceUrgency.LIVE_ONLY
        external_inputs = tuple(
            item
            for alternative in candidate.alternatives
            for item in alternative.external_inputs
            if item.kind != ExternalInputKind.OMITTED_OPTIONAL
        )
        if any(item.kind == ExternalInputKind.LIVE_SOURCE for item in external_inputs):
            return EvidenceUrgency.LIVE_ONLY
        expirations = [item.expires_at for item in external_inputs if item.expires_at is not None]
        if expirations and min(expirations) <= observed_now + timedelta(
            milliseconds=self.near_expiry_horizon_ms
        ):
            return EvidenceUrgency.EXPIRING_RETAINED
        return EvidenceUrgency.RETAINED

    def within_task_key(self, candidate: PlanCandidate, *, now: datetime) -> tuple:
        urgency = self.evidence_urgency(candidate, now=now)
        return (
            urgency.rank,
            candidate.latest_start,
            candidate.incremental_resource_cost_units,
            candidate.startup_cost_ms,
            candidate.transfer_bytes,
            candidate.fallback_rank,
            candidate.candidate_id,
        )

    def order(
        self,
        candidates: Iterable[PlanCandidate],
        *,
        now: datetime | None = None,
    ) -> tuple[PlanCandidate, ...]:
        observed_now = ensure_utc(now or utc_now())
        by_priority_task: dict[
            TaskPriorityClass, dict[str, list[PlanCandidate]]
        ] = defaultdict(lambda: defaultdict(list))
        for candidate in candidates:
            by_priority_task[candidate.task_policy.priority_class][candidate.request_id].append(
                candidate
            )

        ordered: list[PlanCandidate] = []
        for priority in sorted(by_priority_task, key=lambda item: item.rank):
            task_queues = {
                task_id: deque(sorted(items, key=lambda item: self.within_task_key(item, now=observed_now)))
                for task_id, items in by_priority_task[priority].items()
            }
            task_ids = sorted(task_queues)
            if not task_ids:
                continue
            last = self._last_served_task.get(priority)
            if last in task_ids:
                start = (task_ids.index(last) + 1) % len(task_ids)
                task_ids = task_ids[start:] + task_ids[:start]
            last_served: str | None = None
            while any(task_queues[task_id] for task_id in task_ids):
                for task_id in task_ids:
                    if not task_queues[task_id]:
                        continue
                    ordered.append(task_queues[task_id].popleft())
                    last_served = task_id
            if last_served is not None:
                self._last_served_task[priority] = last_served
        return tuple(ordered)
