"""Scoped cancellation and semantic-checkpoint execution control."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from uuid import UUID

from fable.common.enums import CancellationScope, HypothesisLifecycle
from fable.common.schemas import PredicateDemand, PredicateResult
from fable.common.time import ensure_utc, utc_now
from fable.planning.artifact_catalog import ArtifactCatalog
from fable.semantic.models import RuntimeTransition

from .lifecycle import ProviderLifecycleManager
from .models import (
    ArtifactRetentionUpdate,
    CancellationOutcome,
    CancellationRequest,
    CheckpointControlOutcome,
    HistoricalDemand,
    ReplanRequest,
)


class CancellationManager:
    def __init__(self, lifecycle: ProviderLifecycleManager) -> None:
        self.lifecycle = lifecycle

    def cancel(
        self,
        request: CancellationRequest,
        *,
        now: datetime | None = None,
    ) -> CancellationOutcome:
        observed_now = ensure_utc(now or utc_now())
        matching = []
        for managed in self.lifecycle.active_leases:
            if managed.request_id != request.request_id:
                continue
            if request.scope == CancellationScope.TASK:
                matching.append(managed)
            elif request.scope == CancellationScope.HYPOTHESIS:
                if managed.hypothesis_id == request.hypothesis_id:
                    matching.append(managed)
            elif request.scope == CancellationScope.BRANCH:
                if (
                    managed.hypothesis_id == request.hypothesis_id
                    and managed.graph_node_id in set(request.graph_node_ids)
                ):
                    matching.append(managed)

        affected_instances = {item.lease.provider_instance_id for item in matching}
        affected_demands = {item.lease.demand_id for item in matching}
        released = []
        for demand_id in sorted(affected_demands, key=str):
            released.extend(self.lifecycle.cancel_demand(demand_id, now=observed_now))

        preserved: list[str] = []
        idle: list[str] = []
        for instance_id in sorted(affected_instances):
            instance = self.lifecycle.instances[instance_id]
            if instance.active_lease_ids:
                preserved.append(instance_id)
            else:
                idle.append(instance_id)

        cancelled_plans = tuple(
            sorted(
                (
                    plan_id
                    for plan_id, managed in self.lifecycle.plans.items()
                    if not managed.active_demand_ids and managed.cancelled_demand_ids
                ),
                key=str,
            )
        )
        return CancellationOutcome(
            request=request,
            released_lease_ids=tuple(sorted(set(released), key=str)),
            cancelled_demand_ids=tuple(sorted(affected_demands, key=str)),
            preserved_provider_instance_ids=tuple(preserved),
            idle_provider_instance_ids=tuple(idle),
            cancelled_plan_ids=cancelled_plans,
        )


class CheckpointController:
    """Bridges authoritative semantic transitions to physical execution changes."""

    def __init__(
        self,
        *,
        lifecycle: ProviderLifecycleManager,
        artifact_catalog: ArtifactCatalog,
    ) -> None:
        self.lifecycle = lifecycle
        self.artifacts = artifact_catalog
        self.cancellations = CancellationManager(lifecycle)

    def handle_predicate_result(
        self,
        *,
        result: PredicateResult,
        transition: RuntimeTransition,
        request_id: str,
        hypothesis_id: UUID,
        next_demands: Iterable[PredicateDemand] = (),
        continuation_artifact_ids: Iterable[UUID] = (),
        historical_demands: Iterable[HistoricalDemand] = (),
        hypothesis_lifecycle: HypothesisLifecycle = HypothesisLifecycle.ACTIVE,
        now: datetime | None = None,
    ) -> CheckpointControlOutcome:
        observed_now = ensure_utc(now or utc_now())
        completed = self.lifecycle.complete_demand(result.demand_id, now=observed_now)

        cancellation = None
        if transition.cancellation.node_ids:
            cancellation = self.cancellations.cancel(
                CancellationRequest(
                    scope=CancellationScope.BRANCH,
                    request_id=request_id,
                    hypothesis_id=hypothesis_id,
                    graph_node_ids=transition.cancellation.node_ids,
                    reason=transition.cancellation.reason or "semantic branch resolved",
                ),
                now=observed_now,
            )
        elif hypothesis_lifecycle in (
            HypothesisLifecycle.COMPLETED,
            HypothesisLifecycle.INVALIDATED,
            HypothesisLifecycle.EXPIRED,
        ):
            cancellation = self.cancellations.cancel(
                CancellationRequest(
                    scope=CancellationScope.HYPOTHESIS,
                    request_id=request_id,
                    hypothesis_id=hypothesis_id,
                    reason=f"hypothesis became {hypothesis_lifecycle}",
                ),
                now=observed_now,
            )

        next_demands_tuple = tuple(next_demands)
        retention_updates = self._extend_continuations(
            artifact_ids=tuple(continuation_artifact_ids),
            next_demands=next_demands_tuple,
        )
        replans = self._replan_requests(
            transition,
            request_id=request_id,
            fallback_hypothesis_id=hypothesis_id,
        )
        return CheckpointControlOutcome(
            completed_lease_ids=tuple(sorted(completed, key=str)),
            cancellation=cancellation,
            retention_updates=retention_updates,
            replan_requests=replans,
            historical_demand_ids=tuple(
                sorted((item.demand.demand_id for item in historical_demands), key=str)
            ),
        )

    def cancel_hypothesis(
        self,
        *,
        request_id: str,
        hypothesis_id: UUID,
        reason: str,
        now: datetime | None = None,
    ) -> CancellationOutcome:
        return self.cancellations.cancel(
            CancellationRequest(
                scope=CancellationScope.HYPOTHESIS,
                request_id=request_id,
                hypothesis_id=hypothesis_id,
                reason=reason,
            ),
            now=now,
        )

    def cancel_task(
        self,
        *,
        request_id: str,
        reason: str,
        now: datetime | None = None,
    ) -> CancellationOutcome:
        return self.cancellations.cancel(
            CancellationRequest(
                scope=CancellationScope.TASK,
                request_id=request_id,
                reason=reason,
            ),
            now=now,
        )

    def _extend_continuations(
        self,
        *,
        artifact_ids: tuple[UUID, ...],
        next_demands: tuple[PredicateDemand, ...],
    ) -> tuple[ArtifactRetentionUpdate, ...]:
        updates: list[ArtifactRetentionUpdate] = []
        requirements = [
            requirement
            for demand in next_demands
            for requirement in demand.continuation_requirements
        ]
        for artifact_id in artifact_ids:
            artifact = self.artifacts.get(artifact_id)
            matching = [
                requirement
                for requirement in requirements
                if requirement.artifact_type == artifact.artifact_type
                and all(
                    role in artifact.bindings
                    for role in requirement.required_bindings
                )
            ]
            if not matching:
                continue
            required_until = max(item.required_until for item in matching)
            previous = artifact.expires_at
            if previous is not None and previous >= required_until:
                continue
            updated = self.artifacts.extend_retention(
                artifact_id,
                required_until=required_until,
            )
            updates.append(
                ArtifactRetentionUpdate(
                    artifact_id=artifact_id,
                    previous_expires_at=previous,
                    new_expires_at=updated.expires_at or required_until,
                    reason="next semantic checkpoint requires this continuation artifact",
                )
            )
        return tuple(updates)

    @staticmethod
    def _replan_requests(
        transition: RuntimeTransition,
        *,
        request_id: str,
        fallback_hypothesis_id: UUID,
    ) -> tuple[ReplanRequest, ...]:
        if not transition.frontiers:
            return ()
        hypothesis_ids = transition.hypothesis_ids
        requests = []
        for index, frontier in enumerate(transition.frontiers):
            if len(hypothesis_ids) == len(transition.frontiers):
                hypothesis_id = hypothesis_ids[index]
            elif len(hypothesis_ids) == 1:
                hypothesis_id = hypothesis_ids[0]
            else:
                hypothesis_id = fallback_hypothesis_id
            requests.append(
                ReplanRequest(
                    request_id=request_id,
                    hypothesis_id=hypothesis_id,
                    frontier_id=frontier.snapshot.frontier_id,
                    checkpoint_ids=frontier.snapshot.checkpoint_ids,
                    reason="semantic checkpoint changed the active frontier",
                )
            )
        return tuple(requests)
