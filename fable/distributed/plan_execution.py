"""Dependency-aware execution state for admitted physical plans.

The planner is allowed to produce multi-step provider DAGs.  This tracker keeps
that representation honest at deployment time: cross-worker downstream steps
are not activated until an upstream step reports a produced artifact/result.
Logical steps inside one physical worker are co-activated because the worker
owns their internal dataflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from fable.scheduling.models import PlanCandidate

from .config import ProviderRuntimeResolver


@dataclass
class _PlanExecutionState:
    candidate: PlanCandidate
    worker_by_step: dict[str, str]
    dispatched: set[str] = field(default_factory=set)
    completed: set[str] = field(default_factory=set)


class PlanExecutionTracker:
    def __init__(self, runtime_resolver: ProviderRuntimeResolver) -> None:
        self.runtime_resolver = runtime_resolver
        self._plans: dict[UUID, _PlanExecutionState] = {}

    def register(self, candidate: PlanCandidate) -> tuple[str, ...]:
        state = self._plans.get(candidate.plan.plan_id)
        if state is None:
            workers = {
                step.step_id: self.runtime_resolver.worker_key(step.node_id, step.provider_id)
                for step in candidate.plan.steps
            }
            state = _PlanExecutionState(candidate=candidate, worker_by_step=workers)
            self._plans[candidate.plan.plan_id] = state
        return self._claim_ready(state)

    def complete_step(self, plan_id: UUID, step_id: str) -> tuple[str, ...]:
        state = self._plans.get(plan_id)
        if state is None or step_id not in state.worker_by_step:
            return ()
        # Completion of one capability means the physical worker has satisfied
        # all co-activated internal dependencies for this plan.
        worker = state.worker_by_step[step_id]
        state.completed.update(
            item for item, item_worker in state.worker_by_step.items() if item_worker == worker
        )
        return self._claim_ready(state)

    def cancel_plan(self, plan_id: UUID) -> None:
        self._plans.pop(plan_id, None)

    def snapshot(self, plan_id: UUID) -> dict[str, object] | None:
        state = self._plans.get(plan_id)
        if state is None:
            return None
        return {
            "plan_id": str(plan_id),
            "dispatched_step_ids": sorted(state.dispatched),
            "completed_step_ids": sorted(state.completed),
            "worker_by_step": dict(sorted(state.worker_by_step.items())),
        }

    def _claim_ready(self, state: _PlanExecutionState) -> tuple[str, ...]:
        by_id = {step.step_id: step for step in state.candidate.plan.steps}

        def dependency_ready(step, dependency_id: str) -> bool:
            if dependency_id in state.completed:
                return True
            dependency = by_id.get(dependency_id)
            if dependency is None:
                return True
            if state.worker_by_step.get(dependency_id) == state.worker_by_step[step.step_id]:
                return True
            # A typed broker edge is a streaming dataflow contract, not a
            # batch-completion barrier.  Start the subscriber with the
            # publisher so it cannot miss the first artifact.  Cross-worker
            # edges without such a contract remain gated until an explicit
            # artifact/result completion is observed.
            common_types = set(dependency.output_data_types) & set(step.input_data_types)
            return any(
                self.runtime_resolver.supports_artifact_topic_transfer(
                    source_node_id=dependency.node_id,
                    source_provider_id=dependency.provider_id,
                    target_node_id=step.node_id,
                    target_provider_id=step.provider_id,
                    data_type=data_type,
                )
                for data_type in common_types
            )

        ready: list[str] = []
        progress = True
        while progress:
            progress = False
            for step in state.candidate.plan.steps:
                if step.step_id in state.dispatched:
                    continue
                worker = state.worker_by_step[step.step_id]
                dependencies_ready = all(
                    dependency_ready(step, dependency)
                    for dependency in step.depends_on_step_ids
                    if dependency in by_id
                )
                if not dependencies_ready:
                    continue
                # Co-activate the entire worker group that is reachable without a
                # cross-worker predecessor. This models one warm container exposing
                # several logical provider capabilities.
                group = [
                    member
                    for member in state.candidate.plan.steps
                    if state.worker_by_step[member.step_id] == worker
                    and member.step_id not in state.dispatched
                    and all(
                        dependency_ready(member, dependency)
                        for dependency in member.depends_on_step_ids
                        if dependency in by_id
                    )
                ]
                for member in group:
                    state.dispatched.add(member.step_id)
                    ready.append(member.step_id)
                if group:
                    progress = True
        return tuple(ready)
