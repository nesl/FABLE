"""Provider-token sharing, leases, and lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from collections.abc import Callable
from uuid import UUID

from fable.common.enums import PlanStatus, ProviderLeaseStatus
from fable.common.ids import deterministic_id
from fable.common.schemas import PlanStep, ProviderLease, ResourceReservation
from fable.common.time import ensure_utc, utc_now
from fable.planning.provider_registry import ProviderRegistry

from .capacity import CapacityLedger
from .models import (
    ManagedLease,
    ManagedPlan,
    PlanCandidate,
    ProviderInstanceLifecycle,
    ProviderInstanceRecord,
    ProviderShareKey,
)


class ProviderLifecycleError(ValueError):
    """Raised for invalid provider lifecycle or lease operations."""


@dataclass(frozen=True)
class StepLeaseIntent:
    """Scheduler-ready intent derived only from an ``ExecutionPlan`` step."""

    demand_id: UUID
    request_id: str
    hypothesis_id: UUID
    checkpoint_id: UUID
    graph_node_id: str
    cancellation_scope: object
    plan_step: PlanStep
    share_key: ProviderShareKey
    reservation: ResourceReservation


@dataclass(frozen=True)
class AttachResult:
    lease_ids: tuple[UUID, ...]
    created_provider_instance_ids: tuple[str, ...]
    reused_provider_instance_ids: tuple[str, ...]
    incremental_reservations: tuple[ResourceReservation, ...]


class ProviderLifecycleManager:
    """Owns provider instances and demand-scoped leases.

    One provider instance is keyed by an exact compatibility token.  Semantic
    hypotheses never share bindings or windows; they only attach independent
    leases to a compatible physical provider token.
    """

    def __init__(
        self,
        *,
        provider_registry: ProviderRegistry,
        capacity: CapacityLedger,
        idle_grace_ms: int = 2_000,
        capacity_group_resolver: Callable[
            [str, str, ResourceReservation, str], tuple[str, ResourceReservation]
        ] | None = None,
    ) -> None:
        self.providers = provider_registry
        self.capacity = capacity
        self.idle_grace_ms = idle_grace_ms
        self.capacity_group_resolver = capacity_group_resolver
        self.instances: dict[str, ProviderInstanceRecord] = {}
        self.leases: dict[UUID, ManagedLease] = {}
        self.plans: dict[UUID, ManagedPlan] = {}
        self._reusable_by_key: dict[str, str] = {}
        self._generation_by_key: dict[str, int] = {}
        self._lease_idempotency: dict[tuple[UUID, UUID, str], UUID] = {}
        self._capacity_group_refcounts: dict[str, int] = {}

    @property
    def active_instances(self) -> tuple[ProviderInstanceRecord, ...]:
        return tuple(
            sorted(
                (
                    instance
                    for instance in self.instances.values()
                    if instance.lifecycle
                    not in (ProviderInstanceLifecycle.DRAINING, ProviderInstanceLifecycle.FAILED)
                ),
                key=lambda item: item.provider_instance_id,
            )
        )

    @property
    def active_leases(self) -> tuple[ManagedLease, ...]:
        return tuple(
            sorted(
                (
                    lease
                    for lease in self.leases.values()
                    if lease.lease.status
                    not in (
                        ProviderLeaseStatus.RELEASED,
                        ProviderLeaseStatus.FAILED,
                        ProviderLeaseStatus.EXPIRED,
                    )
                ),
                key=lambda item: str(item.lease.lease_id),
            )
        )

    def preview_candidate(
        self,
        candidate: PlanCandidate,
    ) -> tuple[StepLeaseIntent, ...]:
        """Translate a concrete execution plan into provider lease intents.

        This boundary intentionally consumes only ``ExecutionPlan`` and
        ``PredicateDemand`` contracts.  Planner-internal alternatives and
        placements are not required by scheduling.
        """

        demand_map = {demand.demand_id: demand for demand in candidate.demands}
        sole_demand = candidate.demands[0] if len(candidate.demands) == 1 else None
        intents: list[StepLeaseIntent] = []
        for plan_step in sorted(candidate.plan.steps, key=lambda item: item.step_id):
            demand_id = plan_step.demand_id
            if demand_id is None:
                if sole_demand is None:
                    raise ProviderLifecycleError(
                        f"step {plan_step.step_id} is not scoped to a demand"
                    )
                demand = sole_demand
            else:
                try:
                    demand = demand_map[demand_id]
                except KeyError as exc:
                    raise ProviderLifecycleError(
                        f"step {plan_step.step_id} references unknown demand {demand_id}"
                    ) from exc
            key = self._share_key(
                candidate=candidate,
                demand=demand,
                plan_step=plan_step,
            )
            intents.append(
                StepLeaseIntent(
                    demand_id=demand.demand_id,
                    request_id=demand.request_id,
                    hypothesis_id=demand.hypothesis_id,
                    checkpoint_id=demand.checkpoint_id,
                    graph_node_id=demand.graph_node_id,
                    cancellation_scope=demand.cancellation_scope,
                    plan_step=plan_step,
                    share_key=key,
                    reservation=ResourceReservation(
                        node_id=plan_step.node_id,
                        cpu_cores=plan_step.cpu_cores,
                        memory_mb=plan_step.memory_mb,
                        gpu_memory_mb=plan_step.gpu_memory_mb,
                        network_bytes=plan_step.estimated_transfer_bytes,
                    ),
                )
            )
        return tuple(intents)

    def preview_incremental_reservations(
        self,
        candidate: PlanCandidate,
    ) -> tuple[tuple[str, ResourceReservation], ...]:
        result: dict[str, ResourceReservation] = {}
        for intent in self.preview_candidate(candidate):
            key_id = intent.share_key.key_id
            assert key_id is not None
            if self._find_reusable(intent.share_key) is not None:
                continue
            owner_id, reservation = self._capacity_group(
                intent.plan_step.node_id,
                intent.plan_step.provider_id,
                intent.reservation,
                fallback_owner_id=key_id,
            )
            # A logical provider token and, optionally, an entire physical worker
            # bundle reserve capacity once.
            if self._capacity_group_refcounts.get(owner_id, 0) > 0:
                continue
            if owner_id not in result:
                result[owner_id] = reservation
        return tuple(sorted(result.items(), key=lambda item: item[0]))

    def attach_candidate(
        self,
        candidate: PlanCandidate,
        *,
        now: datetime | None = None,
    ) -> AttachResult:
        observed_now = ensure_utc(now or utc_now())
        intents = self.preview_candidate(candidate)
        reservation_preview = self.preview_incremental_reservations(candidate)
        feasible, reason = self.capacity.can_reserve(reservation_preview)
        if not feasible:
            raise ProviderLifecycleError(reason)

        created: set[str] = set()
        reused: set[str] = set()
        incremental: list[ResourceReservation] = []
        attached: list[UUID] = []
        newly_created_instances: list[str] = []
        try:
            for intent in intents:
                instance = self._find_reusable(intent.share_key)
                if instance is None:
                    instance = self._create_instance(
                        intent.share_key,
                        intent.reservation,
                        now=observed_now,
                    )
                    created.add(instance.provider_instance_id)
                    newly_created_instances.append(instance.provider_instance_id)
                    incremental.append(instance.reservation)
                else:
                    reused.add(instance.provider_instance_id)
                    if instance.lifecycle == ProviderInstanceLifecycle.IDLE_LEASE:
                        self._transition(
                            instance,
                            ProviderInstanceLifecycle.ACTIVE,
                            now=observed_now,
                        )
                        instance.idle_until = None

                lease = self._attach_lease(
                    candidate=candidate,
                    intent=intent,
                    instance=instance,
                    now=observed_now,
                )
                attached.append(lease.lease.lease_id)

            existing_plan = self.plans.get(candidate.plan.plan_id)
            if existing_plan is None:
                admitted_plan = candidate.plan.model_copy(update={"status": PlanStatus.ADMITTED})
                self.plans[admitted_plan.plan_id] = ManagedPlan(
                    candidate_id=candidate.candidate_id or "",
                    plan=admitted_plan,
                    active_demand_ids=admitted_plan.demand_ids,
                )
            return AttachResult(
                lease_ids=tuple(attached),
                created_provider_instance_ids=tuple(sorted(created)),
                reused_provider_instance_ids=tuple(sorted(reused - created)),
                incremental_reservations=tuple(
                    sorted(incremental, key=lambda item: (item.node_id, item.cpu_cores, item.memory_mb))
                ),
            )
        except Exception:
            for lease_id in reversed(attached):
                self.release_lease(lease_id, now=observed_now, immediate=True)
            for instance_id in newly_created_instances:
                instance = self.instances.get(instance_id)
                if instance is not None and not instance.active_lease_ids:
                    self._retire_immediately(instance, now=observed_now)
            raise

    def mark_ready(self, provider_instance_id: str, *, now: datetime | None = None) -> None:
        observed_now = ensure_utc(now or utc_now())
        instance = self._instance(provider_instance_id)
        if instance.lifecycle not in (
            ProviderInstanceLifecycle.WARMING,
            ProviderInstanceLifecycle.ACTIVE,
        ):
            raise ProviderLifecycleError(
                f"provider {provider_instance_id} cannot become active from {instance.lifecycle}"
            )
        self._transition(instance, ProviderInstanceLifecycle.ACTIVE, now=observed_now)
        for lease_id in instance.active_lease_ids:
            managed = self.leases[lease_id]
            if managed.lease.status in (
                ProviderLeaseStatus.REQUESTED,
                ProviderLeaseStatus.STARTING,
            ):
                managed.lease.status = ProviderLeaseStatus.ACTIVE

    def complete_demand(self, demand_id: UUID, *, now: datetime | None = None) -> tuple[UUID, ...]:
        observed_now = ensure_utc(now or utc_now())
        released = []
        for managed in list(self.active_leases):
            if managed.lease.demand_id == demand_id:
                self.release_lease(managed.lease.lease_id, now=observed_now)
                released.append(managed.lease.lease_id)
        self._move_plan_demand(demand_id, completed=True)
        return tuple(released)

    def release_lease(
        self,
        lease_id: UUID,
        *,
        now: datetime | None = None,
        immediate: bool = False,
    ) -> str | None:
        observed_now = ensure_utc(now or utc_now())
        managed = self.leases.get(lease_id)
        if managed is None:
            return None
        if managed.lease.status in (
            ProviderLeaseStatus.RELEASED,
            ProviderLeaseStatus.FAILED,
            ProviderLeaseStatus.EXPIRED,
        ):
            return managed.lease.provider_instance_id
        managed.lease.status = ProviderLeaseStatus.RELEASED
        instance = self._instance(managed.lease.provider_instance_id)
        instance.active_lease_ids = tuple(
            item for item in instance.active_lease_ids if item != lease_id
        )
        instance.updated_at = observed_now
        if not instance.active_lease_ids:
            if immediate:
                self._retire_immediately(instance, now=observed_now)
            else:
                instance.idle_until = observed_now + timedelta(milliseconds=self.idle_grace_ms)
                self._transition(
                    instance,
                    ProviderInstanceLifecycle.IDLE_LEASE,
                    now=observed_now,
                )
        return instance.provider_instance_id

    def cancel_demand(self, demand_id: UUID, *, now: datetime | None = None) -> tuple[UUID, ...]:
        observed_now = ensure_utc(now or utc_now())
        released = []
        for managed in list(self.active_leases):
            if managed.lease.demand_id != demand_id:
                continue
            self.release_lease(managed.lease.lease_id, now=observed_now)
            released.append(managed.lease.lease_id)
        self._move_plan_demand(demand_id, completed=False)
        return tuple(released)

    def tick(self, *, now: datetime | None = None) -> tuple[str, ...]:
        observed_now = ensure_utc(now or utc_now())
        draining: list[str] = []
        for instance in self.instances.values():
            if (
                instance.lifecycle == ProviderInstanceLifecycle.IDLE_LEASE
                and instance.idle_until is not None
                and instance.idle_until <= observed_now
            ):
                self._transition(
                    instance,
                    ProviderInstanceLifecycle.DRAINING,
                    now=observed_now,
                )
                self._reusable_by_key.pop(instance.share_key.key_id or "", None)
                self._release_instance_capacity(instance)
                draining.append(instance.provider_instance_id)
        return tuple(sorted(draining))

    def mark_failed(
        self,
        provider_instance_id: str,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> tuple[UUID, ...]:
        observed_now = ensure_utc(now or utc_now())
        instance = self._instance(provider_instance_id)
        affected_demands: set[UUID] = set()
        for lease_id in instance.active_lease_ids:
            managed = self.leases[lease_id]
            affected_demands.add(managed.lease.demand_id)
            managed.lease.status = ProviderLeaseStatus.FAILED
        instance.active_lease_ids = ()
        instance.failure_reason = reason
        self._transition(instance, ProviderInstanceLifecycle.FAILED, now=observed_now)
        self._reusable_by_key.pop(instance.share_key.key_id or "", None)
        self._release_instance_capacity(instance)
        return tuple(sorted(affected_demands, key=str))

    def leases_for_instance(self, provider_instance_id: str) -> tuple[ManagedLease, ...]:
        instance = self._instance(provider_instance_id)
        return tuple(self.leases[item] for item in instance.active_lease_ids)

    def _attach_lease(
        self,
        *,
        candidate: PlanCandidate,
        intent: StepLeaseIntent,
        instance: ProviderInstanceRecord,
        now: datetime,
    ) -> ManagedLease:
        idempotency_key = (intent.demand_id, candidate.plan.plan_id, intent.plan_step.step_id)
        existing_id = self._lease_idempotency.get(idempotency_key)
        if existing_id is not None:
            return self.leases[existing_id]
        demand = next(item for item in candidate.demands if item.demand_id == intent.demand_id)
        expires_at = demand.deadline.latest_useful_completion
        if candidate.plan.expires_at is not None:
            expires_at = min(expires_at, candidate.plan.expires_at)
        if expires_at <= now:
            raise ProviderLifecycleError(f"demand {demand.demand_id} lease is already expired")
        status = (
            ProviderLeaseStatus.ACTIVE
            if instance.lifecycle == ProviderInstanceLifecycle.ACTIVE
            else ProviderLeaseStatus.STARTING
        )
        lease = ProviderLease(
            provider_instance_id=instance.provider_instance_id,
            provider_id=intent.plan_step.provider_id,
            provider_contract_version=intent.share_key.provider_contract_version,
            demand_id=intent.demand_id,
            plan_id=candidate.plan.plan_id,
            node_id=intent.plan_step.node_id,
            configuration_hash=intent.share_key.configuration_hash,
            status=status,
            starts_at=now,
            expires_at=expires_at,
        )
        managed = ManagedLease(
            lease=lease,
            request_id=intent.request_id,
            hypothesis_id=intent.hypothesis_id,
            checkpoint_id=intent.checkpoint_id,
            graph_node_id=intent.graph_node_id,
            step_id=intent.plan_step.step_id,
            share_key_id=intent.share_key.key_id or "",
            cancellation_scope=intent.cancellation_scope,
            execution_mode=intent.plan_step.execution_mode,
            created_at=now,
        )
        self.leases[lease.lease_id] = managed
        self._lease_idempotency[idempotency_key] = lease.lease_id
        instance.active_lease_ids = tuple(
            sorted((*instance.active_lease_ids, lease.lease_id), key=str)
        )
        instance.updated_at = now
        if instance.lifecycle == ProviderInstanceLifecycle.COLD:
            self._transition(instance, ProviderInstanceLifecycle.WARMING, now=now)
        return managed

    def _create_instance(
        self,
        share_key: ProviderShareKey,
        reservation: ResourceReservation,
        *,
        now: datetime,
    ) -> ProviderInstanceRecord:
        key_id = share_key.key_id or ""
        generation = self._generation_by_key.get(key_id, 0) + 1
        self._generation_by_key[key_id] = generation
        provider_instance_id = deterministic_id(
            "pinst",
            {"share_key": key_id, "generation": generation},
            length=32,
        )
        capacity_owner_id, capacity_reservation = self._capacity_group(
            share_key.node_id,
            share_key.provider_id,
            reservation,
            fallback_owner_id=provider_instance_id,
        )
        if self._capacity_group_refcounts.get(capacity_owner_id, 0) == 0:
            self.capacity.reserve(capacity_owner_id, capacity_reservation)
        self._capacity_group_refcounts[capacity_owner_id] = (
            self._capacity_group_refcounts.get(capacity_owner_id, 0) + 1
        )
        instance = ProviderInstanceRecord(
            provider_instance_id=provider_instance_id,
            share_key=share_key,
            lifecycle=ProviderInstanceLifecycle.COLD,
            reservation=reservation,
            capacity_owner_id=capacity_owner_id,
            capacity_reservation=capacity_reservation,
            created_at=now,
            updated_at=now,
            lifecycle_history=(ProviderInstanceLifecycle.COLD,),
        )
        self.instances[provider_instance_id] = instance
        self._reusable_by_key[key_id] = provider_instance_id
        return instance

    def _find_reusable(self, share_key: ProviderShareKey) -> ProviderInstanceRecord | None:
        instance_id = self._reusable_by_key.get(share_key.key_id or "")
        if instance_id is None:
            return None
        instance = self.instances[instance_id]
        if instance.lifecycle in (
            ProviderInstanceLifecycle.COLD,
            ProviderInstanceLifecycle.WARMING,
            ProviderInstanceLifecycle.ACTIVE,
            ProviderInstanceLifecycle.IDLE_LEASE,
        ):
            return instance
        return None

    def _share_key(
        self,
        *,
        candidate: PlanCandidate,
        demand,
        plan_step: PlanStep,
    ) -> ProviderShareKey:
        provider = self.providers.provider(plan_step.provider_id)
        configuration_hash = deterministic_id(
            "config",
            {
                "provider_id": plan_step.provider_id,
                "parameters": plan_step.parameters,
            },
            length=32,
        )
        policy_hash = deterministic_id(
            "policy",
            demand.hard_constraints,
            length=32,
        )
        directly_semantic = (
            demand.semantic_predicate.predicate_id
            in provider.semantic_capabilities.predicate_ids
        )
        binding_signature = (
            tuple(sorted(demand.bound_roles.items())) if directly_semantic else ()
        )
        discriminator = None
        if not provider.execution_capabilities.supports_shared_execution:
            discriminator = f"{demand.demand_id}:{plan_step.step_id}"
        return ProviderShareKey(
            provider_id=plan_step.provider_id,
            provider_contract_version=provider.contract_version,
            node_id=plan_step.node_id,
            configuration_hash=configuration_hash,
            source_signature=plan_step.source_signature,
            input_artifact_ids=tuple(sorted(plan_step.input_artifact_ids, key=str)),
            input_data_types=tuple(sorted(plan_step.input_data_types)),
            output_data_types=tuple(sorted(plan_step.output_data_types)),
            event_time_interval=demand.event_time_interval,
            execution_mode=plan_step.execution_mode,
            policy_hash=policy_hash,
            semantic_binding_signature=binding_signature,
            nonshareable_discriminator=discriminator,
        )

    def _transition(
        self,
        instance: ProviderInstanceRecord,
        lifecycle: ProviderInstanceLifecycle,
        *,
        now: datetime,
    ) -> None:
        if instance.lifecycle == lifecycle:
            instance.updated_at = now
            return
        history = (*instance.lifecycle_history, lifecycle)
        # These fields form one invariant and must be updated atomically; ordinary
        # assignment validation would observe an invalid intermediate state.
        object.__setattr__(instance, "lifecycle", lifecycle)
        object.__setattr__(instance, "lifecycle_history", history)
        object.__setattr__(instance, "updated_at", now)

    def _retire_immediately(self, instance: ProviderInstanceRecord, *, now: datetime) -> None:
        self._transition(instance, ProviderInstanceLifecycle.DRAINING, now=now)
        self._reusable_by_key.pop(instance.share_key.key_id or "", None)
        self._release_instance_capacity(instance)

    def _capacity_group(
        self,
        node_id: str,
        provider_id: str,
        reservation: ResourceReservation,
        *,
        fallback_owner_id: str,
    ) -> tuple[str, ResourceReservation]:
        if self.capacity_group_resolver is None:
            return fallback_owner_id, reservation
        owner_id, grouped = self.capacity_group_resolver(
            node_id, provider_id, reservation, fallback_owner_id
        )
        if not owner_id:
            raise ProviderLifecycleError("capacity group resolver returned an empty owner ID")
        if grouped.node_id != node_id:
            raise ProviderLifecycleError("capacity group reservation must stay on the provider node")
        return owner_id, grouped

    def _release_instance_capacity(self, instance: ProviderInstanceRecord) -> None:
        owner_id = instance.capacity_owner_id or instance.provider_instance_id
        count = self._capacity_group_refcounts.get(owner_id, 0)
        if count <= 1:
            self._capacity_group_refcounts.pop(owner_id, None)
            self.capacity.release(owner_id)
        else:
            self._capacity_group_refcounts[owner_id] = count - 1

    def _instance(self, provider_instance_id: str) -> ProviderInstanceRecord:
        try:
            return self.instances[provider_instance_id]
        except KeyError as exc:
            raise ProviderLifecycleError(f"unknown provider instance {provider_instance_id}") from exc

    def _move_plan_demand(self, demand_id: UUID, *, completed: bool) -> None:
        for plan_id, managed in list(self.plans.items()):
            if demand_id not in managed.active_demand_ids:
                continue
            active = tuple(item for item in managed.active_demand_ids if item != demand_id)
            cancelled = managed.cancelled_demand_ids
            completed_ids = managed.completed_demand_ids
            if completed:
                completed_ids = tuple(sorted((*completed_ids, demand_id), key=str))
            else:
                cancelled = tuple(sorted((*cancelled, demand_id), key=str))
            status = managed.plan.status
            if not active:
                if completed_ids and not cancelled:
                    status = PlanStatus.COMPLETED
                elif cancelled:
                    status = PlanStatus.CANCELLED
            updated_plan = managed.plan.model_copy(update={"status": status})
            self.plans[plan_id] = ManagedPlan(
                candidate_id=managed.candidate_id,
                plan=updated_plan,
                active_demand_ids=active,
                cancelled_demand_ids=cancelled,
                completed_demand_ids=completed_ids,
            )
