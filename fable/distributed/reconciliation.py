"""Orchestrator restart reconstruction and agent-state reconciliation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any
from uuid import UUID

from fable.common.enums import ProviderLeaseStatus
from fable.common.schemas import NodeHeartbeat
from fable.common.time import ensure_utc, utc_now
from fable.scheduling.lifecycle import ProviderLifecycleManager
from fable.scheduling.models import (
    ManagedLease,
    ManagedPlan,
    ProviderInstanceLifecycle,
    ProviderInstanceRecord,
)

from .models import (
    ControlEvent,
    ControlEventType,
    ReconciliationReport,
)
from .persistence import StateStore


class RuntimeReconciler:
    """Rebuilds Phase-5 in-memory indexes from durable compact records."""

    def __init__(self, *, store: StateStore, lifecycle: ProviderLifecycleManager) -> None:
        self.store = store
        self.lifecycle = lifecycle

    def restore(
        self,
        *,
        heartbeats: tuple[NodeHeartbeat, ...] = (),
        require_heartbeat_for_active_instances: bool = False,
        now: datetime | None = None,
    ) -> ReconciliationReport:
        observed_now = ensure_utc(now or utc_now())
        instances = self.store.list("provider_instances", ProviderInstanceRecord)
        leases = self.store.list("leases", ManagedLease)
        plans = self.store.list("plans", ManagedPlan)

        self.lifecycle.instances.clear()
        self.lifecycle.leases.clear()
        self.lifecycle.plans.clear()
        self.lifecycle._reusable_by_key.clear()  # type: ignore[attr-defined]
        self.lifecycle._generation_by_key.clear()  # type: ignore[attr-defined]
        self.lifecycle._lease_idempotency.clear()  # type: ignore[attr-defined]
        for owner_id, _reservation in tuple(self.lifecycle.capacity.reservations):
            self.lifecycle.capacity.release(owner_id)

        failed: list[str] = []
        restored_instances: list[str] = []
        generation_counts: Counter[str] = Counter()
        for instance in instances:
            instance_copy = instance.model_copy(deep=True)
            active_lifecycle = instance_copy.lifecycle not in (
                ProviderInstanceLifecycle.DRAINING,
                ProviderInstanceLifecycle.FAILED,
            )
            if active_lifecycle:
                try:
                    self.lifecycle.capacity.reserve(
                        instance_copy.provider_instance_id,
                        instance_copy.reservation,
                    )
                except Exception as exc:
                    instance_copy.failure_reason = f"restart capacity reconciliation failed: {exc}"
                    object.__setattr__(instance_copy, "lifecycle", ProviderInstanceLifecycle.FAILED)
                    object.__setattr__(
                        instance_copy,
                        "lifecycle_history",
                        (*instance_copy.lifecycle_history, ProviderInstanceLifecycle.FAILED),
                    )
                    object.__setattr__(instance_copy, "updated_at", observed_now)
                    failed.append(instance_copy.provider_instance_id)
            self.lifecycle.instances[instance_copy.provider_instance_id] = instance_copy
            key_id = instance_copy.share_key.key_id or ""
            generation_counts[key_id] += 1
            if instance_copy.lifecycle in (
                ProviderInstanceLifecycle.COLD,
                ProviderInstanceLifecycle.WARMING,
                ProviderInstanceLifecycle.ACTIVE,
                ProviderInstanceLifecycle.IDLE_LEASE,
            ):
                self.lifecycle._reusable_by_key[key_id] = instance_copy.provider_instance_id  # type: ignore[attr-defined]
            restored_instances.append(instance_copy.provider_instance_id)

        self.lifecycle._generation_by_key.update(generation_counts)  # type: ignore[attr-defined]

        restored_leases: list[UUID] = []
        stale_leases: list[UUID] = []
        for managed in leases:
            lease_copy = managed.model_copy(deep=True)
            instance = self.lifecycle.instances.get(lease_copy.lease.provider_instance_id)
            if instance is None or instance.lifecycle == ProviderInstanceLifecycle.FAILED:
                lease_copy.lease.status = ProviderLeaseStatus.FAILED
                stale_leases.append(lease_copy.lease.lease_id)
            self.lifecycle.leases[lease_copy.lease.lease_id] = lease_copy
            self.lifecycle._lease_idempotency[(  # type: ignore[attr-defined]
                lease_copy.lease.demand_id,
                lease_copy.lease.plan_id,
                lease_copy.step_id,
            )] = lease_copy.lease.lease_id
            restored_leases.append(lease_copy.lease.lease_id)

        restored_plans: list[UUID] = []
        for managed_plan in plans:
            plan_copy = managed_plan.model_copy(deep=True)
            self.lifecycle.plans[plan_copy.plan.plan_id] = plan_copy
            restored_plans.append(plan_copy.plan.plan_id)

        latest_by_node = _latest_heartbeats(heartbeats)
        orphan_agent_instances: list[str] = []
        for node_id, heartbeat in latest_by_node.items():
            persisted_on_node = {
                item.provider_instance_id
                for item in self.lifecycle.instances.values()
                if item.share_key.node_id == node_id
                and item.lifecycle not in (
                    ProviderInstanceLifecycle.DRAINING,
                    ProviderInstanceLifecycle.FAILED,
                )
            }
            agent_active = set(heartbeat.active_provider_instance_ids)
            orphan_agent_instances.extend(sorted(agent_active - persisted_on_node))
            missing_on_agent = persisted_on_node - agent_active
            for instance_id in sorted(missing_on_agent):
                affected = self.lifecycle.mark_failed(
                    instance_id,
                    reason="provider absent from node heartbeat during restart reconciliation",
                    now=observed_now,
                )
                failed.append(instance_id)
                for demand_id in affected:
                    for lease in self.lifecycle.leases.values():
                        if lease.lease.demand_id == demand_id:
                            stale_leases.append(lease.lease.lease_id)

            heartbeat_demands = set(heartbeat.active_demand_ids)
            for managed in self.lifecycle.active_leases:
                if managed.lease.node_id != node_id:
                    continue
                if managed.lease.demand_id not in heartbeat_demands:
                    stale_leases.append(managed.lease.lease_id)

        if require_heartbeat_for_active_instances:
            heartbeat_nodes = set(latest_by_node)
            for instance in tuple(self.lifecycle.active_instances):
                if instance.share_key.node_id in heartbeat_nodes:
                    continue
                affected = self.lifecycle.mark_failed(
                    instance.provider_instance_id,
                    reason="no node heartbeat available during restart reconciliation",
                    now=observed_now,
                )
                failed.append(instance.provider_instance_id)
                for demand_id in affected:
                    for lease in self.lifecycle.leases.values():
                        if lease.lease.demand_id == demand_id:
                            stale_leases.append(lease.lease.lease_id)

        # Persist any status changes produced during reconciliation.
        for instance in self.lifecycle.instances.values():
            self.store.put(
                "provider_instances", instance.provider_instance_id, instance
            )
        for lease in self.lifecycle.leases.values():
            self.store.put("leases", str(lease.lease.lease_id), lease)

        report = ReconciliationReport(
            restored_provider_instance_ids=tuple(sorted(set(restored_instances))),
            restored_lease_ids=tuple(sorted(set(restored_leases), key=str)),
            restored_plan_ids=tuple(sorted(set(restored_plans), key=str)),
            failed_provider_instance_ids=tuple(sorted(set(failed))),
            orphan_agent_provider_instance_ids=tuple(sorted(set(orphan_agent_instances))),
            stale_lease_ids=tuple(sorted(set(stale_leases), key=str)),
            notes=(
                "semantic graphs, hypotheses, frontiers, and demands remain in their durable collections",
                "provider and lease indexes were rebuilt from compact records",
            ),
        )
        self.store.append_event(
            ControlEvent(
                event_type=ControlEventType.RECONCILIATION_COMPLETED,
                entity_type="orchestrator",
                entity_id="runtime",
                payload={
                    "restored_provider_instances": len(report.restored_provider_instance_ids),
                    "restored_leases": len(report.restored_lease_ids),
                    "failed_provider_instances": len(report.failed_provider_instance_ids),
                },
            )
        )
        return report


class EventEmissionLedger:
    """Suppresses duplicate final complex-event emission across restarts."""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def claim(self, event_key: str, payload: dict[str, Any]) -> bool:
        if self.store.contains("emitted_events", event_key):
            return False
        self.store.put("emitted_events", event_key, payload, expected_version=-1)
        self.store.append_event(
            ControlEvent(
                event_type=ControlEventType.CE_EMITTED,
                entity_type="complex_event",
                entity_id=event_key,
                payload=payload,
            )
        )
        return True


def _latest_heartbeats(heartbeats: tuple[NodeHeartbeat, ...]) -> dict[str, NodeHeartbeat]:
    result: dict[str, NodeHeartbeat] = {}
    for heartbeat in heartbeats:
        prior = result.get(heartbeat.node_id)
        if prior is None or (heartbeat.sent_at, heartbeat.sequence) > (
            prior.sent_at,
            prior.sequence,
        ):
            result[heartbeat.node_id] = heartbeat
    return result
