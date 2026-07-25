from __future__ import annotations

from datetime import timedelta

from fable.common.enums import NodeAvailability, ProviderLeaseStatus
from fable.common.time import EventTimeInterval, utc_now
from fable.distributed.demo import build_replay_audio_candidate
from fable.distributed.heartbeat import HeartbeatMonitor
from fable.distributed.reconciliation import EventEmissionLedger, RuntimeReconciler
from fable.planning.testing import fake_deployment
from fable.scheduling.capacity import CapacityLedger
from fable.scheduling.lifecycle import ProviderLifecycleManager

from .fake_phase6_data import make_stack, provider_registry, wait_until


def _candidate(stack, request_id="failure_task"):
    now = utc_now()
    return build_replay_audio_candidate(
        provider_registry=stack.registry,
        node_id="sensor_a",
        source_id="sensor_a:audio",
        event_interval=EventTimeInterval(
            start=now - timedelta(minutes=1),
            end=now + timedelta(minutes=1),
        ),
        request_id=request_id,
        now=now,
        deadline_seconds=60,
    )


def test_three_and_five_missed_heartbeats_produce_suspect_then_unavailable(tmp_path):
    stack = make_stack(tmp_path, heartbeat_interval=100)
    try:
        heartbeat = stack.agents["sensor_a"].emit_heartbeat()
        received_at = utc_now()
        # The agent already sent a first heartbeat during start.  Tick relative to now.
        suspect = stack.orchestrator.check_heartbeats(now=received_at + timedelta(seconds=3.1))
        assert suspect[-1].current == NodeAvailability.SUSPECT
        unavailable = stack.orchestrator.check_heartbeats(now=received_at + timedelta(seconds=5.1))
        assert unavailable[-1].current == NodeAvailability.UNAVAILABLE
    finally:
        stack.stop()


def test_node_unavailability_fails_leases_and_requests_replanning(tmp_path):
    stack = make_stack(tmp_path, heartbeat_interval=100)
    try:
        candidate = _candidate(stack)
        stack.orchestrator.submit_candidates((candidate,), now=utc_now())
        assert wait_until(lambda: len(stack.received_results) == 1)
        tick_from = utc_now()
        transitions = stack.orchestrator.check_heartbeats(
            now=tick_from + timedelta(seconds=5.2)
        )
        assert any(item.current == NodeAvailability.UNAVAILABLE for item in transitions)
        assert stack.orchestrator.replan_requests
        affected = set(stack.orchestrator.replan_requests[-1][1])
        assert candidate.demands[0].demand_id in affected
        managed = next(iter(stack.lifecycle.leases.values()))
        assert managed.lease.status == ProviderLeaseStatus.FAILED
        persisted = stack.store.get_raw("leases", str(managed.lease.lease_id))
        assert persisted["lease"]["status"] == ProviderLeaseStatus.FAILED.value
    finally:
        stack.stop()


def test_restart_reconciliation_restores_provider_plan_and_lease_indexes(tmp_path):
    stack = make_stack(tmp_path)
    try:
        candidate = _candidate(stack, request_id="restart_task")
        stack.orchestrator.submit_candidates((candidate,), now=utc_now())
        assert wait_until(lambda: len(stack.received_results) == 1)

        registry = provider_registry()
        replacement = ProviderLifecycleManager(
            provider_registry=registry,
            capacity=CapacityLedger(fake_deployment()),
        )
        report = RuntimeReconciler(store=stack.store, lifecycle=replacement).restore()
        assert report.restored_plan_ids == (candidate.plan.plan_id,)
        assert len(report.restored_provider_instance_ids) == 1
        assert len(report.restored_lease_ids) == 1
        assert candidate.plan.plan_id in replacement.plans
        assert replacement.active_leases

        emissions = EventEmissionLedger(stack.store)
        assert emissions.claim("event:restart:1", {"request_id": "restart_task"})
        assert not emissions.claim("event:restart:1", {"request_id": "restart_task"})
    finally:
        stack.stop()
