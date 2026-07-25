#!/usr/bin/env python3
"""Demonstrate Phase-5 admission, provider sharing, leases, and replay."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fable.common.base import to_jsonable  # noqa: E402
from fable.common.enums import CancellationScope, ExecutionMode  # noqa: E402
from fable.common.examples import BASE_TIME  # noqa: E402
from fable.common.ids import uuid7  # noqa: E402
from fable.common.time import EventTimeInterval  # noqa: E402
from fable.planning.models import ExternalInputKind  # noqa: E402
from fable.planning.testing import fake_deployment, fake_provider_registry  # noqa: E402
from fable.scheduling import (  # noqa: E402
    CancellationManager,
    CancellationRequest,
    CapacityLedger,
    HistoricalDemandSpec,
    MultiTenantScheduler,
    ProviderLifecycleManager,
    RetrospectiveDemandGenerator,
    TaskSchedulingPolicy,
)
from fable.scheduling.testing import fake_audio_candidate, fake_audio_demand  # noqa: E402


def main() -> int:
    providers = fake_provider_registry()
    lifecycle = ProviderLifecycleManager(
        provider_registry=providers,
        capacity=CapacityLedger(fake_deployment()),
        idle_grace_ms=100,
    )
    scheduler = MultiTenantScheduler(lifecycle=lifecycle)

    hypothesis_a = uuid7()
    hypothesis_b = uuid7()
    demand_a = fake_audio_demand(
        request_id="robbery_a",
        hypothesis_id=hypothesis_a,
        graph_node_id="gunshot_a",
    )
    demand_b = fake_audio_demand(
        request_id="robbery_b",
        hypothesis_id=hypothesis_b,
        graph_node_id="gunshot_b",
    )
    candidate_a = fake_audio_candidate(
        demand_a,
        provider_registry=providers,
        task_policy=TaskSchedulingPolicy(request_id="robbery_a"),
    )
    candidate_b = fake_audio_candidate(
        demand_b,
        provider_registry=providers,
        task_policy=TaskSchedulingPolicy(request_id="robbery_b"),
    )
    admission = scheduler.admit((candidate_a, candidate_b), now=BASE_TIME)
    shared = lifecycle.active_instances[0]
    lifecycle.mark_ready(shared.provider_instance_id, now=BASE_TIME + timedelta(milliseconds=10))

    cancelled_a = CancellationManager(lifecycle).cancel(
        CancellationRequest(
            scope=CancellationScope.HYPOTHESIS,
            request_id="robbery_a",
            hypothesis_id=hypothesis_a,
            reason="hypothesis invalidated",
        ),
        now=BASE_TIME + timedelta(milliseconds=20),
    )

    original_history = fake_audio_demand(
        request_id="robbery_b",
        hypothesis_id=hypothesis_b,
        graph_node_id="historical_arrival",
        interval=EventTimeInterval(
            start=BASE_TIME - timedelta(seconds=20),
            end=BASE_TIME - timedelta(seconds=19),
        ),
    )
    historical = RetrospectiveDemandGenerator().generate(
        (
            HistoricalDemandSpec(
                original_demand=original_history,
                historical_interval=original_history.event_time_interval,
                source_id="microphone_store",
                retained_input_type="audio_segment.v1",
                raw_buffer_interval=EventTimeInterval(
                    start=BASE_TIME - timedelta(minutes=5),
                    end=BASE_TIME,
                ),
                buffer_expires_at=BASE_TIME + timedelta(minutes=2),
                reason="gunshot activated recovery of an earlier interval",
            ),
        ),
        now=BASE_TIME,
    ).demands[0]
    historical_candidate = fake_audio_candidate(
        historical.demand,
        provider_registry=providers,
        task_policy=TaskSchedulingPolicy(request_id="robbery_b"),
        execution_mode=ExecutionMode.RETROSPECTIVE,
        input_kind=ExternalInputKind.RETAINED_ARTIFACT,
        artifact_id=uuid7(),
        expires_at=historical.buffer_expires_at,
    )
    replay_admission = scheduler.admit((historical_candidate,), now=BASE_TIME)

    summary = {
        "initial_admission": admission,
        "shared_provider": shared,
        "shared_lease_hypotheses": tuple(
            lease.hypothesis_id
            for lease in lifecycle.leases_for_instance(shared.provider_instance_id)
        ),
        "first_hypothesis_cancellation": cancelled_a,
        "historical_demand": historical,
        "historical_admission": replay_admission,
        "active_execution_modes": tuple(
            sorted(lease.execution_mode.value for lease in lifecycle.active_leases)
        ),
        "capacity_used": {
            node_id: lifecycle.capacity.used(node_id)
            for node_id in sorted(lifecycle.capacity.deployment.nodes)
        },
    }
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
