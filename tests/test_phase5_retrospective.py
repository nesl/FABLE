from __future__ import annotations

from datetime import timedelta

from fable.common.enums import ExecutionMode
from fable.common.examples import BASE_TIME
from fable.common.ids import uuid7
from fable.common.time import EventTimeInterval
from fable.planning.models import ExternalInputKind
from fable.planning.testing import fake_deployment, fake_provider_registry
from fable.scheduling import (
    CapacityLedger,
    HistoricalDemandSpec,
    MultiTenantScheduler,
    ProviderLifecycleManager,
    RetrospectiveConfig,
    RetrospectiveDemandGenerator,
    TaskSchedulingPolicy,
)
from fable.scheduling.models import AdmissionDecision
from fable.scheduling.testing import fake_audio_candidate, fake_audio_demand


def test_historical_generation_is_bounded_by_buffer_deadline_and_hypothesis_limit() -> None:
    original = fake_audio_demand(
        request_id="history_task",
        interval=EventTimeInterval(
            start=BASE_TIME - timedelta(seconds=10),
            end=BASE_TIME - timedelta(seconds=9),
        ),
    )
    generator = RetrospectiveDemandGenerator(
        RetrospectiveConfig(
            maximum_interval_ms=5_000,
            maximum_lookback_ms=60_000,
            maximum_outstanding_per_hypothesis=1,
        )
    )
    spec = HistoricalDemandSpec(
        original_demand=original,
        historical_interval=original.event_time_interval,
        source_id="microphone_store",
        retained_input_type="audio_segment.v1",
        raw_buffer_interval=EventTimeInterval(
            start=BASE_TIME - timedelta(minutes=1),
            end=BASE_TIME,
        ),
        buffer_expires_at=BASE_TIME + timedelta(minutes=1),
        reason="gunshot made the earlier audio interval relevant",
    )
    first = generator.generate((spec,), now=BASE_TIME)
    assert len(first.demands) == 1
    historical = first.demands[0]
    assert historical.demand.event_time_interval == original.event_time_interval
    assert historical.demand.demand_id != original.demand_id
    assert "audio_segment.v1" in historical.demand.required_input_artifact_types

    second = generator.generate((spec,), now=BASE_TIME)
    assert second.rejections[0].code == "HYPOTHESIS_LIMIT"


def test_live_and_historical_work_share_the_same_admission_path() -> None:
    registry = fake_provider_registry()
    lifecycle = ProviderLifecycleManager(
        provider_registry=registry,
        capacity=CapacityLedger(fake_deployment()),
    )
    scheduler = MultiTenantScheduler(lifecycle=lifecycle)

    live_demand = fake_audio_demand(
        request_id="combined_task",
        hypothesis_id=uuid7(),
        graph_node_id="live_escape",
    )
    original_history = fake_audio_demand(
        request_id="combined_task",
        hypothesis_id=uuid7(),
        graph_node_id="historical_arrival",
        interval=EventTimeInterval(
            start=BASE_TIME - timedelta(seconds=20),
            end=BASE_TIME - timedelta(seconds=19),
        ),
    )
    generator = RetrospectiveDemandGenerator()
    generated = generator.generate(
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
                reason="semantic trigger requested historical recovery",
            ),
        ),
        now=BASE_TIME,
    )
    historical = generated.demands[0]
    policy = TaskSchedulingPolicy(request_id="combined_task")
    live_candidate = fake_audio_candidate(
        live_demand,
        provider_registry=registry,
        task_policy=policy,
        execution_mode=ExecutionMode.LIVE,
        input_kind=ExternalInputKind.LIVE_SOURCE,
    )
    historical_candidate = fake_audio_candidate(
        historical.demand,
        provider_registry=registry,
        task_policy=policy,
        execution_mode=ExecutionMode.RETROSPECTIVE,
        input_kind=ExternalInputKind.RETAINED_ARTIFACT,
        artifact_id=uuid7(),
        expires_at=historical.buffer_expires_at,
    )

    batch = scheduler.admit((historical_candidate, live_candidate), now=BASE_TIME)
    assert batch.ordered_candidate_ids[0] == live_candidate.candidate_id
    assert all(record.decision == AdmissionDecision.ADMITTED for record in batch.records)
    assert {lease.execution_mode for lease in lifecycle.active_leases} == {
        ExecutionMode.LIVE,
        ExecutionMode.RETROSPECTIVE,
    }
