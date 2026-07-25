#!/usr/bin/env python3
"""Generate deterministic-shaped fake records for Phase-5 tests."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fable.common.base import to_jsonable  # noqa: E402
from fable.common.examples import BASE_TIME  # noqa: E402
from fable.common.ids import uuid7  # noqa: E402
from fable.common.time import EventTimeInterval  # noqa: E402
from fable.planning.testing import fake_deployment, fake_provider_registry  # noqa: E402
from fable.scheduling import (  # noqa: E402
    CapacityLedger,
    HistoricalDemandSpec,
    MultiTenantScheduler,
    ProviderLifecycleManager,
    RetrospectiveDemandGenerator,
    TaskSchedulingPolicy,
)
from fable.scheduling.testing import fake_audio_candidate, fake_audio_demand  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    providers = fake_provider_registry()
    lifecycle = ProviderLifecycleManager(
        provider_registry=providers,
        capacity=CapacityLedger(fake_deployment()),
    )
    scheduler = MultiTenantScheduler(lifecycle=lifecycle)
    hypothesis_a = uuid7()
    hypothesis_b = uuid7()
    demand_a = fake_audio_demand(request_id="fixture_a", hypothesis_id=hypothesis_a)
    demand_b = fake_audio_demand(request_id="fixture_b", hypothesis_id=hypothesis_b)
    candidate_a = fake_audio_candidate(
        demand_a,
        provider_registry=providers,
        task_policy=TaskSchedulingPolicy(request_id="fixture_a"),
    )
    candidate_b = fake_audio_candidate(
        demand_b,
        provider_registry=providers,
        task_policy=TaskSchedulingPolicy(request_id="fixture_b"),
    )
    admission = scheduler.admit((candidate_a, candidate_b), now=BASE_TIME)
    instance = lifecycle.active_instances[0]
    lease = lifecycle.active_leases[0]

    historical_original = fake_audio_demand(
        request_id="fixture_a",
        hypothesis_id=hypothesis_a,
        interval=EventTimeInterval(
            start=BASE_TIME - timedelta(seconds=10),
            end=BASE_TIME - timedelta(seconds=9),
        ),
    )
    historical = RetrospectiveDemandGenerator().generate(
        (
            HistoricalDemandSpec(
                original_demand=historical_original,
                historical_interval=historical_original.event_time_interval,
                source_id="microphone_store",
                retained_input_type="audio_segment.v1",
                raw_buffer_interval=EventTimeInterval(
                    start=BASE_TIME - timedelta(minutes=1),
                    end=BASE_TIME,
                ),
                buffer_expires_at=BASE_TIME + timedelta(minutes=1),
                reason="fixture retrospective trigger",
            ),
        ),
        now=BASE_TIME,
    ).demands[0]

    output = PROJECT_ROOT / "tests" / "phase5_fixtures"
    write_json(output / "admission_batch.json", admission)
    write_json(output / "provider_instance.json", instance)
    write_json(output / "managed_lease.json", lease)
    write_json(output / "historical_demand.json", historical)
    print(f"wrote fixtures to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
