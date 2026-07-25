from __future__ import annotations

import unittest
from pathlib import Path

from fable.scheduling import (
    AdmissionBatchResult,
    AdmissionDecision,
    HistoricalDemand,
    ManagedLease,
    ProviderInstanceLifecycle,
    ProviderInstanceRecord,
)


FIXTURE_DIR = Path(__file__).parent / "phase5_fixtures"


class Phase5FixtureTests(unittest.TestCase):
    def test_admission_batch_validates_and_records_provider_reuse(self) -> None:
        batch = AdmissionBatchResult.model_validate_json(
            (FIXTURE_DIR / "admission_batch.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(batch.records), 2)
        self.assertTrue(all(record.decision == AdmissionDecision.ADMITTED for record in batch.records))
        self.assertEqual(len(batch.admitted_plan_ids), 2)
        self.assertTrue(any(record.reused_provider_instance_ids for record in batch.records))

    def test_provider_instance_and_lease_validate(self) -> None:
        instance = ProviderInstanceRecord.model_validate_json(
            (FIXTURE_DIR / "provider_instance.json").read_text(encoding="utf-8")
        )
        lease = ManagedLease.model_validate_json(
            (FIXTURE_DIR / "managed_lease.json").read_text(encoding="utf-8")
        )
        self.assertEqual(instance.lifecycle, ProviderInstanceLifecycle.WARMING)
        self.assertIn(lease.lease.lease_id, instance.active_lease_ids)
        self.assertEqual(lease.lease.provider_instance_id, instance.provider_instance_id)
        self.assertEqual(lease.share_key_id, instance.share_key.key_id)

    def test_historical_demand_validates(self) -> None:
        historical = HistoricalDemand.model_validate_json(
            (FIXTURE_DIR / "historical_demand.json").read_text(encoding="utf-8")
        )
        self.assertEqual(historical.historical_interval, historical.demand.event_time_interval)
        self.assertEqual(historical.retained_input_type, "audio_segment.v1")
        self.assertLess(historical.historical_interval.end, historical.created_at)
        self.assertGreater(historical.buffer_expires_at, historical.created_at)


if __name__ == "__main__":
    unittest.main()
