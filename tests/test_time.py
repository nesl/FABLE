from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from fable.common.time import (
    EventTimeInterval,
    LatenessPolicy,
    SourceWatermark,
    WatermarkSnapshot,
    interval_closed_by_watermarks,
)


class EventTimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_naive_timestamp_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            EventTimeInterval(
                start=datetime(2026, 1, 1),
                end=datetime(2026, 1, 1, 0, 0, 1),
            )

    def test_instant_interval_is_valid(self) -> None:
        interval = EventTimeInterval(start=self.base, end=self.base)
        self.assertTrue(interval.is_instant)
        self.assertEqual(interval.duration, timedelta(0))

    def test_watermark_closes_window_only_after_lateness(self) -> None:
        interval = EventTimeInterval(
            start=self.base,
            end=self.base + timedelta(seconds=10),
        )
        snapshot = WatermarkSnapshot(
            sources={
                "camera_a": SourceWatermark(
                    source_id="camera_a",
                    event_time=self.base + timedelta(seconds=10, milliseconds=999),
                )
            }
        )
        self.assertFalse(
            interval_closed_by_watermarks(
                interval,
                snapshot,
                ("camera_a",),
                LatenessPolicy(allowed_lateness_ms=1000),
            )
        )
        snapshot.sources["camera_a"].event_time = self.base + timedelta(seconds=11)
        self.assertTrue(
            interval_closed_by_watermarks(
                interval,
                snapshot,
                ("camera_a",),
                LatenessPolicy(allowed_lateness_ms=1000),
            )
        )

    def test_absence_window_requires_operational_coverage(self) -> None:
        interval = EventTimeInterval(
            start=self.base,
            end=self.base + timedelta(seconds=10),
        )
        snapshot = WatermarkSnapshot(
            sources={
                "camera_a": SourceWatermark(
                    source_id="camera_a",
                    event_time=self.base + timedelta(seconds=20),
                    operational_coverage=False,
                )
            }
        )
        self.assertFalse(
            interval_closed_by_watermarks(
                interval,
                snapshot,
                ("camera_a",),
                LatenessPolicy(),
            )
        )


if __name__ == "__main__":
    unittest.main()
