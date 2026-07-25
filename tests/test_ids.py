from __future__ import annotations

import unittest
import time

from fable.common.ids import canonical_json_bytes, deterministic_id, uuid7


class IdentifierTests(unittest.TestCase):
    def test_uuid7_has_correct_version_variant_and_timestamp(self) -> None:
        timestamp_ms = int(time.time_ns() // 1_000_000) + 60_000
        value = uuid7(now_ms=timestamp_ms)
        self.assertEqual(value.version, 7)
        self.assertEqual(value.variant, "specified in RFC 4122")
        self.assertEqual(value.int >> 80, timestamp_ms)

    def test_uuid7_is_monotonic_within_one_millisecond(self) -> None:
        timestamp_ms = int(time.time_ns() // 1_000_000) + 60_001
        left = uuid7(now_ms=timestamp_ms)
        right = uuid7(now_ms=timestamp_ms)
        self.assertLess(left.int, right.int)

    def test_canonical_json_is_independent_of_mapping_order(self) -> None:
        left = {"b": 2, "a": {"y": 4, "x": 3}}
        right = {"a": {"x": 3, "y": 4}, "b": 2}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(
            deterministic_id("example", left),
            deterministic_id("example", right),
        )


if __name__ == "__main__":
    unittest.main()
