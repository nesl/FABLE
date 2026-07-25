from __future__ import annotations

import json
import unittest
from pathlib import Path

from fable.common.serialization import SCHEMA_REGISTRY, load_versioned, schemas_compatible


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


class SerializationFixtureTests(unittest.TestCase):
    def test_all_json_fixtures_load_through_version_registry(self) -> None:
        fixture_paths = sorted(FIXTURE_DIR.glob("*.json"))
        self.assertGreaterEqual(len(fixture_paths), len(SCHEMA_REGISTRY))
        observed_versions = set()
        for path in fixture_paths:
            raw = json.loads(path.read_text(encoding="utf-8"))
            record = load_versioned(raw)
            observed_versions.add(record.schema_version)
            round_trip = load_versioned(record.model_dump_json())
            self.assertEqual(round_trip, record, msg=path.name)
        self.assertEqual(observed_versions, set(SCHEMA_REGISTRY))

    def test_every_registered_record_exports_json_schema(self) -> None:
        for version, model in SCHEMA_REGISTRY.items():
            schema = model.model_json_schema()
            self.assertEqual(schema.get("type"), "object", msg=version)
            self.assertIn("properties", schema, msg=version)

    def test_schema_compatibility_is_exact_by_family_and_major(self) -> None:
        self.assertTrue(
            schemas_compatible("fable.predicate_demand.v1", "fable.predicate_demand.v1")
        )
        self.assertFalse(
            schemas_compatible("fable.predicate_demand.v1", "fable.predicate_result.v1")
        )
        self.assertFalse(
            schemas_compatible("fable.predicate_demand.v1", "fable.predicate_demand.v2")
        )


if __name__ == "__main__":
    unittest.main()
