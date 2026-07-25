from __future__ import annotations

import unittest
from pathlib import Path

from fable.common.enums import ProviderPortKind
from fable.common.provider_catalog import load_provider_contracts
from providers.tools.chain_validator import ChainValidator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProviderCatalogContractTests(unittest.TestCase):
    def test_existing_catalog_loads_as_phase0_provider_contracts(self) -> None:
        contracts = load_provider_contracts(
            PROJECT_ROOT / "providers" / "registry" / "catalog.yaml"
        )
        self.assertGreaterEqual(len(contracts), 20)
        tracker = contracts["multi_object_tracker"]
        self.assertIn("TRACKS", tracker.semantic_capabilities.predicate_ids)
        kinds = {port.kind for port in tracker.ports}
        self.assertNotIn(ProviderPortKind.STATE_INPUT, kinds)
        self.assertNotIn(ProviderPortKind.STATE_OUTPUT, kinds)
        self.assertIn("track_summary.v1", {port.data_type for port in tracker.ports})

    def test_existing_provider_chain_catalog_remains_valid(self) -> None:
        validator = ChainValidator.from_files(
            PROJECT_ROOT / "providers" / "registry" / "data_types.yaml",
            PROJECT_ROOT / "providers" / "registry" / "catalog.yaml",
        )
        report = validator.validate_all()
        self.assertTrue(report.ok, msg=report.format())


if __name__ == "__main__":
    unittest.main()
