from __future__ import annotations

import unittest

from pydantic import ValidationError

from fable.common.examples import fake_convoy_runtime_records
from fable.common.provider_catalog import load_provider_contracts


class BoundaryGuardTests(unittest.TestCase):
    def test_predicate_demand_rejects_provider_selection_fields(self) -> None:
        *_, demand, _ = fake_convoy_runtime_records()
        payload = demand.model_dump(mode="python")
        payload["provider_id"] = "follows_cross_sensor"
        with self.assertRaises(ValidationError):
            type(demand).model_validate(payload)

    def test_predicate_demand_rejects_selected_node_fields(self) -> None:
        *_, demand, _ = fake_convoy_runtime_records()
        payload = demand.model_dump(mode="python")
        payload["selected_node_id"] = "edge_1"
        with self.assertRaises(ValidationError):
            type(demand).model_validate(payload)

    def test_provider_contract_rejects_hypothesis_control_fields(self) -> None:
        contracts = load_provider_contracts("providers/registry/catalog.yaml")
        contract = contracts["follows_local_geometry"]
        payload = contract.model_dump(mode="python")
        payload["advance_hypothesis"] = True
        payload["fork_policy"] = "BY_BINDING"
        with self.assertRaises(ValidationError):
            type(contract).model_validate(payload)


if __name__ == "__main__":
    unittest.main()
