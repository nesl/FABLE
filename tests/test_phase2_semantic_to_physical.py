from __future__ import annotations

import unittest

from fable.common.enums import BindingCapability
from fable.common.examples import predicate
from fable.common.enums import ResultKind
from fable.planning import PredicateSchemaError, default_predicate_registry
from fable.planning.testing import fake_follow_demand, fake_provider_registry


class Phase2SemanticToPhysicalTests(unittest.TestCase):
    def test_follow_frontier_compiles_to_provider_independent_demand(self) -> None:
        demand = fake_follow_demand()
        self.assertEqual(demand.semantic_predicate.predicate_id, "FOLLOWS")
        self.assertEqual(demand.bound_roles, {"leader": "vehicle_17"})
        self.assertEqual(demand.unbound_roles, ("follower",))
        self.assertEqual(
            demand.binding_policy.role_modes["leader"],
            BindingCapability.VALIDATE,
        )
        self.assertEqual(
            demand.binding_policy.role_modes["follower"],
            BindingCapability.INTRODUCE,
        )
        self.assertEqual(demand.binding_policy.forkable_roles, ("follower",))
        self.assertNotIn("provider_id", type(demand).model_fields)
        self.assertNotIn("node_id", type(demand).model_fields)
        self.assertEqual(
            set(demand.desired_continuation_types),
            {"pair_trajectory.v1", "track_summary.v1"},
        )

    def test_one_follow_demand_has_multiple_concrete_chains(self) -> None:
        demand = fake_follow_demand()
        registry = fake_provider_registry()
        chain_ids = {chain.chain_id for chain in registry.candidate_chains(demand)}
        self.assertEqual(
            chain_ids,
            {
                "follows_local_tracks",
                "follows_local_from_retained_detections",
                "follows_cross_camera_reid",
            },
        )

    def test_predicate_schema_rejects_unknown_parameter(self) -> None:
        registry = default_predicate_registry()
        invalid = predicate(
            "FOLLOWS",
            (
                ("leader", "leader", "vehicle"),
                ("follower", "follower", "vehicle"),
            ),
            result_kind=ResultKind.INTERVAL_MATCH,
            parameters={"not_a_parameter": 1},
        )
        with self.assertRaises(PredicateSchemaError):
            registry.validate(invalid)


if __name__ == "__main__":
    unittest.main()
