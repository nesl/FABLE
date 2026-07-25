from __future__ import annotations

import unittest

from fable.common.enums import PlanStatus
from fable.common.examples import fake_convoy_runtime_records
from fable.common.schemas import ExecutionPlan, PhysicalPlanLabel, PlanCost, PlanStep


class PlanRecordTests(unittest.TestCase):
    def test_plan_label_id_is_content_deterministic(self) -> None:
        *_, checkpoint, demand, artifact = fake_convoy_runtime_records()
        step = PlanStep(
            step_id="step_1",
            provider_id="follows_cross_sensor",
            node_id="edge_1",
            input_artifact_ids=(artifact.artifact_id,),
            output_data_types=("predicate_match.v1",),
            estimated_execution_ms=10,
        )
        kwargs = dict(
            checkpoint_id=checkpoint.checkpoint_id,
            covered_demand_ids=(demand.demand_id,),
            steps=(step,),
            cost=PlanCost(
                predicted_completion_ms=10,
                deadline_slack_ms=100,
                startup_cost_ms=0,
                resource_cost_units=1.0,
                transfer_bytes=0,
            ),
            hard_constraints_satisfied=True,
            quality_floor_satisfied=True,
        )
        left = PhysicalPlanLabel(**kwargs)
        right = PhysicalPlanLabel(**kwargs)
        self.assertEqual(left.label_id, right.label_id)

    def test_execution_plan_uses_selected_label_but_not_vice_versa(self) -> None:
        *_, checkpoint, demand, _ = fake_convoy_runtime_records()
        step = PlanStep(
            step_id="step_1",
            provider_id="follows_local_geometry",
            node_id="sensor_1",
            output_data_types=("predicate_match.v1",),
            estimated_execution_ms=10,
        )
        label = PhysicalPlanLabel(
            checkpoint_id=checkpoint.checkpoint_id,
            covered_demand_ids=(demand.demand_id,),
            steps=(step,),
            cost=PlanCost(
                predicted_completion_ms=10,
                deadline_slack_ms=100,
                startup_cost_ms=0,
                resource_cost_units=1.0,
                transfer_bytes=0,
            ),
            hard_constraints_satisfied=True,
            quality_floor_satisfied=True,
        )
        plan = ExecutionPlan(
            label_id=label.label_id,
            checkpoint_id=checkpoint.checkpoint_id,
            demand_ids=(demand.demand_id,),
            steps=(step,),
            status=PlanStatus.CANDIDATE,
        )
        self.assertEqual(plan.label_id, label.label_id)


if __name__ == "__main__":
    unittest.main()
