"""Projection of internal search labels into downstream execution contracts."""

from __future__ import annotations

from datetime import datetime

from fable.common.enums import PlanStatus
from fable.common.schemas import ExecutionPlan, ResourceReservation
from fable.common.time import ensure_utc, utc_now
from fable.planning.search_models import LabelSearchState


class ExecutionPlanProjector:
    """Convert a selected search state to the complete execution contract."""

    def project(
        self,
        state: LabelSearchState | None,
        *,
        now: datetime | None = None,
    ) -> ExecutionPlan | None:
        if state is None:
            return None
        network_bytes_by_node: dict[str, int] = {}
        for step in state.label.steps:
            network_bytes_by_node[step.node_id] = (
                network_bytes_by_node.get(step.node_id, 0)
                + step.estimated_transfer_bytes
            )
        return ExecutionPlan(
            label_id=state.label_id,
            checkpoint_id=state.label.checkpoint_id,
            demand_ids=state.label.covered_demand_ids,
            steps=state.label.steps,
            reservations=tuple(
                ResourceReservation(
                    node_id=item.node_id,
                    cpu_cores=item.cpu_cores,
                    memory_mb=item.memory_mb,
                    gpu_memory_mb=item.gpu_memory_mb,
                    network_bytes=network_bytes_by_node.get(item.node_id, 0),
                )
                for item in state.node_resources
            ),
            status=PlanStatus.CANDIDATE,
            created_at=ensure_utc(now or utc_now()),
            expires_at=state.expires_at,
        )


__all__ = ["ExecutionPlanProjector"]
