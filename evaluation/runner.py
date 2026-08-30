"""Planning-smoke runner for immutable evaluation manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import time

from fable.language import load_and_compile_event
from fable.planning import PhysicalPlanner
from fable.runtime import CEInstanceManager

from .baselines import resolve_policy
from .deployment import load_runtime_state
from .manifest import EvaluationCell
from .metrics import CellOutcome


def run_planning_cell(cell: EvaluationCell) -> tuple[CellOutcome, dict[str, object]]:
    """Compile and plan one cell without fabricating live accuracy evidence."""

    started = time.monotonic()
    try:
        event = load_and_compile_event(cell.event)
        policy = resolve_policy(
            cell.policy,
            event_name=event.name,
            static_placements=cell.static_placements,
        )
        runtime = load_runtime_state(cell.deployment)
        now = datetime.now(timezone.utc)
        frontier = CEInstanceManager(event).current_frontier(now)
        plan = policy.plan(frontier, runtime, now=now)
        elapsed = time.monotonic() - started
        detail = {
            "schema_version": "fable.evaluation.planning_cell.v1",
            "cell_id": cell.cell_id,
            "event": event.name,
            "policy": cell.policy,
            "provider_ids": list(plan.provider_ids),
            "steps": [
                {
                    "provider_id": step.provider_id,
                    "node_id": step.node_id,
                    "source_ids": list(step.source_ids),
                    "output_type": step.output_type,
                }
                for step in plan.steps
            ],
            "predicted_completion_ms": plan.predicted_completion_ms,
            "estimated_transfer_bytes": plan.transfer_bytes,
            "quality": plan.quality,
        }
        return (
            CellOutcome(
                cell.cell_id,
                event.name,
                cell.policy,
                "SUCCESS",
                elapsed,
                len(plan.steps),
                plan.predicted_completion_ms,
            ),
            detail,
        )
    except Exception as exc:
        return (
            CellOutcome(
                cell.cell_id,
                cell.event.stem,
                cell.policy,
                "FAILED",
                time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            ),
            {},
        )
