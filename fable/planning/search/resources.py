"""Resource-footprint projection shared by feasibility and label extension."""

from __future__ import annotations

from collections.abc import Mapping

from fable.planning.models import PhysicalAlternative
from fable.planning.search_models import NodeResourceFootprint


def combine_resources(
    parent_resources: Mapping[str, NodeResourceFootprint],
    alternative: PhysicalAlternative,
    *,
    existing_provider_keys: frozenset[tuple[str, str]] = frozenset(),
) -> dict[str, NodeResourceFootprint]:
    values: dict[str, tuple[float, int, int]] = {
        node_id: (item.cpu_cores, item.memory_mb, item.gpu_memory_mb)
        for node_id, item in parent_resources.items()
    }
    for step in alternative.step_placements:
        # A compatible active provider is already consuming its resources; the
        # physical planner charges only incremental capacity. Scheduling later
        # attaches an independent demand lease to that provider instance.
        if (
            step.reused_provider_instance_id is not None
            or (step.node_id, step.provider_id) in existing_provider_keys
        ):
            continue
        cpu, memory, gpu = values.get(step.node_id, (0.0, 0, 0))
        values[step.node_id] = (
            cpu + step.cpu_cores,
            memory + step.memory_mb,
            gpu + step.gpu_memory_mb,
        )
    return {
        node_id: NodeResourceFootprint(
            node_id=node_id,
            cpu_cores=cpu,
            memory_mb=memory,
            gpu_memory_mb=gpu,
        )
        for node_id, (cpu, memory, gpu) in values.items()
    }


__all__ = ["combine_resources"]
