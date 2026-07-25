"""Checkpoint-facing spatial coordination helpers."""

from __future__ import annotations

from collections.abc import Iterable

from fable.common.schemas import PredicateResult
from fable.semantic.models import DerivedFrontier

from .models import (
    SpatialCheckpointGuidance,
    SpatialObservation,
    SpatialSensorBindings,
)
from .transition_model import SiteSensorTransitionModel


class SpatialCheckpointCoordinator:
    """Derive selective next-sensor guidance after a semantic transition.

    Heading is supplied by a tracking/trajectory provider or caller.  The
    coordinator does not ask an LLM to infer geometry and does not change event
    truth; it only predicts where useful evidence is likely to appear next.
    """

    def __init__(
        self,
        *,
        transition_model: SiteSensorTransitionModel,
        bindings: SpatialSensorBindings,
    ) -> None:
        self.transition_model = transition_model
        self.bindings = bindings

    def after_result(
        self,
        *,
        result: PredicateResult,
        frontier: DerivedFrontier,
        observed_heading: str | None,
        active_deployment_id: str | None,
        corridor_id: str | None = None,
        branch_unresolved: bool = False,
        maximum_observation_groups: int = 1,
    ) -> SpatialCheckpointGuidance:
        if not result.provenance.source_ids:
            raise ValueError("predicate result has no source provenance for spatial prediction")
        runtime_source_id = result.provenance.source_ids[0]
        topology_sensor_id = (
            self.bindings.sensor_for_source(
                runtime_source_id, active_deployment_id
            )
            or runtime_source_id
        )
        observation = SpatialObservation(
            current_sensor_id=topology_sensor_id,
            observed_heading=observed_heading,
            active_deployment_id=active_deployment_id,
            corridor_id=corridor_id,
            branch_unresolved=branch_unresolved,
            maximum_observation_groups=maximum_observation_groups,
        )
        return SpatialCheckpointGuidance(
            prediction=self.transition_model.predict(observation, bindings=self.bindings),
            graph_node_ids=frontier.snapshot.enabled_node_ids,
        )

    def manual(
        self,
        *,
        observation: SpatialObservation,
        graph_node_ids: Iterable[str],
    ) -> SpatialCheckpointGuidance:
        return SpatialCheckpointGuidance(
            prediction=self.transition_model.predict(observation, bindings=self.bindings),
            graph_node_ids=tuple(graph_node_ids),
            reason="MANUAL_LOOKUP",
        )
