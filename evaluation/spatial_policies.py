"""Controlled sensor-selection policies for the spatial-coordination experiment."""

from __future__ import annotations

from pydantic import Field

from fable.common.base import FableModel
from .schemas import BaselineId
from .spatial_scope import SpatialReplaySelection

# Compatibility alias: spatial policies use the common baseline identifier so every
# evaluation record can be grouped without a second, incompatible ID namespace.
SpatialPolicyId = BaselineId


class SpatialSelectionContext(FableModel):
    replay_scope: SpatialReplaySelection
    resource_load_by_sensor: dict[str, float] = Field(default_factory=dict)
    branch_unresolved: bool = False
    maximum_active_sensors: int = Field(default=2, ge=1)
    actual_downstream_sensor_id: str | None = None


class SpatialPolicyDecision(FableModel):
    policy_id: BaselineId
    activated_sensor_ids: tuple[str, ...] = ()
    predicted_sensor_ids: tuple[str, ...] = ()
    unavailable_mobile_sensor_ids: tuple[str, ...] = ()
    reason: str = ""


class BroadcastSpatialPolicy:
    policy_id = BaselineId.SPATIAL_BROADCAST

    def select(self, context: SpatialSelectionContext) -> SpatialPolicyDecision:
        return SpatialPolicyDecision(
            policy_id=self.policy_id,
            activated_sensor_ids=context.replay_scope.all_replay_sensor_ids,
            predicted_sensor_ids=context.replay_scope.supported_sensor_ids,
            unavailable_mobile_sensor_ids=context.replay_scope.unavailable_mobile_sensor_ids,
            reason="Activate every currently replay-supported fixed Orin sensor.",
        )


class TopologyShortlistPolicy:
    policy_id = BaselineId.SPATIAL_TOPOLOGY_SHORTLIST

    def select(self, context: SpatialSelectionContext) -> SpatialPolicyDecision:
        predicted = context.replay_scope.supported_sensor_ids
        return SpatialPolicyDecision(
            policy_id=self.policy_id,
            activated_sensor_ids=predicted,
            predicted_sensor_ids=predicted,
            unavailable_mobile_sensor_ids=context.replay_scope.unavailable_mobile_sensor_ids,
            reason="Activate the qualitative topology shortlist without resource or hypothesis-specific ranking.",
        )


class ResourceOnlySpatialPolicy:
    policy_id = BaselineId.SPATIAL_RESOURCE_ONLY

    def select(self, context: SpatialSelectionContext) -> SpatialPolicyDecision:
        all_sensors = tuple(context.replay_scope.all_replay_sensor_ids)
        count = min(context.maximum_active_sensors, len(all_sensors))
        selected = tuple(
            sorted(
                all_sensors,
                key=lambda sensor: (
                    context.resource_load_by_sensor.get(sensor, 0.0),
                    sensor,
                ),
            )[:count]
        )
        return SpatialPolicyDecision(
            policy_id=self.policy_id,
            activated_sensor_ids=selected,
            predicted_sensor_ids=(),
            unavailable_mobile_sensor_ids=context.replay_scope.unavailable_mobile_sensor_ids,
            reason="Select the least-loaded replay sensors without using topology, route, heading, or identity bindings.",
        )


class FableSpatialPolicy:
    policy_id = BaselineId.SPATIAL_FABLE

    def select(self, context: SpatialSelectionContext) -> SpatialPolicyDecision:
        predicted = tuple(context.replay_scope.supported_sensor_ids)
        if not context.replay_scope.evaluation_eligible:
            return SpatialPolicyDecision(
                policy_id=self.policy_id,
                unavailable_mobile_sensor_ids=context.replay_scope.unavailable_mobile_sensor_ids,
                reason="Spatial selection disabled because calibrated/qualitative topology is unavailable for this campaign.",
            )
        limit = len(predicted) if context.branch_unresolved else context.maximum_active_sensors
        selected = tuple(
            sorted(
                predicted,
                key=lambda sensor: (
                    context.resource_load_by_sensor.get(sensor, 0.0),
                    predicted.index(sensor),
                    sensor,
                ),
            )[:limit]
        )
        return SpatialPolicyDecision(
            policy_id=self.policy_id,
            activated_sensor_ids=selected,
            predicted_sensor_ids=predicted,
            unavailable_mobile_sensor_ids=context.replay_scope.unavailable_mobile_sensor_ids,
            reason=(
                "Use hypothesis-specific topology predictions, preserving all overlapping candidates "
                "when the route branch is unresolved, then prefer lower-load feasible fixed Orin sensors."
            ),
        )


class OracleSpatialPolicy:
    policy_id = BaselineId.SPATIAL_ORACLE

    def select(self, context: SpatialSelectionContext) -> SpatialPolicyDecision:
        actual = context.actual_downstream_sensor_id
        supported = set(context.replay_scope.supported_sensor_ids)
        selected = (actual,) if actual in supported else ()
        return SpatialPolicyDecision(
            policy_id=self.policy_id,
            activated_sensor_ids=selected,
            predicted_sensor_ids=selected,
            unavailable_mobile_sensor_ids=context.replay_scope.unavailable_mobile_sensor_ids,
            reason=(
                "Activate the sensor that actually observes the entity; unavailable mobile targets "
                "are excluded until mobile replay support is implemented."
            ),
        )
