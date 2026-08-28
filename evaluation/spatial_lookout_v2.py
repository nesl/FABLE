"""Isolated RQ3b-v2 policy and lookout-intent construction.

This module is intentionally not imported by the active RQ3a execution path.
It converts a semantic checkpoint's bound object, current sensor, heading, and
route into ranked downstream sensor activation intents.  A later live adapter
may translate these intents into the existing typed provider lease commands.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from fable.common.base import FableModel
from fable.spatial import (
    SiteSensorTransitionModel,
    SpatialObservation,
    SpatialPrediction,
    SpatialSensorBindings,
)


class SpatialLookoutPolicyId(StrEnum):
    S0_NO_HANDOFF = "S0_NO_HANDOFF"
    S1_BROADCAST = "S1_BROADCAST"
    S2_TOPOLOGY_ONLY = "S2_TOPOLOGY_ONLY"
    S3_RESOURCE_AWARE_TOPOLOGY = "S3_RESOURCE_AWARE_TOPOLOGY"
    FABLE_SPATIAL = "FABLE_SPATIAL"
    O_SPACE = "O_SPACE"


class SensorOperatingState(FableModel):
    sensor_id: str = Field(min_length=1)
    available: bool = True
    replay_supported: bool = True
    normalized_load: float = Field(default=0.0, ge=0.0, le=1.0)
    provider_ready_ms: int = Field(default=0, ge=0)


class SpatialLookoutContext(FableModel):
    prediction: SpatialPrediction
    object_binding_id: str | None = None
    semantic_frontier_id: str = Field(min_length=1)
    prediction_time: datetime
    operating_state_by_sensor: dict[str, SensorOperatingState]
    maximum_active_sensors: int = Field(default=2, ge=1)
    branch_unresolved: bool = False
    actual_downstream_sensor_id: str | None = None


class RankedLookoutSensor(FableModel):
    sensor_id: str
    topology_group_rank: int
    topology_confidence: float
    normalized_load: float
    provider_ready_ms: int
    selected: bool = False
    exclusion_reason: str | None = None


class SpatialLookoutDecision(FableModel):
    policy_id: SpatialLookoutPolicyId
    activated_sensor_ids: tuple[str, ...] = ()
    ranked_sensors: tuple[RankedLookoutSensor, ...] = ()
    object_binding_id: str | None = None
    semantic_frontier_id: str
    reason: str


class LookoutActivationIntent(FableModel):
    action: str = Field(pattern=r"^(ACTIVATE|RELEASE)$")
    sensor_id: str
    source_ids: tuple[str, ...] = ()
    node_ids: tuple[str, ...] = ()
    object_binding_id: str
    semantic_frontier_id: str
    prediction_id: str
    prediction_time: datetime


class SpatialLookoutUpdate(FableModel):
    decision: SpatialLookoutDecision
    intents: tuple[LookoutActivationIntent, ...]


class SpatialLookoutCoordinator:
    """Create policy decisions and lease-like deltas at frontier boundaries."""

    def __init__(
        self,
        *,
        model: SiteSensorTransitionModel,
        bindings: SpatialSensorBindings,
    ) -> None:
        self.model = model
        self.bindings = bindings

    def update(
        self,
        *,
        policy_id: SpatialLookoutPolicyId,
        observation: SpatialObservation,
        semantic_frontier_id: str,
        prediction_time: datetime,
        operating_state_by_sensor: dict[str, SensorOperatingState],
        previously_active_sensor_ids: tuple[str, ...] = (),
        maximum_active_sensors: int = 2,
        actual_downstream_sensor_id: str | None = None,
    ) -> SpatialLookoutUpdate:
        prediction = self.model.predict(observation, bindings=self.bindings)
        context = SpatialLookoutContext(
            prediction=prediction,
            object_binding_id=observation.object_binding_id,
            semantic_frontier_id=semantic_frontier_id,
            prediction_time=prediction_time,
            operating_state_by_sensor=operating_state_by_sensor,
            maximum_active_sensors=maximum_active_sensors,
            branch_unresolved=observation.branch_unresolved,
            actual_downstream_sensor_id=actual_downstream_sensor_id,
        )
        decision = select_spatial_lookout(policy_id, context)
        if not decision.object_binding_id:
            return SpatialLookoutUpdate(decision=decision, intents=())
        selected = set(decision.activated_sensor_ids)
        previous = set(previously_active_sensor_ids)
        intents = []
        for action, sensor_ids in (
            ("RELEASE", sorted(previous - selected)),
            ("ACTIVATE", sorted(selected - previous)),
        ):
            for sensor_id in sensor_ids:
                intents.append(
                    LookoutActivationIntent(
                        action=action,
                        sensor_id=sensor_id,
                        source_ids=self.bindings.sources(
                            sensor_id, observation.active_deployment_id
                        ),
                        node_ids=self.bindings.nodes(
                            sensor_id, observation.active_deployment_id
                        ),
                        object_binding_id=decision.object_binding_id,
                        semantic_frontier_id=semantic_frontier_id,
                        prediction_id=prediction.prediction_id,
                        prediction_time=prediction_time,
                    )
                )
        return SpatialLookoutUpdate(decision=decision, intents=tuple(intents))


def select_spatial_lookout(
    policy_id: SpatialLookoutPolicyId,
    context: SpatialLookoutContext,
) -> SpatialLookoutDecision:
    candidates = _ranked_candidates(context)
    available_all = tuple(
        sorted(
            sensor_id
            for sensor_id, state in context.operating_state_by_sensor.items()
            if state.available and state.replay_supported
        )
    )
    if policy_id == SpatialLookoutPolicyId.S0_NO_HANDOFF:
        selected: tuple[str, ...] = ()
        reason = "Do not activate a downstream lookout after upstream progress."
    elif policy_id == SpatialLookoutPolicyId.S1_BROADCAST:
        selected = available_all
        reason = "Activate every available replay-supported downstream sensor."
    elif policy_id == SpatialLookoutPolicyId.S2_TOPOLOGY_ONLY:
        selected = tuple(item.sensor_id for item in candidates)
        reason = "Use all topology successors without resource or identity ranking."
    elif policy_id == SpatialLookoutPolicyId.S3_RESOURCE_AWARE_TOPOLOGY:
        selected = tuple(
            item.sensor_id
            for item in sorted(
                candidates,
                key=lambda item: (
                    item.normalized_load,
                    item.provider_ready_ms,
                    item.topology_group_rank,
                    item.sensor_id,
                ),
            )[: context.maximum_active_sensors]
        )
        reason = "Choose low-load topology candidates without using entity identity."
    elif policy_id == SpatialLookoutPolicyId.FABLE_SPATIAL:
        if not context.object_binding_id:
            selected = ()
            reason = "Reject predictive handoff because no canonical object binding is available."
        else:
            limit = (
                max(context.maximum_active_sensors, len(candidates))
                if context.branch_unresolved
                else context.maximum_active_sensors
            )
            selected = tuple(item.sensor_id for item in candidates[:limit])
            reason = (
                "Rank successors by direction/corridor group, topology confidence, "
                "provider readiness, and load for the bound object."
            )
    elif policy_id == SpatialLookoutPolicyId.O_SPACE:
        actual = context.actual_downstream_sensor_id
        selected = (actual,) if actual in available_all else ()
        reason = "Use the annotated future sensor as a non-online upper bound."
    else:  # pragma: no cover - exhaustive StrEnum boundary
        raise ValueError(f"unsupported spatial lookout policy: {policy_id}")

    selected_set = set(selected)
    ranked = tuple(
        item.model_copy(update={"selected": item.sensor_id in selected_set})
        for item in candidates
    )
    return SpatialLookoutDecision(
        policy_id=policy_id,
        activated_sensor_ids=selected,
        ranked_sensors=ranked,
        object_binding_id=context.object_binding_id,
        semantic_frontier_id=context.semantic_frontier_id,
        reason=reason,
    )


def _ranked_candidates(
    context: SpatialLookoutContext,
) -> tuple[RankedLookoutSensor, ...]:
    candidates = []
    seen = set()
    for group in context.prediction.groups:
        for sensor_id in group.sensor_ids:
            if sensor_id in seen:
                continue
            seen.add(sensor_id)
            state = context.operating_state_by_sensor.get(sensor_id)
            if state is None or not state.available or not state.replay_supported:
                continue
            candidates.append(
                RankedLookoutSensor(
                    sensor_id=sensor_id,
                    topology_group_rank=group.group_rank,
                    topology_confidence=group.confidence_score,
                    normalized_load=state.normalized_load,
                    provider_ready_ms=state.provider_ready_ms,
                )
            )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.topology_group_rank,
                -item.topology_confidence,
                item.provider_ready_ms,
                item.normalized_load,
                item.sensor_id,
            ),
        )
    )
