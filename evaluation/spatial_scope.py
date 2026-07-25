"""Evaluation-time guards for topology availability and fixed-Orin replay support."""

from __future__ import annotations

from pydantic import Field

from fable.common.base import FableModel
from fable.spatial.models import SpatialPrediction
from fable.spatial.transition_model import SiteSensorTransitionModel


class SpatialReplaySelection(FableModel):
    campaign_year: int
    topology_available: bool
    evaluation_eligible: bool
    prediction_id: str | None = None
    all_replay_sensor_ids: tuple[str, ...] = ()
    supported_sensor_ids: tuple[str, ...] = ()
    supported_source_ids: tuple[str, ...] = ()
    supported_node_ids: tuple[str, ...] = ()
    unavailable_mobile_sensor_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class ReplaySpatialScope:
    """Restrict spatial evaluation to known topology and currently replayed sensors."""

    def __init__(self, transition_model: SiteSensorTransitionModel) -> None:
        self.transition_model = transition_model

    def apply(
        self,
        prediction: SpatialPrediction,
        *,
        campaign_year: int,
        available_replay_sensor_ids: tuple[str, ...] | None = None,
    ) -> SpatialReplaySelection:
        if campaign_year not in (2024, 2025):
            return SpatialReplaySelection(
                campaign_year=campaign_year,
                topology_available=False,
                evaluation_eligible=False,
                prediction_id=prediction.prediction_id,
                all_replay_sensor_ids=(),
                warnings=(
                    "Topology-based spatial coordination is disabled because sensor locations "
                    "are not available for this campaign.",
                ),
            )
        supported_sensors: list[str] = []
        mobile_sensors: list[str] = []
        supported_sources: list[str] = []
        supported_nodes: list[str] = []
        for group in prediction.groups:
            for sensor_id in group.sensor_ids:
                sensor = self.transition_model.model.sensors.get(sensor_id)
                if sensor is not None and sensor.fixed and sensor_id.startswith("orin_"):
                    supported_sensors.append(sensor_id)
                    # Prediction groups already contain resolved runtime source/node IDs for
                    # all members. Keep only identifiers whose sensor token is fixed Orin.
                else:
                    mobile_sensors.append(sensor_id)
            for source_id in group.source_ids:
                if "orin" in source_id.lower():
                    supported_sources.append(source_id)
            for node_id in group.node_ids:
                if "orin" in node_id.lower():
                    supported_nodes.append(node_id)
        warnings = list(prediction.warnings)
        if mobile_sensors:
            warnings.append(
                "Mobile sensor candidates were predicted but excluded because replay containers "
                "currently cover fixed Orin devices only."
            )
        all_fixed_set = {
            sensor_id for sensor_id, sensor in self.transition_model.model.sensors.items()
            if sensor.fixed and sensor_id.startswith("orin_")
        }
        if available_replay_sensor_ids is not None:
            all_fixed_set &= set(available_replay_sensor_ids)
        all_fixed = tuple(sorted(all_fixed_set))
        supported_sensors = [item for item in supported_sensors if item in all_fixed_set]
        return SpatialReplaySelection(
            campaign_year=campaign_year,
            topology_available=True,
            evaluation_eligible=True,
            prediction_id=prediction.prediction_id,
            all_replay_sensor_ids=all_fixed,
            supported_sensor_ids=tuple(dict.fromkeys(supported_sensors)),
            supported_source_ids=tuple(dict.fromkeys(supported_sources)),
            supported_node_ids=tuple(dict.fromkeys(supported_nodes)),
            unavailable_mobile_sensor_ids=tuple(dict.fromkeys(mobile_sensors)),
            warnings=tuple(dict.fromkeys(warnings)),
        )
