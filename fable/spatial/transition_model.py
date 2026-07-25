"""Load and query qualitative site sensor-transition knowledge."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from fable.common.ids import deterministic_id

from .models import (
    PredictedObservationGroup,
    SpatialCorridor,
    SpatialDirectionalRule,
    SpatialMatchKind,
    SpatialNextSensorCandidate,
    SpatialObservation,
    SpatialObservationGroup,
    SpatialPrediction,
    SpatialSensor,
    SpatialSensorBindings,
    SpatialTransitionModel,
)


_CONFIDENCE_SCORE = {
    "high": 1.0,
    "medium_high": 0.85,
    "medium-high": 0.85,
    "medium": 0.65,
    "low": 0.35,
}

_HEADING_ANGLE = {
    "N": 90.0,
    "N-NE": 67.5,
    "NE": 45.0,
    "E-NE": 22.5,
    "E": 0.0,
    "E-SE": 337.5,
    "SE": 315.0,
    "S-SE": 292.5,
    "S": 270.0,
    "S-SW": 247.5,
    "SW": 225.0,
    "W-SW": 202.5,
    "W": 180.0,
    "W-NW": 157.5,
    "NW": 135.0,
    "N-NW": 112.5,
}


class SpatialModelError(ValueError):
    """Raised when spatial knowledge is missing or internally inconsistent."""


def confidence_score(value: str) -> float:
    return _CONFIDENCE_SCORE.get(value.strip().lower(), 0.5)


def normalize_heading(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper().replace("_", "-").replace(" ", "")
    aliases = {
        "NORTH": "N",
        "SOUTH": "S",
        "EAST": "E",
        "WEST": "W",
        "NORTHEAST": "NE",
        "NORTHWEST": "NW",
        "SOUTHEAST": "SE",
        "SOUTHWEST": "SW",
    }
    return aliases.get(normalized, normalized)


def heading_from_vector(dx: float, dy: float) -> str:
    """Quantize a map-frame motion vector to an eight-way heading."""

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        raise ValueError("cannot infer heading from a zero motion vector")
    angle = math.degrees(math.atan2(dy, dx)) % 360.0
    labels = (
        (0.0, "E"),
        (45.0, "NE"),
        (90.0, "N"),
        (135.0, "NW"),
        (180.0, "W"),
        (225.0, "SW"),
        (270.0, "S"),
        (315.0, "SE"),
        (360.0, "E"),
    )
    return min(labels, key=lambda item: abs(item[0] - angle))[1]


class SiteSensorTransitionModel:
    def __init__(self, model: SpatialTransitionModel) -> None:
        self.model = model

    @classmethod
    def from_json(cls, path: str | Path) -> "SiteSensorTransitionModel":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise SpatialModelError("spatial transition model must be a JSON object")
        return cls(_normalize_document(raw))

    def predict(
        self,
        observation: SpatialObservation,
        *,
        bindings: SpatialSensorBindings | None = None,
    ) -> SpatialPrediction:
        bindings = bindings or SpatialSensorBindings()
        heading = normalize_heading(observation.observed_heading)
        warnings: list[str] = []
        direct = self._directional_groups(observation, heading)
        corridor_id = observation.corridor_id
        match_kind = SpatialMatchKind.NONE
        groups: list[tuple[tuple[str, ...], str, str]] = []
        if direct:
            match_kind = SpatialMatchKind.DIRECTIONAL_RULE
            groups, direct_corridor = direct
            corridor_id = corridor_id or direct_corridor
        else:
            corridor = self._select_corridor(observation, heading)
            if corridor is not None:
                match_kind = SpatialMatchKind.CORRIDOR
                corridor_id = corridor.corridor_id
                groups = self._corridor_groups(corridor, observation, heading)

        if not groups:
            warnings.append("no directional rule or corridor successor matched the observation")

        # A low-confidence or unresolved branch keeps one extra group as a
        # fallback, while a high-confidence resolved transition normally uses
        # only the immediate group.
        group_limit = observation.maximum_observation_groups
        if observation.branch_unresolved:
            group_limit = max(group_limit, 2)
        if groups and confidence_score(groups[0][1]) < 0.8:
            group_limit = max(group_limit, 2)
        groups = groups[:group_limit]

        predicted: list[PredictedObservationGroup] = []
        all_sources: list[str] = []
        all_nodes: list[str] = []
        for index, (sensor_ids, confidence, reason) in enumerate(groups, start=1):
            source_ids = tuple(
                dict.fromkeys(
                    source_id
                    for sensor_id in sensor_ids
                    for source_id in bindings.sources(
                        sensor_id, observation.active_deployment_id
                    )
                )
            )
            node_ids = tuple(
                dict.fromkeys(
                    node_id
                    for sensor_id in sensor_ids
                    for node_id in bindings.nodes(
                        sensor_id, observation.active_deployment_id
                    )
                )
            )
            for sensor_id in sensor_ids:
                if not bindings.sources(sensor_id, observation.active_deployment_id):
                    warnings.append(f"topology sensor {sensor_id} has no runtime source binding")
            all_sources.extend(source_ids)
            all_nodes.extend(node_ids)
            predicted.append(
                PredictedObservationGroup(
                    group_rank=index,
                    sensor_ids=sensor_ids,
                    source_ids=source_ids,
                    node_ids=node_ids,
                    confidence=confidence,
                    confidence_score=confidence_score(confidence),
                    reason=reason,
                )
            )

        payload = {
            "model": self.model.model_name,
            "current_sensor": observation.current_sensor_id,
            "heading": heading,
            "deployment": observation.active_deployment_id,
            "corridor": corridor_id,
            "groups": [item.model_dump(mode="json") for item in predicted],
        }
        return SpatialPrediction(
            prediction_id=deterministic_id("spatial", payload, length=32),
            match_kind=match_kind,
            current_sensor_id=observation.current_sensor_id,
            normalized_heading=heading,
            active_deployment_id=observation.active_deployment_id,
            corridor_id=corridor_id,
            groups=tuple(predicted),
            recommended_source_ids=tuple(dict.fromkeys(all_sources)),
            wake_node_ids=tuple(dict.fromkeys(all_nodes)),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _directional_groups(
        self,
        observation: SpatialObservation,
        heading: str | None,
    ) -> tuple[list[tuple[tuple[str, ...], str, str]], str | None] | None:
        if heading is None:
            return None
        matching_rules = [
            rule
            for rule in self.model.directional_rules
            if rule.current_sensor_id == observation.current_sensor_id
            and any(_headings_compatible(heading, candidate) for candidate in rule.observed_headings)
        ]
        if not matching_rules:
            return None
        candidates: list[SpatialNextSensorCandidate] = []
        corridor_id: str | None = None
        for rule in matching_rules:
            corridor_id = corridor_id or rule.corridor_id
            for candidate in rule.likely_next:
                if candidate.deployment_ids and observation.active_deployment_id not in candidate.deployment_ids:
                    continue
                candidates.append(candidate)
        if not candidates:
            return None
        grouped: dict[int, list[SpatialNextSensorCandidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.rank].append(candidate)
        result = []
        for rank in sorted(grouped):
            members = grouped[rank]
            sensor_ids = tuple(dict.fromkeys(item.sensor_id for item in members))
            confidence = max(members, key=lambda item: confidence_score(item.confidence)).confidence
            result.append(
                (
                    sensor_ids,
                    confidence,
                    f"directional transition from {observation.current_sensor_id} heading {heading}",
                )
            )
        return result, corridor_id

    def _select_corridor(
        self,
        observation: SpatialObservation,
        heading: str | None,
    ) -> SpatialCorridor | None:
        if observation.corridor_id is not None:
            try:
                return self.model.corridors[observation.corridor_id]
            except KeyError as exc:
                raise SpatialModelError(f"unknown corridor: {observation.corridor_id}") from exc
        candidates = []
        for corridor in self.model.corridors.values():
            forward = _group_index(corridor.forward_groups, observation.current_sensor_id)
            reverse = _group_index(corridor.reverse_groups, observation.current_sensor_id)
            if forward is None and reverse is None:
                continue
            score = confidence_score(corridor.confidence)
            if heading is not None and _motion_mentions_heading(corridor.motion_forward, heading):
                score += 0.2
            candidates.append((score, corridor.corridor_id, corridor))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))[2]

    def _corridor_groups(
        self,
        corridor: SpatialCorridor,
        observation: SpatialObservation,
        heading: str | None,
    ) -> list[tuple[tuple[str, ...], str, str]]:
        forward_index = _group_index(corridor.forward_groups, observation.current_sensor_id)
        reverse_index = _group_index(corridor.reverse_groups, observation.current_sensor_id)
        use_reverse = False
        groups = corridor.forward_groups
        current_index = forward_index
        if current_index is None and reverse_index is not None:
            groups = corridor.reverse_groups
            current_index = reverse_index
            use_reverse = True
        elif forward_index is not None and reverse_index is not None and heading is not None:
            # Prefer the ordering whose next group is directionally closer to
            # the observed heading using coarse sensor positions.
            forward_angle = self._next_group_angle(corridor.forward_groups, forward_index, observation.current_sensor_id)
            reverse_angle = self._next_group_angle(corridor.reverse_groups, reverse_index, observation.current_sensor_id)
            if reverse_angle is not None and (
                forward_angle is None
                or _angular_distance(_HEADING_ANGLE.get(heading), reverse_angle)
                < _angular_distance(_HEADING_ANGLE.get(heading), forward_angle)
            ):
                groups = corridor.reverse_groups
                current_index = reverse_index
                use_reverse = True
        if current_index is None or current_index + 1 >= len(groups):
            return []

        result: list[tuple[tuple[str, ...], str, str]] = []
        for next_group in groups[current_index + 1 :]:
            sensor_ids = list(next_group.sensor_ids)
            sensor_ids.extend(
                self._overlapping_mobile_sensors(
                    fixed_sensor_ids=next_group.sensor_ids,
                    deployment_id=observation.active_deployment_id,
                    corridor=corridor,
                )
            )
            direction = "reverse" if use_reverse else "forward"
            result.append(
                (
                    tuple(dict.fromkeys(sensor_ids)),
                    corridor.confidence,
                    f"{direction} successor group on corridor {corridor.corridor_id}",
                )
            )
        return result

    def _overlapping_mobile_sensors(
        self,
        *,
        fixed_sensor_ids: tuple[str, ...],
        deployment_id: str | None,
        corridor: SpatialCorridor,
    ) -> tuple[str, ...]:
        if deployment_id is None:
            return ()
        augmentations = corridor.mobile_augmentations.get(deployment_id, ())
        fixed_zones = {
            zone
            for sensor_id in fixed_sensor_ids
            for zone in self.model.sensors.get(
                sensor_id,
                SpatialSensor(sensor_id=sensor_id, position=(0.0, 0.0)),
            ).coverage_zones
        }
        result = []
        for group in augmentations:
            for sensor_id in group.sensor_ids:
                sensor = self.model.sensors.get(sensor_id)
                if sensor is not None and (not fixed_zones or set(sensor.coverage_zones) & fixed_zones):
                    result.append(sensor_id)
        return tuple(dict.fromkeys(result))

    def _next_group_angle(
        self,
        groups: tuple[SpatialObservationGroup, ...],
        current_index: int,
        current_sensor_id: str,
    ) -> float | None:
        if current_index + 1 >= len(groups):
            return None
        current = self.model.sensors.get(current_sensor_id)
        if current is None:
            return None
        positions = [
            self.model.sensors[sensor_id].position
            for sensor_id in groups[current_index + 1].sensor_ids
            if sensor_id in self.model.sensors
        ]
        if not positions:
            return None
        x = sum(item[0] for item in positions) / len(positions)
        y = sum(item[1] for item in positions) / len(positions)
        return math.degrees(math.atan2(y - current.position[1], x - current.position[0])) % 360.0


def load_sensor_bindings(path: str | Path) -> SpatialSensorBindings:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise SpatialModelError("spatial binding configuration must be a mapping")
    sensors = raw.get("sensors", {})
    sources: dict[str, tuple[str, ...]] = {}
    nodes: dict[str, tuple[str, ...]] = {}
    for sensor_id, value in sensors.items():
        if not isinstance(value, dict):
            raise SpatialModelError(f"sensor binding {sensor_id} must be a mapping")
        sources[sensor_id] = tuple(value.get("source_ids", ()))
        nodes[sensor_id] = tuple(value.get("node_ids", ()))
    deployment_sources: dict[str, dict[str, tuple[str, ...]]] = {}
    deployment_nodes: dict[str, dict[str, tuple[str, ...]]] = {}
    for deployment_id, deployment in raw.get("deployments", {}).items():
        deployment_sources[deployment_id] = {}
        deployment_nodes[deployment_id] = {}
        for sensor_id, value in deployment.get("sensors", {}).items():
            if not isinstance(value, dict):
                raise SpatialModelError(
                    f"deployment sensor binding {deployment_id}/{sensor_id} must be a mapping"
                )
            deployment_sources[deployment_id][sensor_id] = tuple(
                value.get("source_ids", ())
            )
            deployment_nodes[deployment_id][sensor_id] = tuple(
                value.get("node_ids", ())
            )
    return SpatialSensorBindings(
        source_ids_by_sensor=sources,
        node_ids_by_sensor=nodes,
        source_ids_by_deployment=deployment_sources,
        node_ids_by_deployment=deployment_nodes,
    )


def _normalize_document(raw: dict[str, Any]) -> SpatialTransitionModel:
    sensors: dict[str, SpatialSensor] = {}
    for sensor_id, value in raw.get("fixed_sensors", {}).items():
        sensors[sensor_id] = SpatialSensor(
            sensor_id=sensor_id,
            position=tuple(float(item) for item in value["position"]),
            coverage_zones=tuple(value.get("primary_zones", ())),
            camera_facing_approx=value.get("camera_facing_approx"),
            confidence=value.get("facing_confidence", "medium"),
            microphone=bool(value.get("microphone", False)),
            fixed=True,
        )
    mobile_deployments = raw.get("mobile_deployments", {})
    for deployment_id, deployment in mobile_deployments.items():
        for sensor_id, value in deployment.get("nodes", {}).items():
            existing = sensors.get(sensor_id)
            deployment_ids = (
                tuple(dict.fromkeys((*existing.deployment_ids, deployment_id)))
                if existing is not None
                else (deployment_id,)
            )
            sensors[sensor_id] = SpatialSensor(
                sensor_id=sensor_id,
                position=tuple(float(item) for item in value["position"]),
                coverage_zones=tuple(value.get("coverage_zones", ())),
                camera_facing_approx=value.get("camera_facing_approx"),
                confidence=value.get("confidence", "medium"),
                microphone=True,
                fixed=False,
                deployment_ids=deployment_ids,
            )

    corridors = {}
    for corridor_id, value in raw.get("corridors", {}).items():
        corridors[corridor_id] = SpatialCorridor(
            corridor_id=corridor_id,
            from_zone=value["from_zone"],
            to_zone=value["to_zone"],
            reverse_supported=bool(value.get("reverse_supported", False)),
            motion_forward=value.get("motion_forward", ""),
            forward_groups=tuple(
                SpatialObservationGroup(sensor_ids=tuple(group))
                for group in value.get("fixed_observation_groups_forward", ())
            ),
            reverse_groups=tuple(
                SpatialObservationGroup(sensor_ids=tuple(group))
                for group in value.get("fixed_observation_groups_reverse", ())
            ),
            mobile_augmentations={
                deployment_id: tuple(
                    SpatialObservationGroup(sensor_ids=tuple(group)) for group in groups
                )
                for deployment_id, groups in value.get("mobile_augmentations", {}).items()
            },
            confidence=value.get("confidence", "medium"),
            note=value.get("note", ""),
        )

    rules = tuple(
        SpatialDirectionalRule(
            current_sensor_id=value["current_sensor"],
            observed_headings=tuple(normalize_heading(item) or item for item in value["observed_heading"]),
            likely_next=tuple(
                SpatialNextSensorCandidate(
                    sensor_id=candidate["sensor"],
                    rank=int(candidate["rank"]),
                    confidence=candidate.get("confidence", "medium"),
                    deployment_ids=tuple(candidate.get("deployment_ids", ())),
                )
                for candidate in value.get("likely_next", ())
            ),
            corridor_id=value.get("corridor"),
        )
        for value in raw.get("directional_next_sensor_rules", ())
    )
    zones = {
        zone_id: tuple(float(item) for item in value["center"])
        for zone_id, value in raw.get("zones", {}).items()
    }
    return SpatialTransitionModel(
        schema_version=str(raw.get("schema_version", "unknown")),
        model_name=str(raw.get("model_name", "unnamed spatial transition model")),
        model_type=str(raw.get("model_type", "unknown")),
        zones=zones,
        sensors=sensors,
        mobile_deployments=tuple(sorted(mobile_deployments)),
        corridors=corridors,
        directional_rules=rules,
        assumptions=tuple(raw.get("assumptions", ())),
        known_issues=tuple(raw.get("known_issues", ())),
    )


def _group_index(groups: tuple[SpatialObservationGroup, ...], sensor_id: str) -> int | None:
    for index, group in enumerate(groups):
        if sensor_id in group.sensor_ids:
            return index
    return None


def _headings_compatible(left: str, right: str) -> bool:
    left = normalize_heading(left) or left
    right = normalize_heading(right) or right
    if left == right:
        return True
    left_angle = _HEADING_ANGLE.get(left)
    right_angle = _HEADING_ANGLE.get(right)
    if left_angle is None or right_angle is None:
        return False
    return _angular_distance(left_angle, right_angle) <= 45.0


def _angular_distance(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return 360.0
    delta = abs(left - right) % 360.0
    return min(delta, 360.0 - delta)


def _motion_mentions_heading(text: str, heading: str) -> bool:
    normalized = text.upper()
    return heading in normalized or {
        "N": "NORTH",
        "S": "SOUTH",
        "E": "EAST",
        "W": "WEST",
        "NE": "NORTHEAST",
        "NW": "NORTHWEST",
        "SE": "SOUTHEAST",
        "SW": "SOUTHWEST",
    }.get(heading, heading) in normalized
