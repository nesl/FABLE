"""Person/vehicle transition reasoning for DISEMBARKS and BOARDS."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import hypot

from fable.common.time import EventTimeInterval
from providers.vehicle.models import TrackObservation, TrackSet

from .errors import InteractionStateError
from .models import InteractionPredicateObservation, phase8_occurrence_id


PERSON_CLASSES = frozenset({"person"})
VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "vehicle"})


@dataclass
class _DisembarkCandidate:
    person_id: str
    vehicle_id: str
    started_at: datetime
    initial_distance: float


@dataclass
class _BoardCandidate:
    person_id: str
    vehicle_id: str
    person_last_seen_at: datetime
    vehicle_position_at_disappearance: tuple[float, float]


class PersonVehicleRelationEvaluator:
    """Stateful geometric evidence for person/vehicle transitions.

    ``DISEMBARKS`` requires a newly visible person near a stopped vehicle who
    subsequently separates from it. ``BOARDS`` requires a person to disappear
    while close to a vehicle followed by compatible vehicle departure.  These
    are explicit provider semantics, not hidden CE-transition rules.
    """

    def __init__(
        self,
        *,
        proximity_m: float = 3.0,
        separation_m: float = 5.0,
        vehicle_departure_m: float = 2.0,
        stopped_speed_mps: float = 0.75,
        minimum_stop_seconds: float = 0.5,
        transition_window_seconds: float = 8.0,
        provider_id: str = "person_vehicle_relation_provider",
        provider_version: str = "1",
    ) -> None:
        if min(
            proximity_m,
            separation_m,
            vehicle_departure_m,
            stopped_speed_mps,
            minimum_stop_seconds,
            transition_window_seconds,
        ) < 0:
            raise ValueError("person/vehicle thresholds cannot be negative")
        if separation_m <= proximity_m:
            raise ValueError("separation_m must exceed proximity_m")
        self.proximity_m = proximity_m
        self.separation_m = separation_m
        self.vehicle_departure_m = vehicle_departure_m
        self.stopped_speed_mps = stopped_speed_mps
        self.minimum_stop_seconds = minimum_stop_seconds
        self.transition_window_seconds = transition_window_seconds
        self.provider_id = provider_id
        self.provider_version = provider_version
        self._previous: dict[str, TrackObservation] = {}
        self._last_event_time: datetime | None = None
        self._vehicle_stopped_since: dict[str, datetime] = {}
        self._disembark: dict[str, _DisembarkCandidate] = {}
        self._boarding: dict[str, _BoardCandidate] = {}
        self._emitted: set[tuple[str, str, str]] = set()

    def update(self, tracks: TrackSet) -> tuple[InteractionPredicateObservation, ...]:
        now = tracks.event_time
        if self._last_event_time is not None and now < self._last_event_time:
            raise InteractionStateError("person/vehicle input must be ordered by event time")
        current = {track.scoped_track_id: track for track in tracks.tracks}
        persons = {
            track_id: track
            for track_id, track in current.items()
            if track.class_name.lower() in PERSON_CLASSES
        }
        vehicles = {
            track_id: track
            for track_id, track in current.items()
            if track.class_name.lower() in VEHICLE_CLASSES
        }
        previous_persons = {
            track_id: track
            for track_id, track in self._previous.items()
            if track.class_name.lower() in PERSON_CLASSES
        }
        previous_vehicles = {
            track_id: track
            for track_id, track in self._previous.items()
            if track.class_name.lower() in VEHICLE_CLASSES
        }
        self._update_vehicle_stop_state(vehicles, previous_vehicles, now)
        observations: list[InteractionPredicateObservation] = []

        # A new person near a sufficiently stopped vehicle may be emerging from it.
        for person_id in sorted(set(persons) - set(previous_persons)):
            nearest = _nearest_track(persons[person_id], vehicles.values())
            if nearest is None:
                continue
            vehicle, distance = nearest
            stopped_since = self._vehicle_stopped_since.get(vehicle.scoped_track_id)
            stopped_long_enough = (
                stopped_since is not None
                and (now - stopped_since).total_seconds() >= self.minimum_stop_seconds
            )
            if distance <= self.proximity_m and stopped_long_enough:
                self._disembark[person_id] = _DisembarkCandidate(
                    person_id=person_id,
                    vehicle_id=vehicle.scoped_track_id,
                    started_at=now,
                    initial_distance=distance,
                )

        # Confirm disembarkation only after visible separation from the same vehicle.
        for person_id, candidate in tuple(self._disembark.items()):
            person = persons.get(person_id)
            vehicle = vehicles.get(candidate.vehicle_id)
            age = (now - candidate.started_at).total_seconds()
            if age > self.transition_window_seconds:
                self._disembark.pop(person_id, None)
                continue
            if person is None or vehicle is None:
                continue
            distance = _track_distance(person, vehicle)
            if distance < self.separation_m:
                continue
            key = ("DISEMBARKS", person_id, candidate.vehicle_id)
            if key not in self._emitted:
                interval = EventTimeInterval(start=candidate.started_at, end=now)
                observations.append(
                    self._observation(
                        "DISEMBARKS",
                        interval,
                        {"person": person_id, "vehicle": candidate.vehicle_id},
                        confidence=_transition_confidence(
                            candidate.initial_distance,
                            distance,
                            self.proximity_m,
                            self.separation_m,
                        ),
                        measurements={
                            "initial_distance_m": candidate.initial_distance,
                            "final_distance_m": distance,
                        },
                        source_ids=(tracks.source_id,),
                    )
                )
                self._emitted.add(key)
            self._disembark.pop(person_id, None)

        # A person disappearing while adjacent to a vehicle creates a pending board
        # candidate. The next vehicle motion confirms rather than merely assumes it.
        for person_id in sorted(set(previous_persons) - set(persons)):
            nearest = _nearest_track(previous_persons[person_id], previous_vehicles.values())
            if nearest is None:
                continue
            vehicle, distance = nearest
            if distance <= self.proximity_m:
                self._boarding[person_id] = _BoardCandidate(
                    person_id=person_id,
                    vehicle_id=vehicle.scoped_track_id,
                    person_last_seen_at=previous_persons[person_id].event_time,
                    vehicle_position_at_disappearance=_track_point(vehicle),
                )

        for person_id, candidate in tuple(self._boarding.items()):
            age = (now - candidate.person_last_seen_at).total_seconds()
            if age > self.transition_window_seconds:
                self._boarding.pop(person_id, None)
                continue
            vehicle = vehicles.get(candidate.vehicle_id)
            if vehicle is None:
                continue
            moved = _point_distance(
                candidate.vehicle_position_at_disappearance,
                _track_point(vehicle),
            )
            speed = abs(float(vehicle.velocity_mps or 0.0))
            if moved < self.vehicle_departure_m and speed <= self.stopped_speed_mps:
                continue
            key = ("BOARDS", person_id, candidate.vehicle_id)
            if key not in self._emitted:
                interval = EventTimeInterval(start=candidate.person_last_seen_at, end=now)
                observations.append(
                    self._observation(
                        "BOARDS",
                        interval,
                        {"person": person_id, "vehicle": candidate.vehicle_id},
                        confidence=max(
                            0.0,
                            min(1.0, 0.6 + 0.4 * min(1.0, moved / max(self.vehicle_departure_m, 1e-6))),
                        ),
                        measurements={
                            "vehicle_departure_m": moved,
                            "vehicle_speed_mps": speed,
                        },
                        source_ids=(tracks.source_id,),
                    )
                )
                self._emitted.add(key)
            self._boarding.pop(person_id, None)

        self._previous = current
        self._last_event_time = now
        return tuple(observations)

    def _update_vehicle_stop_state(
        self,
        vehicles: dict[str, TrackObservation],
        previous: dict[str, TrackObservation],
        now: datetime,
    ) -> None:
        for vehicle_id, vehicle in vehicles.items():
            speed = vehicle.velocity_mps
            if speed is None and vehicle_id in previous:
                delta = (vehicle.event_time - previous[vehicle_id].event_time).total_seconds()
                if delta > 0:
                    speed = _track_distance(vehicle, previous[vehicle_id]) / delta
            stopped = speed is None or abs(float(speed)) <= self.stopped_speed_mps
            if stopped:
                self._vehicle_stopped_since.setdefault(vehicle_id, now)
            else:
                self._vehicle_stopped_since.pop(vehicle_id, None)
        for missing in set(self._vehicle_stopped_since) - set(vehicles):
            self._vehicle_stopped_since.pop(missing, None)

    def _observation(
        self,
        predicate_id: str,
        interval: EventTimeInterval,
        bindings: dict[str, str],
        *,
        confidence: float,
        measurements: dict[str, float],
        source_ids: tuple[str, ...],
    ) -> InteractionPredicateObservation:
        return InteractionPredicateObservation(
            occurrence_id=phase8_occurrence_id(
                predicate_id,
                bindings,
                interval,
                self.provider_id,
            ),
            predicate_id=predicate_id,
            truth=True,
            confidence=confidence,
            event_time_interval=interval,
            bindings=bindings,
            source_ids=source_ids,
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            supporting_artifact_types=("track_set.v1", "track_summary.v1"),
            measurements=measurements,
        )


def _track_point(track: TrackObservation) -> tuple[float, float]:
    if track.world_point is not None:
        return (track.world_point.x, track.world_point.y)
    return track.bbox.center


def _track_distance(left: TrackObservation, right: TrackObservation) -> float:
    if left.world_point is not None and right.world_point is not None:
        if left.world_point.coordinate_frame_id != right.world_point.coordinate_frame_id:
            raise InteractionStateError("tracks use incompatible coordinate frames")
    return _point_distance(_track_point(left), _track_point(right))


def _point_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def _nearest_track(
    target: TrackObservation,
    candidates: object,
) -> tuple[TrackObservation, float] | None:
    values = list(candidates)  # type: ignore[arg-type]
    if not values:
        return None
    ranked = sorted(
        ((_track_distance(target, candidate), candidate) for candidate in values),
        key=lambda item: (item[0], item[1].scoped_track_id),
    )
    distance, candidate = ranked[0]
    return candidate, distance


def _transition_confidence(
    initial: float,
    final: float,
    proximity: float,
    separation: float,
) -> float:
    near = max(0.0, min(1.0, 1.0 - initial / max(proximity, 1e-6)))
    away = max(0.0, min(1.0, (final - proximity) / max(separation - proximity, 1e-6)))
    return max(0.0, min(1.0, 0.5 + 0.25 * near + 0.25 * away))
