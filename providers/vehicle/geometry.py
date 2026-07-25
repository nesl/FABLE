"""Geometry, motion, route, transition, and dwell predicate providers."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import hypot
from typing import Iterable

from pydantic import Field, model_validator

from fable.common.base import FableModel
from fable.common.time import EventTimeInterval

from .errors import ArtifactCompatibilityError, InvalidProviderInput
from .models import (
    Point2D,
    PredicateObservation,
    TrackObservation,
    TrackSet,
    VehicleZone,
    occurrence_id,
)


class ReferenceLine(FableModel):
    reference_id: str = Field(min_length=1)
    start: Point2D
    end: Point2D
    direction: int = Field(default=0, ge=-1, le=1)

    @model_validator(mode="after")
    def _frame(self) -> "ReferenceLine":
        if self.start.coordinate_frame_id != self.end.coordinate_frame_id:
            raise ValueError("reference-line endpoints must share a coordinate frame")
        if self.start.x == self.end.x and self.start.y == self.end.y:
            raise ValueError("reference line cannot have zero length")
        return self


class RoutePolyline(FableModel):
    route_id: str = Field(min_length=1)
    coordinate_frame_id: str = Field(min_length=1)
    points: tuple[Point2D, ...]

    @model_validator(mode="after")
    def _valid(self) -> "RoutePolyline":
        if len(self.points) < 2:
            raise ValueError("route polyline requires at least two points")
        if any(point.coordinate_frame_id != self.coordinate_frame_id for point in self.points):
            raise ValueError("route points must share coordinate frame")
        return self


@dataclass(frozen=True)
class MotionState:
    scoped_track_id: str
    speed_mps: float
    moving: bool
    interval: EventTimeInterval


class TrackHistory:
    def __init__(self, *, horizon_seconds: float = 30.0) -> None:
        self.horizon = timedelta(seconds=horizon_seconds)
        self._by_track: dict[str, deque[TrackObservation]] = defaultdict(deque)

    def add(self, track_set: TrackSet) -> None:
        for observation in track_set.tracks:
            history = self._by_track[observation.scoped_track_id]
            history.append(observation)
            cutoff = observation.event_time - self.horizon
            while history and history[0].event_time < cutoff:
                history.popleft()

    def get(self, scoped_track_id: str) -> tuple[TrackObservation, ...]:
        return tuple(self._by_track.get(scoped_track_id, ()))


class ZoneMembershipEvaluator:
    provider_id = "zone_membership_evaluator"
    provider_version = "1"

    def evaluate(self, track: TrackObservation, zone: VehicleZone) -> PredicateObservation:
        point = _track_point(track)
        _require_frame(point, zone.coordinate_frame_id)
        truth = point_in_polygon(point, zone.polygon)
        interval = EventTimeInterval(start=track.event_time, end=track.event_time)
        bindings = {"vehicle": track.scoped_track_id, "zone": zone.zone_id}
        return PredicateObservation(
            occurrence_id=occurrence_id("INSIDE", bindings, interval, self.provider_id),
            predicate_id="INSIDE",
            truth=truth,
            confidence=track.confidence,
            event_time_interval=interval,
            bindings=bindings,
            source_ids=(track.source_id,),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            measurements={"x": point.x, "y": point.y},
        )


class ZoneTransitionEvaluator:
    provider_id = "zone_transition_evaluator"
    provider_version = "1"

    def __init__(self) -> None:
        self._membership: dict[tuple[str, str], tuple[bool, datetime]] = {}

    def update(self, track: TrackObservation, zone: VehicleZone) -> PredicateObservation | None:
        point = _track_point(track)
        _require_frame(point, zone.coordinate_frame_id)
        inside = point_in_polygon(point, zone.polygon)
        key = (track.scoped_track_id, zone.zone_id)
        previous = self._membership.get(key)
        self._membership[key] = (inside, track.event_time)
        if previous is None or previous[0] == inside:
            return None
        predicate_id = "ENTERS" if inside else "EXITS"
        interval = EventTimeInterval(start=previous[1], end=track.event_time)
        bindings = {"vehicle": track.scoped_track_id, "zone": zone.zone_id}
        return PredicateObservation(
            occurrence_id=occurrence_id(predicate_id, bindings, interval, self.provider_id),
            predicate_id=predicate_id,
            truth=True,
            confidence=track.confidence,
            event_time_interval=interval,
            bindings=bindings,
            source_ids=(track.source_id,),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            measurements={"entered": inside},
        )


class PassReferenceEvaluator:
    provider_id = "pass_reference_evaluator"
    provider_version = "1"

    def __init__(self) -> None:
        self._previous: dict[tuple[str, str], tuple[float, datetime]] = {}

    def update(self, track: TrackObservation, reference: ReferenceLine) -> PredicateObservation | None:
        point = _track_point(track)
        frame_id = reference.start.coordinate_frame_id
        _require_frame(point, frame_id)
        side = signed_line_side(point, reference)
        key = (track.scoped_track_id, reference.reference_id)
        previous = self._previous.get(key)
        self._previous[key] = (side, track.event_time)
        if previous is None or previous[0] == 0.0 or side == 0.0:
            return None
        crossed = (previous[0] < 0 < side) or (previous[0] > 0 > side)
        if not crossed:
            return None
        crossing_direction = 1 if side > previous[0] else -1
        if reference.direction and crossing_direction != reference.direction:
            return None
        interval = EventTimeInterval(start=previous[1], end=track.event_time)
        bindings = {
            "vehicle": track.scoped_track_id,
            "reference": reference.reference_id,
        }
        return PredicateObservation(
            occurrence_id=occurrence_id("PASSES", bindings, interval, self.provider_id),
            predicate_id="PASSES",
            truth=True,
            confidence=track.confidence,
            event_time_interval=interval,
            bindings=bindings,
            source_ids=(track.source_id,),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            supporting_artifact_types=("track_set.v1",),
            measurements={"crossing_direction": crossing_direction},
        )


class MotionStateEvaluator:
    provider_id = "motion_state_evaluator"
    provider_version = "1"

    def __init__(
        self,
        *,
        moving_threshold_mps: float = 0.75,
        stopped_threshold_mps: float = 0.25,
        minimum_window_s: float = 1.0,
        history_horizon_s: float = 20.0,
    ) -> None:
        if stopped_threshold_mps > moving_threshold_mps:
            raise ValueError("stopped threshold cannot exceed moving threshold")
        self.moving_threshold_mps = moving_threshold_mps
        self.stopped_threshold_mps = stopped_threshold_mps
        self.minimum_window_s = minimum_window_s
        self.history = TrackHistory(horizon_seconds=history_horizon_s)
        self._state: dict[str, bool] = {}

    def update(self, track_set: TrackSet) -> tuple[PredicateObservation, ...]:
        self.history.add(track_set)
        results: list[PredicateObservation] = []
        for track in track_set.tracks:
            history = self.history.get(track.scoped_track_id)
            state = self._estimate(history)
            if state is None:
                continue
            previous = self._state.get(track.scoped_track_id)
            moving = state.moving
            if previous is not None and self.stopped_threshold_mps < state.speed_mps < self.moving_threshold_mps:
                moving = previous
            self._state[track.scoped_track_id] = moving
            predicate_id = "MOVING" if moving else "STOPPED"
            bindings = {"vehicle": track.scoped_track_id}
            results.append(
                PredicateObservation(
                    occurrence_id=occurrence_id(predicate_id, bindings, state.interval, self.provider_id),
                    predicate_id=predicate_id,
                    truth=True,
                    confidence=track.confidence,
                    event_time_interval=state.interval,
                    bindings=bindings,
                    source_ids=(track.source_id,),
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                    supporting_artifact_types=("track_set.v1",),
                    measurements={"speed_mps": state.speed_mps},
                )
            )
        return tuple(results)

    def _estimate(self, history: tuple[TrackObservation, ...]) -> MotionState | None:
        if len(history) < 2:
            return None
        start = history[0]
        end = history[-1]
        elapsed = (end.event_time - start.event_time).total_seconds()
        if elapsed < self.minimum_window_s:
            return None
        left = _track_point(start)
        right = _track_point(end)
        _require_same_frame(left, right)
        speed = hypot(right.x - left.x, right.y - left.y) / elapsed
        moving = speed >= self.moving_threshold_mps
        return MotionState(
            scoped_track_id=end.scoped_track_id,
            speed_mps=speed,
            moving=moving,
            interval=EventTimeInterval(start=start.event_time, end=end.event_time),
        )


class PairwiseDistanceEvaluator:
    provider_id = "pairwise_distance_evaluator"
    provider_version = "1"

    def evaluate(
        self,
        left: TrackObservation,
        right: TrackObservation,
        *,
        maximum_distance_m: float,
    ) -> PredicateObservation:
        left_point = _track_point(left)
        right_point = _track_point(right)
        _require_same_frame(left_point, right_point)
        distance = hypot(right_point.x - left_point.x, right_point.y - left_point.y)
        timestamp = max(left.event_time, right.event_time)
        interval = EventTimeInterval(start=timestamp, end=timestamp)
        bindings = {"left": left.scoped_track_id, "right": right.scoped_track_id}
        return PredicateObservation(
            occurrence_id=occurrence_id("DISTANCE_LT", bindings, interval, self.provider_id),
            predicate_id="DISTANCE_LT",
            truth=distance <= maximum_distance_m,
            confidence=min(left.confidence, right.confidence),
            event_time_interval=interval,
            bindings=bindings,
            source_ids=tuple(sorted({left.source_id, right.source_id})),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            measurements={"distance_m": distance, "maximum_distance_m": maximum_distance_m},
        )


class RelativeOrderEvaluator:
    provider_id = "relative_order_evaluator"
    provider_version = "1"

    def evaluate_route_order(
        self,
        *,
        leader: TrackObservation,
        follower: TrackObservation,
        minimum_behind_m: float = 0.0,
    ) -> PredicateObservation:
        if leader.route_id is None or follower.route_id is None:
            raise InvalidProviderInput("route-relative order requires route assignments")
        if leader.route_id != follower.route_id:
            raise ArtifactCompatibilityError("route-relative order requires the same route")
        if leader.route_progress_m is None or follower.route_progress_m is None:
            raise InvalidProviderInput("route-relative order requires progress values")
        gap = leader.route_progress_m - follower.route_progress_m
        timestamp = max(leader.event_time, follower.event_time)
        interval = EventTimeInterval(start=timestamp, end=timestamp)
        bindings = {"leader": leader.scoped_track_id, "follower": follower.scoped_track_id}
        return PredicateObservation(
            occurrence_id=occurrence_id("BEHIND", bindings, interval, self.provider_id),
            predicate_id="BEHIND",
            truth=gap >= minimum_behind_m,
            confidence=min(leader.confidence, follower.confidence),
            event_time_interval=interval,
            bindings=bindings,
            source_ids=tuple(sorted({leader.source_id, follower.source_id})),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            measurements={"gap_m": gap, "minimum_behind_m": minimum_behind_m},
        )


class RouteMapMatcher:
    provider_id = "route_map_matcher"
    provider_version = "1"

    def match(self, track: TrackObservation, routes: Iterable[RoutePolyline]) -> TrackObservation:
        point = _track_point(track)
        candidates: list[tuple[float, float, RoutePolyline]] = []
        for route in routes:
            _require_frame(point, route.coordinate_frame_id)
            distance, progress = project_point_to_polyline(point, route.points)
            candidates.append((distance, progress, route))
        if not candidates:
            raise InvalidProviderInput("at least one route is required")
        distance, progress, route = min(candidates, key=lambda item: (item[0], item[2].route_id))
        return track.model_copy(
            update={
                "route_id": route.route_id,
                "route_progress_m": progress,
                "attributes": {**track.attributes, "route_distance_m": distance},
            }
        )


class DwellEvaluator:
    provider_id = "dwell_evaluator"
    provider_version = "1"

    def __init__(self) -> None:
        self._entered_at: dict[tuple[str, str], datetime] = {}
        self._emitted: set[tuple[str, str, datetime]] = set()

    def update(
        self,
        track: TrackObservation,
        zone: VehicleZone,
        *,
        minimum_duration_s: float,
    ) -> PredicateObservation | None:
        point = _track_point(track)
        _require_frame(point, zone.coordinate_frame_id)
        key = (track.scoped_track_id, zone.zone_id)
        inside = point_in_polygon(point, zone.polygon)
        if not inside:
            self._entered_at.pop(key, None)
            return None
        entered = self._entered_at.setdefault(key, track.event_time)
        elapsed = (track.event_time - entered).total_seconds()
        emitted_key = (track.scoped_track_id, zone.zone_id, entered)
        if elapsed < minimum_duration_s or emitted_key in self._emitted:
            return None
        self._emitted.add(emitted_key)
        interval = EventTimeInterval(start=entered, end=track.event_time)
        bindings = {"vehicle": track.scoped_track_id, "zone": zone.zone_id}
        return PredicateObservation(
            occurrence_id=occurrence_id("DWELLS", bindings, interval, self.provider_id),
            predicate_id="DWELLS",
            truth=True,
            confidence=track.confidence,
            event_time_interval=interval,
            bindings=bindings,
            source_ids=(track.source_id,),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            measurements={"duration_s": elapsed, "minimum_duration_s": minimum_duration_s},
        )


def point_in_polygon(point: Point2D, polygon: tuple[Point2D, ...]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        intersects = ((current.y > point.y) != (previous.y > point.y)) and (
            point.x
            < (previous.x - current.x) * (point.y - current.y)
            / ((previous.y - current.y) or 1e-12)
            + current.x
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def signed_line_side(point: Point2D, line: ReferenceLine) -> float:
    return (
        (line.end.x - line.start.x) * (point.y - line.start.y)
        - (line.end.y - line.start.y) * (point.x - line.start.x)
    )


def project_point_to_polyline(
    point: Point2D,
    polyline: tuple[Point2D, ...],
) -> tuple[float, float]:
    best_distance = float("inf")
    best_progress = 0.0
    cumulative = 0.0
    for left, right in zip(polyline, polyline[1:]):
        vx = right.x - left.x
        vy = right.y - left.y
        length_sq = vx * vx + vy * vy
        length = length_sq ** 0.5
        if length_sq == 0:
            continue
        t = max(0.0, min(1.0, ((point.x - left.x) * vx + (point.y - left.y) * vy) / length_sq))
        px = left.x + t * vx
        py = left.y + t * vy
        distance = hypot(point.x - px, point.y - py)
        progress = cumulative + t * length
        if (distance, progress) < (best_distance, best_progress):
            best_distance = distance
            best_progress = progress
        cumulative += length
    return best_distance, best_progress


def _track_point(track: TrackObservation) -> Point2D:
    if track.world_point is not None:
        return track.world_point
    x, y = track.bbox.center
    return Point2D(x=x, y=y, coordinate_frame_id=f"image:{track.source_id}")


def _require_frame(point: Point2D, frame_id: str) -> None:
    if point.coordinate_frame_id != frame_id:
        raise ArtifactCompatibilityError(
            f"coordinate frame {point.coordinate_frame_id!r} is incompatible with {frame_id!r}"
        )


def _require_same_frame(left: Point2D, right: Point2D) -> None:
    _require_frame(left, right.coordinate_frame_id)
