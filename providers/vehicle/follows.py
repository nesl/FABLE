"""Same-camera and route-relative FOLLOWS predicate providers."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import timedelta
from math import hypot

from fable.common.time import EventTimeInterval

from .errors import ArtifactCompatibilityError, InvalidProviderInput
from .models import PredicateObservation, TrackObservation, TrackSet, occurrence_id


class FollowsLocalGeometryEvaluator:
    """Evaluate sustained leader/follower geometry in a common coordinate frame.

    The evaluator keeps only compact track observations. It does not own CE
    state, bind canonical cross-camera identities, or infer a convoy by itself.
    A true result means that one concrete pair satisfied the configured local
    geometric relation for the requested duration.
    """

    provider_id = "follows_local_geometry"
    provider_version = "1"

    def __init__(
        self,
        *,
        maximum_gap_m: float = 15.0,
        minimum_gap_m: float = 0.0,
        minimum_duration_s: float = 1.0,
        maximum_time_skew_s: float = 0.5,
        history_horizon_s: float = 30.0,
    ) -> None:
        if maximum_gap_m < minimum_gap_m or minimum_gap_m < 0:
            raise ValueError("FOLLOWS gap bounds are invalid")
        if minimum_duration_s < 0 or maximum_time_skew_s < 0:
            raise ValueError("duration and time skew must be non-negative")
        self.maximum_gap_m = float(maximum_gap_m)
        self.minimum_gap_m = float(minimum_gap_m)
        self.minimum_duration_s = float(minimum_duration_s)
        self.maximum_time_skew_s = float(maximum_time_skew_s)
        self._history_horizon = timedelta(seconds=history_horizon_s)
        self._pairs: dict[
            tuple[str, str], deque[tuple[TrackObservation, TrackObservation, float]]
        ] = defaultdict(deque)
        self._emitted_until: dict[tuple[str, str], object] = {}

    def update(
        self,
        track_set: TrackSet,
        *,
        leader_id: str,
        follower_id: str | None = None,
    ) -> tuple[PredicateObservation, ...]:
        by_id = {track.scoped_track_id: track for track in track_set.tracks}
        leader = by_id.get(leader_id)
        if leader is None:
            return ()
        candidates = (
            [by_id[follower_id]]
            if follower_id is not None and follower_id in by_id
            else [track for track in track_set.tracks if track.scoped_track_id != leader_id]
        )
        outputs: list[PredicateObservation] = []
        for follower in candidates:
            relation = self._relation(leader, follower)
            key = (leader.scoped_track_id, follower.scoped_track_id)
            history = self._pairs[key]
            if relation is None:
                history.clear()
                continue
            history.append((leader, follower, relation))
            cutoff = track_set.event_time - self._history_horizon
            while history and history[0][0].event_time < cutoff:
                history.popleft()
            result = self._evaluate_history(key, history)
            if result is not None:
                outputs.append(result)
        return tuple(outputs)

    def _relation(self, leader: TrackObservation, follower: TrackObservation) -> float | None:
        skew = abs((leader.event_time - follower.event_time).total_seconds())
        if skew > self.maximum_time_skew_s:
            return None
        if (
            leader.route_id is not None
            and follower.route_id is not None
            and leader.route_progress_m is not None
            and follower.route_progress_m is not None
        ):
            if leader.route_id != follower.route_id:
                return None
            gap = leader.route_progress_m - follower.route_progress_m
        else:
            left = leader.world_point
            right = follower.world_point
            if left is None or right is None:
                raise InvalidProviderInput(
                    "local FOLLOWS requires route progress or common-frame world points"
                )
            if left.coordinate_frame_id != right.coordinate_frame_id:
                raise ArtifactCompatibilityError(
                    "local FOLLOWS track points must share a coordinate frame"
                )
            distance = hypot(left.x - right.x, left.y - right.y)
            # Without a route direction, the track must carry an explicit
            # signed longitudinal gap produced by projection/ordering logic.
            raw_gap = follower.attributes.get("longitudinal_gap_to_leader_m")
            if raw_gap is None:
                return None
            gap = float(raw_gap)
            if abs(abs(gap) - distance) > max(2.0, distance * 0.5):
                return None
        if not self.minimum_gap_m <= gap <= self.maximum_gap_m:
            return None
        return float(gap)

    def _evaluate_history(
        self,
        key: tuple[str, str],
        history: deque[tuple[TrackObservation, TrackObservation, float]],
    ) -> PredicateObservation | None:
        if not history:
            return None
        start = history[0][0].event_time
        end = history[-1][0].event_time
        if (end - start).total_seconds() < self.minimum_duration_s:
            return None
        if self._emitted_until.get(key) == end:
            return None
        self._emitted_until[key] = end
        leader = history[-1][0]
        follower = history[-1][1]
        gaps = [item[2] for item in history]
        interval = EventTimeInterval(start=start, end=end)
        bindings = {
            "leader": leader.scoped_track_id,
            "follower": follower.scoped_track_id,
        }
        return PredicateObservation(
            occurrence_id=occurrence_id("FOLLOWS", bindings, interval, self.provider_id),
            predicate_id="FOLLOWS",
            truth=True,
            confidence=min(
                min(item[0].confidence, item[1].confidence) for item in history
            ),
            event_time_interval=interval,
            bindings=bindings,
            source_ids=tuple(sorted({leader.source_id, follower.source_id})),
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            supporting_artifact_types=("track_summary.v1", "pair_trajectory.v1"),
            measurements={
                "mean_gap_m": sum(gaps) / len(gaps),
                "maximum_gap_m": max(gaps),
                "sample_count": len(gaps),
                "duration_s": (end - start).total_seconds(),
            },
        )


def summarize_tracks(track_sets: tuple[TrackSet, ...]) -> dict[str, object]:
    """Return a serializable compact continuation summary for local geometry."""

    observations = [track for item in track_sets for track in item.tracks]
    if not observations:
        return {"schema_version": "track_summary.v1", "tracks": []}
    by_id: dict[str, list[TrackObservation]] = defaultdict(list)
    for observation in observations:
        by_id[observation.scoped_track_id].append(observation)
    summaries: list[dict[str, object]] = []
    for track_id, rows in sorted(by_id.items()):
        rows.sort(key=lambda item: item.event_time)
        last = rows[-1]
        summaries.append(
            {
                "scoped_track_id": track_id,
                "source_id": last.source_id,
                "tracker_session_id": last.tracker_session_id,
                "class_name": last.class_name,
                "event_start": rows[0].event_time.isoformat(),
                "event_end": rows[-1].event_time.isoformat(),
                "route_id": last.route_id,
                "route_progress_m": last.route_progress_m,
                "world_point": (
                    last.world_point.model_dump(mode="json")
                    if last.world_point is not None
                    else None
                ),
                "sample_count": len(rows),
            }
        )
    return {
        "schema_version": "track_summary.v1",
        "source_id": observations[-1].source_id,
        "tracks": summaries,
    }
