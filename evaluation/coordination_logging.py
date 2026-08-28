"""Construct common coordination episodes from the live record stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter_ns

from evaluation.schemas import (
    CoordinationEpisode,
    PlanDecision,
    PredicateObservation,
    ProviderCommand,
    ProviderLifecycleEvent,
)
from evaluation.schemas.records import EvaluationRecord
from fable.common.ids import deterministic_id


@dataclass
class _OpenEpisode:
    upstream: PredicateObservation
    predicted_sensor_ids: tuple[str, ...]
    deadline: datetime | None
    object_binding: str | None
    activated_sensor_ids: set[str] = field(default_factory=set)
    command_times: dict[str, datetime] = field(default_factory=dict)
    provider_ready_times: dict[str, datetime] = field(default_factory=dict)


class CoordinationEpisodeTracker:
    """Join records for one request without relying on planner-private state."""

    def __init__(
        self,
        *,
        campaign_year: int,
        spatial_evaluation_eligible: bool,
        replay_supported_sensor_ids: tuple[str, ...],
        unavailable_mobile_sensor_ids: tuple[str, ...] = (),
        upstream_predicate_ids: tuple[str, ...],
        downstream_predicate_ids: tuple[str, ...],
        topology_confidence: str = "",
        route_ambiguity: int = 1,
    ) -> None:
        self.campaign_year = campaign_year
        self.spatial_evaluation_eligible = spatial_evaluation_eligible
        self.replay_supported_sensor_ids = replay_supported_sensor_ids
        self.unavailable_mobile_sensor_ids = unavailable_mobile_sensor_ids
        self.upstream_predicate_ids = set(upstream_predicate_ids)
        self.downstream_predicate_ids = set(downstream_predicate_ids)
        self.topology_confidence = topology_confidence
        self.route_ambiguity = route_ambiguity
        self._open: dict[str, _OpenEpisode] = {}

    def begin(
        self,
        upstream: PredicateObservation,
        *,
        predicted_sensor_ids: tuple[str, ...],
        deadline: datetime | None = None,
        object_binding: str | None = None,
    ) -> None:
        if upstream.predicate_id not in self.upstream_predicate_ids:
            return
        self._open[upstream.request_id] = _OpenEpisode(
            upstream=upstream,
            predicted_sensor_ids=predicted_sensor_ids,
            deadline=deadline,
            object_binding=object_binding,
        )

    def __call__(
        self,
        record: EvaluationRecord,
    ) -> tuple[CoordinationEpisode, ...]:
        episode = self._open.get(record.request_id)
        if (
            episode is None
            and isinstance(record, PredicateObservation)
            and record.predicate_id in self.upstream_predicate_ids
        ):
            predicted = record.metadata.get("predicted_sensor_ids", ())
            if not isinstance(predicted, (list, tuple)):
                predicted = ()
            object_binding = record.metadata.get("object_binding")
            if object_binding is None and record.bindings:
                object_binding = next(iter(record.bindings.values()))
            self.begin(
                record,
                predicted_sensor_ids=tuple(str(item) for item in predicted),
                object_binding=(
                    str(object_binding) if object_binding is not None else None
                ),
            )
            return ()
        if episode is None:
            return ()
        if isinstance(record, PlanDecision):
            episode.activated_sensor_ids.update(record.selected_node_ids)
        elif isinstance(record, ProviderCommand):
            episode.activated_sensor_ids.add(record.node_id)
            episode.command_times.setdefault(record.node_id, record.emitted_at)
        elif isinstance(record, ProviderLifecycleEvent):
            if record.lifecycle_event.upper() in {"READY", "ACTIVE"}:
                episode.provider_ready_times.setdefault(
                    record.node_id,
                    record.event_time,
                )
        elif (
            isinstance(record, PredicateObservation)
            and record.predicate_id in self.downstream_predicate_ids
            and record.sensor_id is not None
            and record.sensor_id != episode.upstream.sensor_id
        ):
            del self._open[record.request_id]
            return (self._complete(episode, record),)
        return ()

    def expire(
        self,
        request_id: str,
        *,
        observed_at: datetime,
    ) -> CoordinationEpisode | None:
        episode = self._open.pop(request_id, None)
        if episode is None:
            return None
        return self._complete(episode, None, observed_at=observed_at)

    def _complete(
        self,
        episode: _OpenEpisode,
        downstream: PredicateObservation | None,
        *,
        observed_at: datetime | None = None,
    ) -> CoordinationEpisode:
        upstream = episode.upstream
        downstream_time = (
            downstream.event_time if downstream is not None else None
        )
        activated = tuple(sorted(episode.activated_sensor_ids))
        return CoordinationEpisode(
            run_id=upstream.run_id,
            baseline_id=upstream.baseline_id,
            trace_id=upstream.trace_id,
            request_id=upstream.request_id,
            hypothesis_id=upstream.hypothesis_id,
            sensor_id=upstream.sensor_id,
            provider_id=upstream.provider_id,
            event_time=observed_at or downstream_time or upstream.event_time,
            monotonic_timestamp_ns=perf_counter_ns(),
            episode_id=deterministic_id(
                "coordination_episode",
                {
                    "request_id": upstream.request_id,
                    "upstream_observation_id": upstream.observation_id,
                },
                length=32,
            ),
            campaign_year=self.campaign_year,
            spatial_evaluation_eligible=self.spatial_evaluation_eligible,
            upstream_sensor_id=upstream.sensor_id or "unknown",
            object_binding=episode.object_binding,
            predicted_sensor_ids=episode.predicted_sensor_ids,
            activated_sensor_ids=activated,
            replay_supported_sensor_ids=self.replay_supported_sensor_ids,
            unavailable_mobile_sensor_ids=self.unavailable_mobile_sensor_ids,
            actual_downstream_sensor_id=(
                downstream.sensor_id if downstream is not None else None
            ),
            prediction_time=upstream.event_time,
            downstream_observation_time=downstream_time,
            deadline=episode.deadline,
            topology_confidence=self.topology_confidence,
            route_ambiguity=self.route_ambiguity,
            metadata={
                "upstream_observation_id": upstream.observation_id,
                "downstream_observation_id": (
                    downstream.observation_id if downstream is not None else None
                ),
                "command_times": {
                    key: value.isoformat()
                    for key, value in sorted(episode.command_times.items())
                },
                "provider_ready_times": {
                    key: value.isoformat()
                    for key, value in sorted(
                        episode.provider_ready_times.items()
                    )
                },
                "outcome": (
                    "DOWNSTREAM_EVIDENCE"
                    if downstream is not None
                    else "NO_DOWNSTREAM_EVIDENCE"
                ),
            },
        )
