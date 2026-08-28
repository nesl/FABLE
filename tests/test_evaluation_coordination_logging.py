from __future__ import annotations

from datetime import timedelta

from evaluation.coordination_logging import CoordinationEpisodeTracker
from evaluation.schemas import (
    BaselineId,
    PlanDecision,
    PredicateObservation,
    ProviderCommand,
    ProviderLifecycleEvent,
)
from fable.common.examples import BASE_TIME


def _common():
    return {
        "run_id": "run",
        "baseline_id": BaselineId.FABLE,
        "trace_id": "trace",
        "request_id": "request",
        "monotonic_timestamp_ns": 1,
    }


def test_tracker_joins_plan_command_ready_and_downstream_evidence() -> None:
    tracker = CoordinationEpisodeTracker(
        campaign_year=2025,
        spatial_evaluation_eligible=True,
        replay_supported_sensor_ids=("orin11", "orin12", "mobile6"),
        upstream_predicate_ids=("PASSES",),
        downstream_predicate_ids=("PERSON_PROXIMITY",),
        topology_confidence="measured",
        route_ambiguity=2,
    )
    upstream = PredicateObservation(
        **_common(),
        sensor_id="orin11",
        event_time=BASE_TIME,
        observation_id="upstream",
        predicate_id="PASSES",
        event_end_time=BASE_TIME,
        bindings={"vehicle": "vehicle-red"},
        metadata={"predicted_sensor_ids": ["orin12"]},
    )
    assert tracker(upstream) == ()
    decision = PlanDecision(
        **_common(),
        event_time=BASE_TIME + timedelta(seconds=1),
        decision_id="decision",
        checkpoint_id="checkpoint",
        planning_scope="frontier",
        selected_node_ids=("orin12", "mobile6"),
    )
    command = ProviderCommand(
        **_common(),
        event_time=BASE_TIME + timedelta(seconds=2),
        command_id="command",
        command="ACTIVATE",
        node_id="orin12",
        emitted_at=BASE_TIME + timedelta(seconds=2),
    )
    ready = ProviderLifecycleEvent(
        **_common(),
        event_time=BASE_TIME + timedelta(seconds=3),
        provider_instance_id="provider",
        lifecycle_event="READY",
        node_id="orin12",
    )
    downstream = PredicateObservation(
        **_common(),
        sensor_id="orin12",
        event_time=BASE_TIME + timedelta(seconds=6),
        observation_id="downstream",
        predicate_id="PERSON_PROXIMITY",
        event_end_time=BASE_TIME + timedelta(seconds=6),
    )

    assert tracker(decision) == ()
    assert tracker(command) == ()
    assert tracker(ready) == ()
    (episode,) = tracker(downstream)
    assert episode.actual_downstream_sensor_id == "orin12"
    assert episode.predicted_sensor_ids == ("orin12",)
    assert episode.activated_sensor_ids == ("mobile6", "orin12")
    assert episode.object_binding == "vehicle-red"
    assert episode.metadata["outcome"] == "DOWNSTREAM_EVIDENCE"
    assert "orin12" in episode.metadata["command_times"]
    assert "orin12" in episode.metadata["provider_ready_times"]


def test_tracker_emits_explicit_missed_episode_at_expiry() -> None:
    tracker = CoordinationEpisodeTracker(
        campaign_year=2026,
        spatial_evaluation_eligible=False,
        replay_supported_sensor_ids=("orin16",),
        upstream_predicate_ids=("ALARM",),
        downstream_predicate_ids=("PERSON_PRESENT",),
    )
    upstream = PredicateObservation(
        **_common(),
        sensor_id="orin16",
        event_time=BASE_TIME,
        observation_id="alarm",
        predicate_id="ALARM",
        event_end_time=BASE_TIME,
    )
    tracker.begin(upstream, predicted_sensor_ids=("orin16",))
    episode = tracker.expire(
        "request",
        observed_at=BASE_TIME + timedelta(seconds=30),
    )
    assert episode is not None
    assert episode.actual_downstream_sensor_id is None
    assert episode.metadata["outcome"] == "NO_DOWNSTREAM_EVIDENCE"
    assert not episode.spatial_evaluation_eligible
