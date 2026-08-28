"""Canonical production complex-event definition.

The factory bodies in this module preserve the authored graph structure and
graph hashes from the historical modality-grouped modules. New definitions
should use :mod:`fable.semantic.authoring` where practical.
"""

from __future__ import annotations

from fable.common.enums import ResultKind
from fable.common.schemas import SemanticGraph
from fable.semantic.builder import AuthoredGraphBuilder, PredicateRoleSpec

from .policy import trial_rearm_annotations


def multimodal_robbery_graph(
    *,
    lookback_ms: int = 120_000,
    alarm_confirmation_ms: int = 45_000,
    gunshot_confirmation_ms: int = 15_000,
) -> SemanticGraph:
    """Trigger-directed robbery analysis with retrospective vehicle recovery.

    A sparse alarm or gunshot is the live trigger. FABLE then searches retained
    pre-trigger tracks for evidence that a vehicle was present and requires a
    later vehicle departure. The departure is first observed independently so
    tracker-local IDs cannot suppress valid EXITS evidence; an explicit
    SAME_ENTITY step then associates it with the retrospectively recovered
    vehicle using a robbery-specific permissive confidence threshold.
    """

    builder = AuthoredGraphBuilder(
        namespace="fable.examples.robbery.phase8",
        name="Multimodal robbery alternatives",
    )
    builder.role("person", "person")
    builder.role("location", "zone")
    builder.role("trigger_location", "zone")
    builder.role("vehicle", "vehicle")
    builder.role("departing_vehicle", "vehicle")
    prior_entry = builder.primitive(
        "prior_entry",
        name="Recover vehicle observed before the trigger",
        predicate_id="VEHICLE_PRESENT_BEFORE",
        roles=(
            PredicateRoleSpec("vehicle", "vehicle", "vehicle"),
            PredicateRoleSpec("location", "trigger_location", "zone"),
        ),
        parameters={"lookback_ms": lookback_ms},
        result_kind=ResultKind.INTERVAL_MATCH,
        annotations={
            "execution_mode": "retrospective",
            "lookback_ms": lookback_ms,
            "retrospective_anchor": {
                "kind": "trigger_node_end",
                "trigger_authored_keys": [
                    "gunshot_branch",
                    "alarm_branch",
                    "threat_branch",
                ],
                "clamp_to_hypothesis_window": False,
            },
            "analysis_mode": "trigger_directed_vehicle_presence_recovery",
            "continuation_artifact_types": ["track_summary.v1"],
        },
        checkpoint=True,
    )
    gunshot = builder.primitive(
        "gunshot_branch",
        name="Gunshot branch",
        predicate_id="AUDIO_EVENT",
        roles=(PredicateRoleSpec("location", "trigger_location", "zone"),),
        parameters={
            "label": "gunshot",
            # The lightweight replay classifier emits occasional startup
            # gunshot false positives before camera-derived history is
            # available. Keep the watch open through that warm-up instead of
            # permanently consuming the request with an impossible
            # retrospective interval.
            "minimum_watch_age_ms": gunshot_confirmation_ms,
        },
    )
    alarm = builder.primitive(
        "alarm_branch",
        name="Alarm branch",
        predicate_id="AUDIO_EVENT",
        roles=(PredicateRoleSpec("location", "trigger_location", "zone"),),
        parameters={
            "label": "alarm",
            "minimum_confidence": 0.75,
            "minimum_watch_age_ms": alarm_confirmation_ms,
        },
    )
    threat = builder.primitive(
        "threat_branch",
        name="Threat interaction branch",
        predicate_id="THREAT_EVENT",
        roles=(
            PredicateRoleSpec("person", "person", "person"),
            PredicateRoleSpec("location", "trigger_location", "zone"),
        ),
        result_kind=ResultKind.INTERVAL_MATCH,
    )
    evidence = builder.or_group(
        "robbery_evidence",
        {"gunshot": gunshot, "alarm": alarm, "threat": threat},
        name="Authored robbery evidence alternatives",
        checkpoint=True,
    )
    trigger_context = builder.sequence(
        "trigger_context",
        (evidence, prior_entry),
        name="Trigger followed by retrospective vehicle recovery",
    )
    departure = builder.primitive(
        "departure",
        name="A vehicle exits the monitored view after the trigger",
        predicate_id="EXITS",
        roles=(
            PredicateRoleSpec("vehicle", "departing_vehicle", "vehicle"),
        ),
        result_kind=ResultKind.INSTANT_MATCH,
        annotations={
            "execution_mode": "retrospective",
            # The recovered candidate interval is the cursor. Earlier media
            # was already searched by VEHICLE_PRESENT_BEFORE; replay only the
            # gap from that cursor through activation, then follow live.
            "lookback_ms": 0,
            "retrospective_anchor": {
                "kind": "trigger_node_end",
                "trigger_authored_key": "prior_entry",
                "clamp_to_hypothesis_window": False,
            },
            "catch_up_and_follow": True,
        },
        checkpoint=True,
    )
    same_vehicle = builder.primitive(
        "same_vehicle",
        name="Associate the departing vehicle with the recovered vehicle",
        predicate_id="SAME_ENTITY",
        roles=(
            PredicateRoleSpec("left", "vehicle", "vehicle"),
            PredicateRoleSpec("right", "departing_vehicle", "vehicle"),
        ),
        parameters={"minimum_confidence": 0.40},
        result_kind=ResultKind.STATE_OBSERVATION,
        annotations={
            "identity_escalation": "reid_then_bounded_vlm",
            "analysis_mode": "robbery_vehicle_identity_confirmation",
        },
        checkpoint=True,
    )
    root = builder.sequence(
        "robbery",
        (trigger_context, departure, same_vehicle),
        name=(
            "Triggered retrospective vehicle-presence recovery followed by "
            "the recovered vehicle's departure"
        ),
    )
    return builder.root(root).compile()


def alarm_departure_graph(
    *,
    alarm_confirmation_ms: int = 10_000,
    departure_window_ms: int = 180_000,
) -> SemanticGraph:
    """Data-aligned alarm incident without requiring cross-camera ReID.

    This profile is intentionally limited to the legacy ``Robbery with alarm``
    labels. It asserts an alarm followed by a vehicle leaving the observed
    scene, but does not claim that the departing vehicle was retrospectively
    identified as the entrant.
    """

    builder = AuthoredGraphBuilder(
        namespace="fable.examples.alarm_departure.phase8",
        name="Alarm followed by vehicle departure",
        description=(
            "A confirmed alarm is followed by a vehicle exit. Identity linkage "
            "to a pre-trigger entrant is not required."
        ),
    )
    builder.role("location", "zone")
    builder.role("vehicle", "vehicle")
    alarm = builder.primitive(
        "alarm_branch",
        name="Confirmed alarm",
        predicate_id="AUDIO_EVENT",
        roles=(PredicateRoleSpec("location", "location", "zone"),),
        parameters={
            "label": "alarm",
            "minimum_confidence": 0.5,
            "minimum_watch_age_ms": alarm_confirmation_ms,
        },
        checkpoint=True,
    )
    departure = builder.primitive(
        "ordinary_exit",
        name="A vehicle exits the monitored view",
        predicate_id="EXITS",
        roles=(PredicateRoleSpec("vehicle", "vehicle", "vehicle"),),
        checkpoint=True,
    )
    departure_within = builder.within(
        "departure_within_window",
        departure,
        after=(alarm,),
        maximum_ms=departure_window_ms,
        name="Vehicle departure within the incident window",
        checkpoint=True,
    )
    root = builder.sequence(
        "alarm_departure",
        (alarm, departure_within),
        name="Alarm and subsequent vehicle departure",
    )
    return builder.root(root).compile()


def robbery_with_alarm_graph(**parameters) -> SemanticGraph:
    """Build the registered robbery profile after validating its profile name."""

    profile = parameters.pop("evaluation_profile", "cross_sensor")
    if profile == "alarm_departure":
        alarm_ms = int(parameters.pop("alarm_confirmation_ms", 10_000))
        if parameters:
            raise ValueError(
                f"unsupported alarm-departure parameters: {sorted(parameters)}"
            )
        return alarm_departure_graph(alarm_confirmation_ms=alarm_ms)
    if profile not in {"cross_sensor", "default"}:
        raise ValueError(f"unsupported robbery evaluation_profile: {profile}")
    return multimodal_robbery_graph(**parameters)


__all__ = [
    "alarm_departure_graph",
    "multimodal_robbery_graph",
    "robbery_with_alarm_graph",
]
