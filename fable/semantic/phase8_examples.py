"""Authored multimodal graphs used by Phase-8 integration tests and demos."""

from __future__ import annotations

from fable.common.enums import ResultKind
from fable.common.schemas import SemanticGraph

from .builder import AuthoredGraphBuilder, PredicateRoleSpec


def drive_up_shooting_graph(*, lookback_ms: int = 15_000) -> SemanticGraph:
    """Gunshot trigger followed by concurrent historical and live work.

    The graph is semantic rather than physical. ``VEHICLE_PRESENT_BEFORE`` is a
    retrospective obligation over a pre-trigger interval, while ``MOVING`` is a
    live downstream observation.  Provider/placement selection remains in the
    physical planner.
    """

    builder = AuthoredGraphBuilder(
        namespace="fable.examples.drive_up_shooting.phase8",
        name="Drive-up shooting with trigger-directed recovery",
    )
    builder.role("location", "zone")
    builder.role("vehicle", "vehicle")
    builder.role("escape_vehicle", "vehicle")
    gunshot = builder.primitive(
        "gunshot",
        name="Gunshot at target location",
        predicate_id="AUDIO_EVENT",
        roles=(PredicateRoleSpec("location", "location", "zone"),),
        parameters={"label": "gunshot"},
        checkpoint=True,
    )
    historical = builder.primitive(
        "recover_vehicle_history",
        name="Recover vehicle from pre-trigger interval",
        predicate_id="VEHICLE_PRESENT_BEFORE",
        roles=(
            PredicateRoleSpec("vehicle", "vehicle", "vehicle"),
            PredicateRoleSpec("location", "location", "zone"),
        ),
        result_kind=ResultKind.INTERVAL_MATCH,
        parameters={"lookback_ms": lookback_ms},
        annotations={
            "execution_mode": "retrospective",
            "lookback_ms": lookback_ms,
        },
    )
    live_escape = builder.primitive(
        "live_escape_tracking",
        name="Track live moving vehicles after trigger",
        predicate_id="MOVING",
        roles=(PredicateRoleSpec("vehicle", "escape_vehicle", "vehicle"),),
        result_kind=ResultKind.STATE_OBSERVATION,
        annotations={"execution_mode": "live"},
    )
    response = builder.and_group(
        "triggered_response",
        (historical, live_escape),
        name="Historical recovery and live downstream tracking",
        checkpoint=True,
        annotations={"continuation_artifact_types": ["track_summary.v1"]},
    )
    root = builder.sequence(
        "drive_up_shooting",
        (gunshot, response),
        name="Gunshot-directed execution",
    )
    return builder.root(root).compile()


def multimodal_robbery_graph() -> SemanticGraph:
    """Authored robbery alternatives that share entry and departure logic."""

    builder = AuthoredGraphBuilder(
        namespace="fable.examples.robbery.phase8",
        name="Multimodal robbery alternatives",
    )
    builder.role("person", "person")
    builder.role("location", "zone")
    builder.role("vehicle", "vehicle")
    entry = builder.primitive(
        "entry",
        name="Suspicious entry",
        predicate_id="SUSPICIOUS_ENTRY",
        roles=(
            PredicateRoleSpec("person", "person", "person"),
            PredicateRoleSpec("location", "location", "zone"),
        ),
        checkpoint=True,
    )
    gunshot = builder.primitive(
        "gunshot_branch",
        name="Gunshot branch",
        predicate_id="AUDIO_EVENT",
        roles=(PredicateRoleSpec("location", "location", "zone"),),
        parameters={"label": "gunshot"},
    )
    alarm = builder.primitive(
        "alarm_branch",
        name="Alarm branch",
        predicate_id="AUDIO_EVENT",
        roles=(PredicateRoleSpec("location", "location", "zone"),),
        parameters={"label": "alarm"},
    )
    threat = builder.primitive(
        "threat_branch",
        name="Threat interaction branch",
        predicate_id="THREAT_EVENT",
        roles=(
            PredicateRoleSpec("person", "person", "person"),
            PredicateRoleSpec("location", "location", "zone"),
        ),
        result_kind=ResultKind.INTERVAL_MATCH,
    )
    evidence = builder.or_group(
        "robbery_evidence",
        {"gunshot": gunshot, "alarm": alarm, "threat": threat},
        name="Authored robbery evidence alternatives",
        checkpoint=True,
    )
    departure = builder.primitive(
        "departure",
        name="Departure or escape",
        predicate_id="DEPARTURE_OR_ESCAPE",
        roles=(
            PredicateRoleSpec("person", "person", "person"),
            PredicateRoleSpec("vehicle", "vehicle", "vehicle"),
        ),
        checkpoint=True,
    )
    root = builder.sequence(
        "robbery",
        (entry, evidence, departure),
        name="Shared entry, alternatives, and departure",
    )
    return builder.root(root).compile()


def package_exchange_graph() -> SemanticGraph:
    """Arrivals activate high-resolution transfer analysis and compact custody."""

    builder = AuthoredGraphBuilder(
        namespace="fable.examples.package_exchange.phase8",
        name="Package exchange with custody continuation",
    )
    builder.role("vehicle_a", "vehicle")
    builder.role("vehicle_b", "vehicle")
    builder.role("package", "package")
    builder.role("source_holder", "entity")
    builder.role("destination_holder", "entity")
    arrive_a = builder.primitive(
        "arrive_a",
        name="First vehicle arrives",
        predicate_id="ENTERS",
        roles=(PredicateRoleSpec("vehicle", "vehicle_a", "vehicle"),),
    )
    arrive_b = builder.primitive(
        "arrive_b",
        name="Second vehicle arrives",
        predicate_id="ENTERS",
        roles=(PredicateRoleSpec("vehicle", "vehicle_b", "vehicle"),),
    )
    arrivals = builder.and_group(
        "arrivals",
        (arrive_a, arrive_b),
        name="Concurrent vehicle arrivals",
        checkpoint=True,
    )
    transfer = builder.primitive(
        "transfer",
        name="Package custody changes",
        predicate_id="TRANSFER",
        roles=(
            PredicateRoleSpec("object", "package", "package"),
            PredicateRoleSpec("source", "source_holder", "entity"),
            PredicateRoleSpec("destination", "destination_holder", "entity"),
        ),
        result_kind=ResultKind.INTERVAL_MATCH,
        checkpoint=True,
        annotations={
            "analysis_mode": "high_resolution_interaction",
            "continuation_artifact_types": ["custody_state.v1"],
        },
    )
    root = builder.sequence(
        "package_exchange",
        (arrivals, transfer),
        name="Arrivals followed by transfer",
    )
    return builder.root(root).compile()
