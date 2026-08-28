"""Historical Phase-8 examples retained for compatibility tests.

Production request compilation uses :mod:`fable.semantic.definitions`.  The
older drive-up and robbery examples below intentionally preserve their original
demo shapes; package exchange is re-exported from the canonical definitions so
the former incomplete copy cannot recur.
"""

from fable.common.enums import ResultKind
from fable.semantic.builder import AuthoredGraphBuilder, PredicateRoleSpec
from fable.semantic.definitions.package_exchange import package_exchange_graph


def drive_up_shooting_graph(*, lookback_ms: int = 15_000):
    builder = AuthoredGraphBuilder(
        namespace="fable.examples.drive_up_shooting.phase8",
        name="Drive-up shooting with trigger-directed recovery",
    )
    builder.role("location", "zone")
    builder.role("vehicle", "vehicle")
    builder.role("escape_vehicle", "vehicle")
    gunshot = builder.primitive(
        "gunshot", name="Gunshot at target location", predicate_id="AUDIO_EVENT",
        roles=(PredicateRoleSpec("location", "location", "zone"),),
        parameters={"label": "gunshot"}, checkpoint=True,
    )
    historical = builder.primitive(
        "recover_vehicle_history", name="Recover vehicle from pre-trigger interval",
        predicate_id="VEHICLE_PRESENT_BEFORE",
        roles=(PredicateRoleSpec("vehicle", "vehicle", "vehicle"), PredicateRoleSpec("location", "location", "zone")),
        result_kind=ResultKind.INTERVAL_MATCH,
        parameters={"lookback_ms": lookback_ms},
        annotations={"execution_mode": "retrospective", "lookback_ms": lookback_ms},
    )
    live = builder.primitive(
        "live_escape_tracking", name="Track live moving vehicles after trigger",
        predicate_id="MOVING",
        roles=(PredicateRoleSpec("vehicle", "escape_vehicle", "vehicle"),),
        result_kind=ResultKind.STATE_OBSERVATION,
        annotations={"execution_mode": "live"},
    )
    response = builder.and_group(
        "triggered_response", (historical, live),
        name="Historical recovery and live downstream tracking", checkpoint=True,
        annotations={"continuation_artifact_types": ["track_summary.v1"]},
    )
    return builder.root(builder.sequence(
        "drive_up_shooting", (gunshot, response), name="Gunshot-directed execution"
    )).compile()


def multimodal_robbery_graph():
    builder = AuthoredGraphBuilder(
        namespace="fable.examples.robbery.phase8", name="Multimodal robbery alternatives"
    )
    builder.role("person", "person")
    builder.role("location", "zone")
    builder.role("vehicle", "vehicle")
    entry = builder.primitive(
        "entry", name="Suspicious entry", predicate_id="SUSPICIOUS_ENTRY",
        roles=(PredicateRoleSpec("person", "person", "person"), PredicateRoleSpec("location", "location", "zone")),
        checkpoint=True,
    )
    gunshot = builder.primitive(
        "gunshot_branch", name="Gunshot branch", predicate_id="AUDIO_EVENT",
        roles=(PredicateRoleSpec("location", "location", "zone"),), parameters={"label": "gunshot"},
    )
    alarm = builder.primitive(
        "alarm_branch", name="Alarm branch", predicate_id="AUDIO_EVENT",
        roles=(PredicateRoleSpec("location", "location", "zone"),), parameters={"label": "alarm"},
    )
    threat = builder.primitive(
        "threat_branch", name="Threat branch", predicate_id="THREAT_EVENT",
        roles=(PredicateRoleSpec("person", "person", "person"), PredicateRoleSpec("location", "location", "zone")),
        result_kind=ResultKind.INTERVAL_MATCH,
    )
    evidence = builder.or_group(
        "robbery_evidence", {"gunshot": gunshot, "alarm": alarm, "threat": threat},
        name="Authored robbery evidence alternatives", checkpoint=True,
    )
    departure = builder.primitive(
        "departure", name="Vehicle departure", predicate_id="EXITS",
        roles=(PredicateRoleSpec("vehicle", "vehicle", "vehicle"),), checkpoint=True,
    )
    return builder.root(builder.sequence(
        "robbery", (entry, evidence, departure), name="Entry, evidence, and departure"
    )).compile()

__all__ = [
    "drive_up_shooting_graph",
    "multimodal_robbery_graph",
    "package_exchange_graph",
]
