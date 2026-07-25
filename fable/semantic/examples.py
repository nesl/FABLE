"""Authored Phase-1 example graphs built with the semantic DSL."""

from __future__ import annotations

from fable.common.enums import ResultKind
from fable.common.schemas import SemanticGraph

from .builder import AuthoredGraphBuilder, PredicateRoleSpec


def repeated_visit_graph(*, return_window_ms: int = 300_000) -> SemanticGraph:
    builder = AuthoredGraphBuilder(
        namespace="fable.examples.repeated_visit.phase1",
        name="Repeated vehicle visit",
        description="A vehicle visits, departs, and the same identity returns within a bounded window.",
    )
    builder.role("vehicle", "vehicle")
    first = builder.primitive(
        "first_visit",
        name="Vehicle enters for first visit",
        predicate_id="ENTERS",
        roles=(PredicateRoleSpec("vehicle", "vehicle", "vehicle"),),
        checkpoint=True,
    )
    departure = builder.primitive(
        "departure",
        name="Bound vehicle departs",
        predicate_id="EXITS",
        roles=(PredicateRoleSpec("vehicle", "vehicle", "vehicle"),),
        checkpoint=True,
    )
    returned = builder.primitive(
        "return_visit",
        name="Same vehicle returns",
        predicate_id="ENTERS",
        roles=(PredicateRoleSpec("vehicle", "vehicle", "vehicle"),),
        checkpoint=False,
    )
    return_within = builder.within(
        "return_within_window",
        returned,
        after=(first,),
        maximum_ms=return_window_ms,
        name="Return within stalking window",
        checkpoint=True,
    )
    root = builder.sequence(
        "repeated_visit_sequence",
        (first, departure, return_within),
        name="Repeated visit progression",
    )
    return builder.root(root).compile()


def all_constructs_graph() -> SemanticGraph:
    """Small graph used to validate every minimum authored construct."""

    builder = AuthoredGraphBuilder(
        namespace="fable.tests.all_constructs",
        name="All Phase-1 constructs",
    )
    builder.role("vehicle", "vehicle")
    p1 = builder.primitive(
        "p1",
        name="P1",
        predicate_id="P1",
        roles=(PredicateRoleSpec("vehicle", "vehicle", "vehicle"),),
        checkpoint=True,
    )
    p2 = builder.primitive(
        "p2",
        name="P2",
        predicate_id="P2",
        roles=(PredicateRoleSpec("vehicle", "vehicle", "vehicle"),),
        result_kind=ResultKind.STATE_OBSERVATION,
    )
    p3 = builder.primitive("p3", name="P3", predicate_id="P3")
    p4 = builder.primitive("p4", name="P4", predicate_id="P4")
    p5 = builder.primitive("p5", name="P5", predicate_id="P5")
    duration = builder.duration("duration", p2, minimum_ms=5_000, name="P2 held")
    absent = builder.absent(
        "absent",
        p3,
        window_ms=10_000,
        required_source_ids=("camera_a",),
        name="P3 absent",
    )
    choices = builder.or_group(
        "choices",
        {"p4": p4, "p5": p5},
        name="P4 or P5",
    )
    cardinality = builder.k_of_n(
        "cardinality",
        (duration, absent, choices),
        k=2,
        name="Two supporting conditions",
    )
    bounded = builder.within(
        "bounded",
        cardinality,
        after=(p1,),
        maximum_ms=60_000,
        name="Support within one minute",
    )
    root = builder.sequence("root", (p1, bounded), name="All constructs sequence")
    return builder.root(root).compile()
