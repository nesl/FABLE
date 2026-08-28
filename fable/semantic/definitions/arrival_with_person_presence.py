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


def arrival_with_person_presence_graph() -> SemanticGraph:
    """Rendezvous proxy for recordings without usable conversation audio."""

    builder = AuthoredGraphBuilder(
        namespace="fable.examples.arrival_person_presence",
        name="Arrival with visual person presence",
        description=(
            "A vehicle already present in the rendezvous view is followed by "
            "stable person co-presence. This avoids treating handheld-camera "
            "motion as evidence that a stationary vehicle passed a gate. It is "
            "a co-presence proxy, not evidence that speech occurred."
        ),
    )
    builder.role("arrival_vehicle", "vehicle")
    builder.role("participant", "person")
    arrival = builder.primitive(
        "arrival",
        name="Vehicle present at rendezvous view",
        predicate_id="INSIDE",
        roles=(
            PredicateRoleSpec("vehicle", "arrival_vehicle", "vehicle"),
            PredicateRoleSpec("zone", "rendezvous_gate", "location"),
        ),
        result_kind=ResultKind.STATE_OBSERVATION,
        checkpoint=True,
    )
    person = builder.primitive(
        "person_present",
        name="Person remains visible after arrival",
        predicate_id="PERSON_PRESENT",
        roles=(PredicateRoleSpec("participant", "participant", "person"),),
        result_kind=ResultKind.STATE_OBSERVATION,
        checkpoint=True,
    )
    root = builder.sequence(
        "arrival_and_presence",
        (arrival, person),
        name="Arrival followed by visual co-presence",
    )
    return builder.root(root).compile()


__all__ = ['arrival_with_person_presence_graph']
