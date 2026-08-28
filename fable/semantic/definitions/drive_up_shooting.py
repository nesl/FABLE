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


def drive_up_shooting_graph(
    *,
    lookback_ms: int = 15_000,
    require_boarding: bool = True,
) -> SemanticGraph:
    """Identity-preserving drive-up robbery sequence.

    A positive event requires a person to leave a vehicle, a subsequent
    gunshot, that same person to board that same vehicle, and the bound vehicle
    to move. ``lookback_ms`` remains accepted for request compatibility but the
    strengthened graph no longer substitutes generic historical vehicle
    presence for the person/vehicle transitions.
    """

    builder = AuthoredGraphBuilder(
        namespace="fable.examples.drive_up_shooting.phase8",
        name="Drive-up shooting with trigger-directed recovery",
    )
    builder.role("person", "person")
    builder.role("vehicle", "vehicle")
    disembarks = builder.primitive(
        "disembarks",
        name="Person disembarks from stopped vehicle",
        predicate_id="DISEMBARKS",
        roles=(
            PredicateRoleSpec("person", "person", "person"),
            PredicateRoleSpec("vehicle", "vehicle", "vehicle"),
        ),
        checkpoint=False,
    )
    gunshot = builder.primitive(
        "gunshot",
        name="Gunshot at target location",
        predicate_id="AUDIO_EVENT",
        roles=(PredicateRoleSpec("location", "scene", "zone"),),
        parameters={"label": "gunshot"},
        annotations={"precedence_tolerance_ms": 1_500},
        checkpoint=True,
    )
    boards = builder.primitive(
        "boards",
        name="Same person returns to and boards same vehicle",
        predicate_id="BOARDS",
        roles=(
            PredicateRoleSpec("person", "person", "person"),
            PredicateRoleSpec("vehicle", "vehicle", "vehicle"),
        ),
        annotations={"precedence_tolerance_ms": 1_500},
        checkpoint=True,
    )
    departure = builder.primitive(
        "departure",
        name="Bound getaway vehicle exits the incident zone",
        predicate_id="EXITS",
        roles=(PredicateRoleSpec("vehicle", "vehicle", "vehicle"),),
        checkpoint=True,
    )
    progression = (
        (disembarks, gunshot, boards, departure)
        if require_boarding
        else (disembarks, gunshot, departure)
    )
    root = builder.sequence(
        "drive_up_shooting",
        progression,
        name=(
            "Disembark, gunshot, same-person return, and bound-vehicle exit"
            if require_boarding
            else "Disembark, gunshot, and bound-vehicle exit"
        ),
        annotations={
            "boarding_required": require_boarding,
            "variant": (
                "observable_boarding"
                if require_boarding
                else "boarding_unobservable"
            ),
            **trial_rearm_annotations(),
        },
    )
    return builder.root(root).compile()


__all__ = ['drive_up_shooting_graph']
