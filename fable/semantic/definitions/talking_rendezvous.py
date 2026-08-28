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


def talking_rendezvous_graph() -> SemanticGraph:
    builder = AuthoredGraphBuilder(
        namespace="fable.examples.talking_rendezvous",
        name="Talking rendezvous",
        description=(
            "A visual arrival is followed by bounded interaction and the "
            "subsequent departure of a vehicle from the rendezvous view."
        ),
    )
    builder.role("arrival_vehicle", "vehicle")
    builder.role("departing_vehicle", "vehicle")
    builder.role("participant_a", "person")
    builder.role("participant_b", "person")
    seed = builder.primitive(
        "arrival",
        name="Arrival at rendezvous",
        predicate_id="PASSES",
        roles=(
            PredicateRoleSpec("vehicle", "arrival_vehicle", "vehicle"),
            PredicateRoleSpec("reference", "rendezvous_gate", "location"),
        ),
        checkpoint=True,
    )
    conversation = builder.primitive(
        "conversation",
        name="Participants converse",
        predicate_id="CONVERSATION",
        roles=(
            PredicateRoleSpec("participant_a", "participant_a", "person"),
            PredicateRoleSpec("participant_b", "participant_b", "person"),
        ),
        parameters={"maximum_distance_m": 5.0, "minimum_duration_s": 3.0},
        result_kind=ResultKind.INTERVAL_MATCH,
        annotations={
            "activation_lookback_ms": 60_000,
            "minimum_delay_tolerance_ms": 60_000,
        },
        checkpoint=False,
    )
    proximity = builder.primitive(
        "visual_proximity",
        name="Participants remain visually proximate",
        predicate_id="PERSON_PROXIMITY",
        roles=(
            PredicateRoleSpec("participant_a", "participant_a", "person"),
            PredicateRoleSpec("participant_b", "participant_b", "person"),
        ),
        parameters={"maximum_normalized_gap": 2.5, "minimum_duration_s": 3.0},
        result_kind=ResultKind.INTERVAL_MATCH,
        annotations={
            "activation_lookback_ms": 60_000,
            "minimum_delay_tolerance_ms": 60_000,
        },
        checkpoint=False,
    )
    interaction = builder.or_group(
        "rendezvous_interaction",
        {"conversation": conversation, "visual_proximity": proximity},
        name="Audible conversation or sustained visual proximity",
        checkpoint=True,
    )
    departure = builder.primitive(
        "arrival_vehicle_exits",
        name="A vehicle leaves the rendezvous zone",
        predicate_id="EXITS",
        roles=(PredicateRoleSpec("vehicle", "departing_vehicle", "vehicle"),),
        annotations={
            "activation_lookback_ms": 60_000,
            "minimum_delay_tolerance_ms": 60_000,
            # Departure is scene-local to the accepted interaction.  The
            # vehicle itself remains intentionally unbound, but cost ties must
            # not place its lifecycle evaluator on an unrelated camera.
            "source_affinity_roles": ["participant_a", "participant_b"],
            # A vehicle may disappear while interaction evidence is still
            # accumulating. Recover that bounded interval, then keep following
            # live frames so either a past or future exit can satisfy EXITS.
            "catch_up_and_follow": True,
        },
        checkpoint=True,
    )
    return builder.root(
        builder.sequence(
            "rendezvous",
            (seed, interaction, departure),
            name="Arrival, interaction, and vehicle departure",
            annotations=trial_rearm_annotations(),
        )
    ).compile()


__all__ = ['talking_rendezvous_graph']
