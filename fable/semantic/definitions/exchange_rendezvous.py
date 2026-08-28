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


def exchange_rendezvous_graph(*, interaction: str = "either") -> SemanticGraph:
    """One rendezvous setup with conversation and package-transfer outcomes."""

    if interaction not in {"either", "conversation", "package_exchange"}:
        raise ValueError(
            "interaction must be either, conversation, or package_exchange"
        )
    builder = AuthoredGraphBuilder(
        namespace=f"fable.examples.exchange_rendezvous.{interaction}.phase8",
        name="Package exchange / rendezvous",
        description=(
            "An observed arrival activates conversation and package-transfer "
            "evidence as outcomes of the same rendezvous setup."
        ),
    )
    builder.role("arrival_vehicle", "vehicle")
    builder.role("participant_a", "person")
    builder.role("participant_b", "person")
    builder.role("package", "package")
    builder.role("source_holder", "entity")
    builder.role("destination_holder", "entity")
    arrival = builder.primitive(
        "arrival",
        name="Arrival at rendezvous",
        predicate_id="PASSES",
        roles=(
            PredicateRoleSpec("vehicle", "arrival_vehicle", "vehicle"),
            PredicateRoleSpec("reference", "rendezvous_gate", "location"),
        ),
        checkpoint=False,
    )
    outcomes = {}
    if interaction in {"either", "conversation"}:
        outcomes["conversation"] = builder.primitive(
            "conversation",
            name="Participants converse",
            predicate_id="CONVERSATION",
            roles=(
                PredicateRoleSpec("participant_a", "participant_a", "person"),
                PredicateRoleSpec("participant_b", "participant_b", "person"),
            ),
            parameters={"maximum_distance_m": 5.0, "minimum_duration_s": 3.0},
            result_kind=ResultKind.INTERVAL_MATCH,
            checkpoint=False,
            annotations={"activation_lookback_ms": 60000},
        )
        outcomes["visual_proximity"] = builder.primitive(
            "visual_proximity",
            name="Participants remain visually proximate",
            predicate_id="PERSON_PROXIMITY",
            roles=(
                PredicateRoleSpec("participant_a", "participant_a", "person"),
                PredicateRoleSpec("participant_b", "participant_b", "person"),
            ),
            parameters={
                "maximum_normalized_gap": 2.5,
                "minimum_duration_s": 3.0,
            },
            result_kind=ResultKind.INTERVAL_MATCH,
            checkpoint=False,
            annotations={"activation_lookback_ms": 60000},
        )
    if interaction in {"either", "package_exchange"}:
        outcomes["package_exchange"] = builder.primitive(
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
                # PASSES is finalized when the arriving vehicle leaves the
                # view. Retain interaction evidence that occurred while that
                # pass was still open so the next frontier is not dependent
                # on callback timing.
                "activation_lookback_ms": 60000,
            },
        )
    outcome = (
        next(iter(outcomes.values()))
        if len(outcomes) == 1
        else builder.or_group(
            "rendezvous_outcome",
            outcomes,
            name="Conversation or package exchange",
            checkpoint=True,
        )
    )
    # This graph intentionally remains the co-presence/interaction proxy.
    # The full package-transfer lifecycle, including separation and bound
    # receiver departure, is authored by ``package_exchange_graph``.
    return builder.root(
        builder.sequence(
            "exchange_rendezvous",
            (arrival, outcome),
            name="Arrival followed by rendezvous interaction",
        )
    ).compile()


__all__ = ['exchange_rendezvous_graph']
