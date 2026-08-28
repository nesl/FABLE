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


def sequential_vehicle_pass_graph(
    *,
    name: str,
    namespace_suffix: str,
    maximum_gap_ms: int = 60_000,
) -> SemanticGraph:
    """Two distinct vehicles cross a view sequentially without co-visibility."""

    builder = AuthoredGraphBuilder(
        namespace=f"fable.examples.{namespace_suffix}",
        name=name,
        description=(
            "A leader crosses a monitored view and a distinct follower crosses "
            "within a bounded interval; simultaneous FOLLOWS evidence is optional."
        ),
    )
    builder.role("leader", "vehicle")
    builder.role("follower", "vehicle", distinct_from=("leader",))
    builder.role("reference", "location")
    leader = builder.primitive(
        "leader_passes",
        name="Leader crosses the monitored view",
        predicate_id="PASSES",
        roles=(
            PredicateRoleSpec("vehicle", "leader", "vehicle"),
            PredicateRoleSpec("reference", "reference", "location"),
        ),
        checkpoint=True,
    )
    follower = builder.primitive(
        "follower_passes",
        name="Distinct follower crosses the monitored view",
        predicate_id="PASSES",
        roles=(
            PredicateRoleSpec("vehicle", "follower", "vehicle"),
            PredicateRoleSpec("reference", "reference", "location"),
        ),
    )
    follower_within = builder.within(
        "follower_within_window",
        follower,
        after=(leader,),
        # Adjacent tracker intervals commonly share an exact boundary when
        # one vehicle leaves as the next arrives. Requiring an arbitrary
        # positive gap made identical replay data nondeterministic across
        # inference cadence. Distinct vehicle bindings prevent the seed from
        # satisfying both positions; the upper bound carries the convoy/chase
        # temporal semantics.
        minimum_ms=0,
        maximum_ms=maximum_gap_ms,
        name="Follower crosses within the sequential convoy window",
        checkpoint=True,
    )
    root = builder.sequence(
        "sequential_vehicle_passes",
        (leader, follower_within),
        name="Sequential vehicle crossings",
        annotations={
            # A convoy occurrence is anchored by the leader crossing.  The
            # runtime may fork several valid follower hypotheses for that one
            # crossing; those are alternate members of the same terminal CE,
            # not separate convoy events.  Retain the leader occurrence (not
            # merely its canonical identity) so a later convoy led by the same
            # vehicle can still be emitted independently.
            "terminal_event_identity_roles": ["leader"],
        },
    )
    return builder.root(root).compile()


def route_convoy_graph(*, maximum_gap_ms: int = 60_000) -> SemanticGraph:
    """Build the registered default pass-follow-clear convoy definition."""

    return sequential_vehicle_pass_graph(
        name="Pass-follow-clear convoy",
        namespace_suffix="pass_follow_clear_convoy",
        maximum_gap_ms=maximum_gap_ms,
    )


__all__ = ["route_convoy_graph", "sequential_vehicle_pass_graph"]
