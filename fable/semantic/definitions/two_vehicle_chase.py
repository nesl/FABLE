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


def two_vehicle_chase_graph() -> SemanticGraph:
    builder = AuthoredGraphBuilder(
        namespace="fable.examples.two_vehicle_chase",
        name="Two-vehicle chase",
        description="A leader passes a route gate and a distinct vehicle follows.",
    )
    builder.role("leader", "vehicle")
    builder.role("follower", "vehicle", distinct_from=("leader",))
    seed = builder.primitive(
        "leader_passes",
        name="Leader passes route gate",
        predicate_id="PASSES",
        roles=(
            PredicateRoleSpec("vehicle", "leader", "vehicle"),
            PredicateRoleSpec("reference", "chase_gate", "location"),
        ),
        checkpoint=True,
    )
    follows = builder.primitive(
        "chase_follows",
        name="Follower pursues leader",
        predicate_id="FOLLOWS",
        roles=(
            PredicateRoleSpec("leader", "leader", "vehicle"),
            PredicateRoleSpec("follower", "follower", "vehicle"),
        ),
        parameters={"max_gap_m": 30.0, "min_duration_ms": 3000},
        result_kind=ResultKind.INTERVAL_MATCH,
        checkpoint=True,
    )
    return builder.root(
        builder.sequence("chase", (seed, follows), name="Chase sequence")
    ).compile()


__all__ = ['two_vehicle_chase_graph']
