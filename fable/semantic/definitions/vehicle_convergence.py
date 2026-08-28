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


def vehicle_convergence_graph(
    *,
    departure_policy: str = "identity_bound",
) -> SemanticGraph:
    if departure_policy not in {"identity_bound", "scene_departures"}:
        raise ValueError(
            "departure_policy must be 'identity_bound' or 'scene_departures'"
        )
    builder = AuthoredGraphBuilder(
        namespace="fable.examples.vehicle_convergence",
        name="Vehicle convergence",
        description=(
            "A seed vehicle passes, two bound vehicles converge, then both "
            "vehicles leave the convergence area asynchronously."
        ),
    )
    builder.role("seed_vehicle", "vehicle")
    builder.role("vehicle_a", "vehicle")
    builder.role("vehicle_b", "vehicle")
    if departure_policy == "scene_departures":
        builder.role(
            "departing_vehicle_a",
            "vehicle",
            distinct_from=("departing_vehicle_b",),
        )
        builder.role(
            "departing_vehicle_b",
            "vehicle",
            distinct_from=("departing_vehicle_a",),
        )
    seed = builder.primitive(
        "seed_passes",
        name="Seed vehicle enters the monitored route",
        predicate_id="PASSES",
        roles=(
            PredicateRoleSpec("vehicle", "seed_vehicle", "vehicle"),
            PredicateRoleSpec("reference", "convergence_gate", "location"),
        ),
        checkpoint=False,
    )
    converges = builder.primitive(
        "vehicles_converge",
        name="Two vehicles converge",
        predicate_id="DISTANCE_LT",
        roles=(
            PredicateRoleSpec("left", "vehicle_a", "vehicle"),
            PredicateRoleSpec("right", "vehicle_b", "vehicle"),
        ),
        parameters={"maximum_distance_m": 10.0},
        result_kind=ResultKind.INTERVAL_MATCH,
        checkpoint=True,
        annotations={
            # The seed identity is tracker/camera scoped. Convergence must be
            # evaluated first where that seed was introduced, rather than
            # spending the bounded placement budget on unrelated cameras.
            "source_affinity_roles": ["seed_vehicle"],
            # Image-space PASSES is only emitted after a traversal completes.
            # The vehicles may already have converged while that seed track
            # was crossing the view, so admit bounded retained interval
            # evidence instead of incorrectly requiring MQTT arrival order.
            "activation_lookback_ms": 60_000,
            "minimum_delay_tolerance_ms": 60_000,
        },
    )
    exit_role_a = (
        "departing_vehicle_a"
        if departure_policy == "scene_departures"
        else "vehicle_a"
    )
    exit_role_b = (
        "departing_vehicle_b"
        if departure_policy == "scene_departures"
        else "vehicle_b"
    )
    exit_a = builder.primitive(
        "vehicle_a_exits",
        name="First bound vehicle leaves convergence zone",
        predicate_id="EXITS",
        roles=(PredicateRoleSpec("vehicle", exit_role_a, "vehicle"),),
        checkpoint=False,
        annotations={
            "activation_lookback_ms": 60_000,
            "minimum_delay_tolerance_ms": 60_000,
            "source_affinity_roles": ["vehicle_a"],
        },
    )
    exit_b = builder.primitive(
        "vehicle_b_exits",
        name="Second bound vehicle leaves convergence zone",
        predicate_id="EXITS",
        roles=(PredicateRoleSpec("vehicle", exit_role_b, "vehicle"),),
        checkpoint=False,
        annotations={
            "activation_lookback_ms": 60_000,
            "minimum_delay_tolerance_ms": 60_000,
            "source_affinity_roles": ["vehicle_b"],
        },
    )
    all_depart = builder.and_group(
        "all_group_members_exit",
        (exit_a, exit_b),
        name="All originally bound vehicles leave asynchronously",
        checkpoint=True,
        annotations={
            "completion_policy": (
                "two_distinct_scene_departures"
                if departure_policy == "scene_departures"
                else "all_bound_members"
            ),
            "asynchronous": True,
        },
    )
    return builder.root(
        builder.sequence(
            "convergence",
            (seed, converges, all_depart),
            name="Convergence and complete group departure",
            annotations=trial_rearm_annotations(),
        )
    ).compile()


__all__ = ['vehicle_convergence_graph']
