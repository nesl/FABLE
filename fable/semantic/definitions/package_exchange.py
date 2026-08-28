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


def package_exchange_graph() -> SemanticGraph:
    """Arrivals and transfer followed by the bound receiver leaving."""

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
    receiver_departure = builder.primitive(
        "receiver_departs",
        name="Bound receiving vehicle leaves the exchange zone",
        predicate_id="EXITS",
        roles=(PredicateRoleSpec("vehicle", "vehicle_b", "vehicle"),),
        checkpoint=True,
        annotations={
            "transfer_role": "receiver_vehicle",
            "destination_alternative": "REACHES(destination_zone)",
        },
    )
    root = builder.sequence(
        "package_exchange",
        (arrivals, transfer, receiver_departure),
        name="Arrivals, transfer, and receiver departure",
        annotations=trial_rearm_annotations(),
    )
    return builder.root(root).compile()


__all__ = ['package_exchange_graph']
