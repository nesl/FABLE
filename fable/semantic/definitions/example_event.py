"""Canonical production complex-event definition.

The factory bodies in this module preserve the authored graph structure and
graph hashes from the historical modality-grouped modules. New definitions
should use :mod:`fable.semantic.authoring` where practical.
"""

from __future__ import annotations

from fable.common.enums import ResultKind
from fable.common.schemas import SemanticGraph
from fable.semantic.builder import AuthoredGraphBuilder, PredicateRoleSpec
from fable.semantic.authoring import ComplexEvent

from .policy import trial_rearm_annotations


def authored_api_example_graph() -> SemanticGraph:
    """Build a small event using the preferred public authoring API.

    The event means: an initially unknown entity passes a reference point and
    the *same bound identity* subsequently exits. ``predicate`` only declares
    semantic requirements; it does not call a detector. At runtime the demand
    compiler maps each active predicate to provider contracts, while provider
    code produces normalized predicate results. ``sequence`` is evaluated in
    event time by :class:`fable.semantic.runtime.SemanticRuntime`.

    This example intentionally uses the generic ``entity`` role. PASSES and
    EXITS narrow that role to ``vehicle`` through their predicate schemas, so
    one authored role can retain the identity across both predicates.
    """

    event = ComplexEvent(
        "authored_api_example",
        namespace="fable.examples.authored_api",
        name="Passing vehicle later exits",
        description="Teaching example for roles, predicates, and sequencing.",
    )
    event.role("destination_holder", "entity")
    event.role("reference", "location")

    # These calls create graph nodes and validate names/types/parameters. They
    # do not execute PASSES or EXITS and do not select a provider.
    passes = event.predicate(
        "PASSES",
        bind={"vehicle": "destination_holder", "reference": "reference"},
        key="passes_reference",
        name="Vehicle passes the reference",
    )
    exits = event.predicate(
        "EXITS",
        bind={"vehicle": "destination_holder"},
        key="vehicle_exits",
        name="The bound vehicle exits",
    )

    # sequence() adds both structural child edges and an event-time precedence
    # edge. build() finalizes stable IDs, validates the DAG, and selects root.
    root = event.sequence(passes, exits, key="root", name="Pass then exit")
    return event.build(root)


__all__ = ["all_constructs_graph", "authored_api_example_graph"]
