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


def repeated_visit_graph(
    *,
    return_window_ms: int = 300_000,
    minimum_return_gap_ms: int = 30_000,
    visit_count: int = 2,
) -> SemanticGraph:
    if visit_count < 2:
        raise ValueError("visit_count must be at least 2")
    builder = AuthoredGraphBuilder(
        namespace="fable.examples.repeated_visit.phase1",
        name="Repeated vehicle visit",
        description=(
            "A vehicle is observed at the site, departs, and the same identity "
            "is observed there again within each bounded return window."
        ),
    )
    builder.role("vehicle", "vehicle")
    first = builder.primitive(
        "first_visit",
        name="Vehicle is present for first visit",
        predicate_id="INSIDE",
        roles=(PredicateRoleSpec("vehicle", "vehicle", "vehicle"),),
        result_kind=ResultKind.STATE_OBSERVATION,
        checkpoint=True,
    )
    progression = [first]
    for index in range(2, visit_count + 1):
        suffix = "" if index == 2 else f"_{index - 1}"
        departure = builder.primitive(
            f"departure{suffix}",
            name=f"Bound vehicle departs after visit {index - 1}",
            predicate_id="EXITS",
            roles=(PredicateRoleSpec("vehicle", "vehicle", "vehicle"),),
            checkpoint=True,
        )
        returned = builder.primitive(
            f"return_visit{suffix}",
            name=f"Same vehicle enters for visit {index}",
            predicate_id="ENTERS",
            roles=(PredicateRoleSpec("vehicle", "vehicle", "vehicle"),),
            result_kind=ResultKind.INSTANT_MATCH,
            checkpoint=False,
        )
        return_within = builder.within(
            f"return_within_window{suffix}",
            returned,
            after=(departure,),
            minimum_ms=minimum_return_gap_ms,
            maximum_ms=return_window_ms,
            name=f"Visit {index} within stalking window",
            checkpoint=True,
        )
        progression.extend((departure, return_within))
    root = builder.sequence(
        "repeated_visit_sequence",
        tuple(progression),
        name="Repeated visit progression",
        annotations=trial_rearm_annotations(),
    )
    return builder.root(root).compile()


def uncalibrated_repeated_pass_graph(
    *,
    visit_count: int = 2,
    minimum_return_gap_ms: int = 30_000,
    return_window_ms: int = 300_000,
    identity_confirmation: bool = False,
) -> SemanticGraph:
    """Repeated visits using view crossings when no site-zone calibration exists."""

    if visit_count < 2:
        raise ValueError("visit_count must be at least 2")
    builder = AuthoredGraphBuilder(
        namespace="fable.examples.repeated_pass_visit",
        name="Uncalibrated repeated vehicle visit",
        description=(
            "The same canonical vehicle crosses an observed camera view on "
            "multiple separated occasions. Each PASSES observation is finalized "
            "only after its track leaves the view, and the next pass must occur "
            "after a minimum absence gap; no calibrated INSIDE zone is assumed."
        ),
    )
    builder.role("vehicle", "vehicle")
    builder.role("visit_reference", "location")
    first = builder.primitive(
        "first_visit",
        name="Vehicle observed on first visit",
        predicate_id="PASSES",
        roles=(
            PredicateRoleSpec("vehicle", "vehicle", "vehicle"),
            PredicateRoleSpec("reference", "visit_reference", "location"),
        ),
        checkpoint=True,
    )
    progression = [first]
    previous = first
    for index in range(2, visit_count + 1):
        suffix = "" if index == 2 else f"_{index - 1}"
        # An uncalibrated repeated-visit hypothesis is camera-local. Fan-out
        # may seed one hypothesis per camera, but a later observation must
        # validate the reference bound by that hypothesis rather than letting
        # the first qualifying result from another camera hijack it. Vehicle
        # identity may still fragment within the camera and is confirmed by
        # SAME_ENTITY below.
        candidate_reference = "visit_reference"
        candidate_variable = (
            f"visit_vehicle_{index}" if identity_confirmation else "vehicle"
        )
        if identity_confirmation:
            builder.role(candidate_variable, "vehicle")
        returned = builder.primitive(
            f"return_visit{suffix}",
            name=f"Candidate vehicle observed for visit {index}",
            predicate_id="PASSES",
            roles=(
                PredicateRoleSpec("vehicle", candidate_variable, "vehicle"),
                PredicateRoleSpec("reference", candidate_reference, "location"),
            ),
            checkpoint=False,
            annotations={
                "minimum_delay_tolerance_ms": 5_000,
                "requires_prior_track_termination": True,
                "absence_gap_ms": minimum_return_gap_ms,
            },
        )
        return_within = builder.within(
            f"return_within_window{suffix}",
            returned,
            after=(previous,),
            minimum_ms=minimum_return_gap_ms,
            maximum_ms=return_window_ms,
            name=f"Visit {index} within uncalibrated return window",
            checkpoint=True,
        )
        if not identity_confirmation:
            progression.append(return_within)
            previous = return_within
            continue
        same_vehicle = builder.primitive(
            f"same_vehicle{suffix}",
            name=f"Associate visit {index} with the original vehicle",
            predicate_id="SAME_ENTITY",
            roles=(
                PredicateRoleSpec("left", "vehicle", "vehicle"),
                PredicateRoleSpec("right", candidate_variable, "vehicle"),
            ),
            # The calibrated ReID provider has already applied its cosine
            # distance acceptance gate. Keep this semantic floor below the
            # confidence of an accepted near-boundary match; otherwise the
            # graph silently imposes a second, stricter distance threshold.
            parameters={"minimum_confidence": 0.40},
            result_kind=ResultKind.STATE_OBSERVATION,
            annotations={
                "identity_escalation": "reid_then_bounded_vlm",
                "analysis_mode": "repeated_visit_identity_confirmation",
            },
            checkpoint=True,
        )
        visit_confirmation = builder.sequence(
            f"confirmed_visit{suffix}",
            (return_within, same_vehicle),
            name=f"Visit {index} and identity confirmation",
            checkpoint=True,
        )
        progression.append(visit_confirmation)
        previous = visit_confirmation
    root = builder.sequence(
        "repeated_visit_sequence",
        tuple(progression),
        name="Uncalibrated repeated-view progression",
        annotations=trial_rearm_annotations(),
    )
    return builder.root(root).compile()


__all__ = ['repeated_visit_graph', 'uncalibrated_repeated_pass_graph']
