"""Bounded matching of provider observations to later semantic frontiers.

Whole-event baselines may execute a predicate before its graph node is enabled.
The provider result therefore cannot be submitted directly to
``SemanticRuntime.apply``: its demand/frontier envelope belongs to the planning
projection, not to the authoritative active hypothesis.  This module retains
the immutable observation and creates a fresh result envelope only after a
compatible grounded demand exists.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections.abc import Mapping
from collections import Counter

from fable.common.ids import uuid7
from fable.common.schemas import BindingDelta, PredicateDemand, PredicateResult
from fable.common.time import ensure_utc


_SYMBOLIC_CAMERA_REFERENCES = frozenset(
    {
        "chase_gate",
        "convoy_gate",
        "convergence_gate",
        "rendezvous_gate",
        "route_gate",
        "visit_reference",
    }
)

# Provider timestamps are commonly quantized at a one-second boundary while
# the demand frontier is opened by the immediately preceding observation.
# Preserve that boundary observation, but do not allow materially older
# occurrences to advance a later repeated-predicate stage.
_EVENT_TIME_BOUNDARY_TOLERANCE = timedelta(seconds=1)


def _compatible_bound_role(expected: str, actual: str) -> bool:
    """Resolve authored camera-role labels at the execution boundary.

    Gate names describe the role a selected camera plays; providers correctly
    report the concrete camera FOV. They are not physical entity identities and
    must not be compared as opaque IDs.
    """

    if expected == actual:
        return True
    if expected in _SYMBOLIC_CAMERA_REFERENCES and actual.startswith("camera_fov:"):
        return True
    return False


@dataclass
class BufferedObservation:
    result: PredicateResult
    buffered_at: datetime
    delivered_targets: set[tuple[str, str]]


class EarlyObservationBuffer:
    """Request-local, bounded, destructive-read observation buffer."""

    def __init__(
        self,
        *,
        max_observations: int = 4096,
        retention_ms: int = 300_000,
    ) -> None:
        if max_observations < 1:
            raise ValueError("max_observations must be positive")
        if retention_ms < 1:
            raise ValueError("retention_ms must be positive")
        self.max_observations = max_observations
        self.retention = timedelta(milliseconds=retention_ms)
        self._items: deque[BufferedObservation] = deque()

    def add(self, result: PredicateResult, *, now: datetime) -> None:
        now = ensure_utc(now)
        self.expire(now=now)
        self._items.append(
            BufferedObservation(
                result=result,
                buffered_at=now,
                delivered_targets=set(),
            )
        )
        while len(self._items) > self.max_observations:
            self._items.popleft()

    def expire(self, *, now: datetime) -> int:
        cutoff = ensure_utc(now) - self.retention
        removed = 0
        while self._items and self._items[0].buffered_at < cutoff:
            self._items.popleft()
            removed += 1
        return removed

    def pop_matches(
        self,
        demand: PredicateDemand,
        *,
        now: datetime,
        source_aliases: Mapping[str, tuple[str, ...]] | None = None,
    ) -> tuple[PredicateResult, ...]:
        """Return compatible observations with an authoritative demand envelope."""

        self.expire(now=now)
        matched: list[PredicateResult] = []
        retained: deque[BufferedObservation] = deque()
        for item in self._items:
            if not _matches(
                item.result,
                demand,
                source_aliases=source_aliases or {},
            ):
                retained.append(item)
                continue
            matched.append(
                item.result.model_copy(
                    update={
                        "result_id": uuid7(),
                        "demand_id": demand.demand_id,
                        "request_id": demand.request_id,
                        "graph_hash": demand.graph_hash,
                        "hypothesis_id": demand.hypothesis_id,
                        "expected_hypothesis_version": demand.hypothesis_version,
                        "frontier_id": demand.frontier_id,
                        "checkpoint_id": demand.checkpoint_id,
                        "graph_node_id": demand.graph_node_id,
                        "semantic_predicate": demand.semantic_predicate,
                        "binding_delta": _project_binding_delta(item.result, demand),
                    }
                )
            )
        self._items = retained
        return tuple(
            sorted(
                matched,
                key=lambda result: (
                    result.event_time_interval.end,
                    result.occurrence_id,
                    str(result.result_id),
                ),
            )
        )

    def match_for_demand(
        self,
        demand: PredicateDemand,
        *,
        now: datetime,
        source_aliases: Mapping[str, tuple[str, ...]] | None = None,
    ) -> tuple[PredicateResult, ...]:
        """Project observations once per semantic target without stealing them.

        One physical occurrence can legitimately advance several rolling
        hypotheses.  Destructive reads made the result depend on hypothesis
        iteration and MQTT arrival order.  Retain the immutable observation
        for the request horizon while recording delivery to each
        ``(hypothesis, graph-node)`` target independently.
        """

        self.expire(now=now)
        target = (str(demand.hypothesis_id), demand.graph_node_id)
        candidates: list[BufferedObservation] = []
        for item in self._items:
            if target in item.delivered_targets or not _matches(
                item.result,
                demand,
                source_aliases=source_aliases or {},
            ):
                continue
            candidates.append(item)
        if not candidates:
            return ()
        ordered = sorted(
            candidates,
            key=lambda item: (
                item.result.event_time_interval.end,
                item.result.occurrence_id,
                str(item.result.result_id),
            ),
        )
        # Materialize the complete deterministic batch against this exact
        # frontier/version. The semantic runtime keeps a fork context for that
        # envelope, allowing later candidates to create sibling hypotheses
        # even after the first candidate advances/forks the parent.
        for item in ordered:
            item.delivered_targets.add(target)
        return tuple(
            item.result.model_copy(
                update={
                    "result_id": uuid7(),
                    "demand_id": demand.demand_id,
                    "request_id": demand.request_id,
                    "graph_hash": demand.graph_hash,
                    "hypothesis_id": demand.hypothesis_id,
                    "expected_hypothesis_version": demand.hypothesis_version,
                    "frontier_id": demand.frontier_id,
                    "checkpoint_id": demand.checkpoint_id,
                    "graph_node_id": demand.graph_node_id,
                    "semantic_predicate": demand.semantic_predicate,
                    "binding_delta": _project_binding_delta(item.result, demand),
                }
            )
            for item in ordered
        )

    def __len__(self) -> int:
        return len(self._items)

    def rejection_counts(
        self,
        demand: PredicateDemand,
        *,
        source_aliases: Mapping[str, tuple[str, ...]] | None = None,
    ) -> dict[str, int]:
        """Summarize why retained observations cannot satisfy ``demand``.

        Values are deliberately categorical: diagnostics must not expose full
        provider payloads or identity bindings in progress messages.
        """

        counts: Counter[str] = Counter()
        for item in self._items:
            reason = mismatch_reason(
                item.result,
                demand,
                source_aliases=source_aliases or {},
            )
            counts[reason or "MATCH"] += 1
        return dict(sorted(counts.items()))


def _matches(
    result: PredicateResult,
    demand: PredicateDemand,
    *,
    source_aliases: Mapping[str, tuple[str, ...]],
) -> bool:
    return mismatch_reason(result, demand, source_aliases=source_aliases) is None


def mismatch_reason(
    result: PredicateResult,
    demand: PredicateDemand,
    *,
    source_aliases: Mapping[str, tuple[str, ...]],
) -> str | None:
    """Return the first stable compatibility failure category, if any."""

    if result.request_id != demand.request_id:
        return "REQUEST_ID"
    if result.graph_hash != demand.graph_hash:
        return "GRAPH_HASH"
    if not _same_provider_predicate_contract(
        result.semantic_predicate,
        demand.semantic_predicate,
    ):
        return "SEMANTIC_PREDICATE"
    # Authored graph-node IDs identify a stage, not a physical observation
    # type. Repeated predicates (visit 1/2/3, retrospective/live exits, or
    # convergence checkpoints) deliberately have distinct node IDs while
    # sharing one typed predicate contract. The active demand envelope below
    # rematerializes the result for the authoritative node; event-time and
    # grounded-binding checks prevent an earlier occurrence from satisfying a
    # later stage incorrectly.
    observed_sources = set(result.provenance.source_ids)
    canonical_sources = set(observed_sources)
    for source_id in observed_sources:
        canonical_sources.update(source_aliases.get(source_id, ()))
    if demand.eligible_source_ids and not (
        canonical_sources & set(demand.eligible_source_ids)
    ):
        return "ELIGIBLE_SOURCE"
    observed = result.event_time_interval
    required = demand.event_time_interval
    if (
        observed.end < required.start - _EVENT_TIME_BOUNDARY_TOLERANCE
        or observed.start > required.end + _EVENT_TIME_BOUNDARY_TOLERANCE
    ):
        return "EVENT_TIME_INTERVAL"

    # Demand bindings are keyed by predicate role name while provider deltas
    # are keyed by authored graph variable. Never rewrite identity evidence:
    # a buffered observation is usable only when it already agrees.
    supplied = {
        **result.binding_delta.introduced,
        **result.binding_delta.validated,
    }
    variables = {
        role.role_name: role.variable for role in result.semantic_predicate.roles
    }
    for role_name, expected in demand.bound_roles.items():
        if expected.startswith("__structural_unbound__:"):
            # The observation grounds this role. Rejecting it here makes the
            # first observation of every late-bound frontier impossible.
            continue
        actual = supplied.get(variables.get(role_name, role_name))
        if actual is not None and not _compatible_bound_role(expected, actual):
            return "BOUND_ROLE"
    return None


def _same_provider_predicate_contract(left, right) -> bool:
    """Compare provider-facing semantics while ignoring graph variable names."""

    return (
        left.predicate_id == right.predicate_id
        and left.parameters == right.parameters
        and left.result_kind == right.result_kind
        and tuple((role.role_name, role.entity_type) for role in left.roles)
        == tuple((role.role_name, role.entity_type) for role in right.roles)
    )


def _project_binding_delta(
    result: PredicateResult,
    demand: PredicateDemand,
) -> BindingDelta:
    """Translate provider-role values into the active stage's graph variables."""

    observed_by_role = {
        role.role_name: role.variable for role in result.semantic_predicate.roles
    }
    target_by_role = {
        role.role_name: role.variable for role in demand.semantic_predicate.roles
    }

    def translate(values: Mapping[str, str]) -> dict[str, str]:
        by_role = {
            role_name: values[variable]
            for role_name, variable in observed_by_role.items()
            if variable in values
        }
        return {
            target_by_role[role_name]: value
            for role_name, value in by_role.items()
            if role_name in target_by_role
        }

    return BindingDelta(
        introduced=translate(result.binding_delta.introduced),
        validated=translate(result.binding_delta.validated),
        rejected_roles=result.binding_delta.rejected_roles,
    )
