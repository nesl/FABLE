"""Manage all plausible candidate instances for one CE definition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Mapping

from fable.language.event_parser import Event
from fable.language.predicates import PredicateCatalog, load_predicates
from fable.providers.predicate_result import PredicateMatch

from .ce_instance import CEInstance
from .frontier import (
    ActiveFrontier,
    FrontierItem,
    advance_instance,
    derive_continuation_frontier,
    derive_discovery_frontier,
    seed_instance_from_match,
)


class CEInstanceManager:
    """Maintain discovery plus every currently plausible CE candidate.

    The manager preserves a prefix when a match introduces a new role binding,
    because later objects may form alternative valid CE occurrences from that
    same prefix.  When an accepted match only validates identities that are
    already bound, the advanced branch replaces the old prefix.  This keeps the
    useful hypothesis branching behavior without leaving completed deterministic
    steps active indefinitely.

    A later physical planner may coalesce identical provider requirements across
    the remaining semantic candidates without merging the candidates themselves.
    """

    def __init__(
        self,
        event: Event,
        *,
        predicate_catalog: PredicateCatalog | None = None,
    ) -> None:
        self.event = event
        self.predicate_catalog = (
            predicate_catalog if predicate_catalog is not None else load_predicates()
        )
        self._active: dict[str, CEInstance] = {}
        self._completed: list[CEInstance] = []
        self._expired: list[CEInstance] = []
        self._seen_observations: set[tuple[object, ...]] = set()
        self._seen_seeds: set[tuple[object, ...]] = set()
        self._next_number = 1

    def discovery_frontier(self) -> tuple[FrontierItem, ...]:
        return derive_discovery_frontier(
            self.event, predicate_catalog=self.predicate_catalog
        )

    def continuation_frontiers(
        self,
        now: datetime,
    ) -> dict[str, tuple[FrontierItem, ...]]:
        self.expire(now)
        result: dict[str, tuple[FrontierItem, ...]] = {}
        for instance_id, instance in self._active.items():
            frontier = derive_continuation_frontier(
                self.event,
                instance,
                now,
                predicate_catalog=self.predicate_catalog,
            )
            if frontier:
                result[instance_id] = frontier
        self._sweep_terminal()
        return result

    def current_frontier(self, now: datetime) -> ActiveFrontier:
        """Return persistent discovery plus every active continuation frontier."""

        return ActiveFrontier(
            discovery=self.discovery_frontier(),
            continuation=self.continuation_frontiers(now),
        )

    def handle_match(self, match: PredicateMatch) -> tuple[CEInstance, ...]:
        """Advance compatible candidates and/or create new candidates.

        A single observation can advance several existing instances and also
        seed a new instance.  Exact repeated semantic observations are ignored
        so message retries do not duplicate candidates.
        """

        observation_key = _observation_key(match)
        if observation_key in self._seen_observations:
            return ()
        self._seen_observations.add(observation_key)

        self.expire(match.event_time)
        produced: list[CEInstance] = []

        # 1. Advance every compatible existing candidate.  Preserve the parent
        # only when the match introduces a new role binding; that is the case
        # where later objects may form alternative valid hypotheses from the
        # same prefix.  Once all identities for the matched predicate were
        # already bound, the accepted branch replaces the old prefix.  This
        # prevents a completed deterministic step (for example exits(V1)) from
        # leaving an immortal duplicate continuation behind.
        for instance_id, instance in tuple(self._active.items()):
            branches = advance_instance(
                self.event,
                instance,
                match,
                predicate_catalog=self.predicate_catalog,
            )
            if branches and not any(
                set(branch.bindings) - set(instance.bindings)
                for branch in branches
            ):
                self._active.pop(instance_id, None)
            for branch in branches:
                branch.instance_id = self._new_instance_id()
                self._store(branch)
                produced.append(branch)

        # 2. Independently check whether the same observation can start a new
        # CE occurrence from the persistent discovery frontier.
        for item in self.discovery_frontier():
            candidate = seed_instance_from_match(
                self.event,
                item,
                match,
                predicate_catalog=self.predicate_catalog,
            )
            if candidate is None:
                continue
            seed_key = _seed_key(self.event.name, item, candidate)
            if seed_key in self._seen_seeds:
                continue
            self._seen_seeds.add(seed_key)
            candidate.instance_id = self._new_instance_id()
            self._store(candidate)
            produced.append(candidate)

        self._sweep_terminal()
        return tuple(produced)

    def expire(self, now: datetime) -> None:
        """Apply semantic expiry rules and remove expired candidates."""

        for instance in tuple(self._active.values()):
            derive_continuation_frontier(
                self.event,
                instance,
                now,
                predicate_catalog=self.predicate_catalog,
            )
        self._sweep_terminal()

    def active_instances(self) -> tuple[CEInstance, ...]:
        return tuple(self._active.values())

    def completed_instances(self) -> tuple[CEInstance, ...]:
        return tuple(self._completed)

    def expired_instances(self) -> tuple[CEInstance, ...]:
        return tuple(self._expired)

    def recanonicalize_identities(self, canonicalize: Callable[[str], str]) -> None:
        """Rewrite stored role bindings after an identity/ReID merge.

        Identity resolution is intentionally separate from CE-instance
        deduplication.  This method only makes bindings comparable across
        sensors; callers may invoke :meth:`deduplicate_instances` separately if
        they have sufficient evidence that two candidates are the same CE
        occurrence.
        """

        for instance in (*self._active.values(), *self._completed, *self._expired):
            instance.bindings = {
                role: canonicalize(object_id)
                for role, object_id in instance.bindings.items()
            }

    def deduplicate_instances(self) -> tuple[str, ...]:
        """Conservatively remove semantically identical active candidates.

        Two candidates are considered duplicates only when they have the same
        seed occurrence (predicate/path, time, and source), the same canonical
        bindings, and the same semantic progress.  Candidates with the same
        physical participants but different seed times or sources are retained.
        """

        seen: dict[tuple[object, ...], str] = {}
        removed: list[str] = []
        for instance_id, instance in tuple(self._active.items()):
            key = _instance_equivalence_key(instance)
            if key not in seen:
                seen[key] = instance_id
                continue
            del self._active[instance_id]
            removed.append(instance_id)
        return tuple(removed)

    def _new_instance_id(self) -> str:
        value = f"{self.event.name}:{self._next_number}"
        self._next_number += 1
        return value

    def _store(self, instance: CEInstance) -> None:
        if instance.completed:
            self._completed.append(instance)
        elif instance.failed:
            self._expired.append(instance)
        else:
            self._active[instance.instance_id] = instance

    def _sweep_terminal(self) -> None:
        for instance_id, instance in tuple(self._active.items()):
            if instance.completed:
                self._completed.append(instance)
                del self._active[instance_id]
            elif instance.failed:
                self._expired.append(instance)
                del self._active[instance_id]


def _instance_equivalence_key(instance: CEInstance) -> tuple[object, ...]:
    return (
        instance.event_name,
        instance.matched_predicate,
        instance.matched_path,
        instance.matched_at,
        instance.matched_source,
        tuple(sorted(instance.bindings.items())),
        tuple(sorted(instance.satisfied_at.items())),
        tuple(sorted(instance.sustain_started_at.items())),
        tuple(sorted(instance.sustain_last_match_at.items())),
    )


def _observation_key(match: PredicateMatch) -> tuple[object, ...]:
    """Deduplicate semantic message retries without relying on an opaque match ID."""

    return (
        match.predicate,
        match.event_time,
        tuple(match.source_ids),
        tuple(sorted(match.arguments.items())),
    )


def _seed_key(
    event_name: str,
    item: FrontierItem,
    candidate: CEInstance,
) -> tuple[object, ...]:
    """Human-readable seed identity: what matched, when, where, and bindings."""

    return (
        event_name,
        item.path,
        candidate.matched_at,
        candidate.matched_source,
        tuple(sorted(candidate.bindings.items())),
    )
