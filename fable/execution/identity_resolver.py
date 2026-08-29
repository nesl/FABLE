"""Canonical object identity between provider results and CE reasoning.

Predicate implementations emit camera-/sensor-local object identities in
:class:`~fable.providers.predicate_result.PredicateMatch`.  Re-identification
providers may later establish that two such identities refer to the same
physical object.  ``IdentityResolver`` maintains those equivalence classes and
rewrites predicate matches *after* predicate evaluation but *before* they are
sent to the CE instance manager.

This layer deliberately resolves object identity only.  It does not decide
whether two CE instances describe the same complex-event occurrence.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Mapping

from fable.providers.predicate_result import PredicateMatch
from fable.providers.provider_capabilities import (
    load_provider_capabilities,
    native_labels_for_visual_class,
    supported_visual_classes,
)


class IdentityResolver:
    """Small union-find registry for local -> canonical object identities.

    The first identity observed in a connected component remains the canonical
    representative.  This keeps identifiers interpretable in traces (for
    example ``camera_a:track_17``) rather than inventing another opaque UUID.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}
        self._first_seen_order: dict[str, int] = {}
        self._classes: dict[str, set[str]] = {}
        self._next_order = 0
        self._last_seen: dict[str, tuple[str, datetime]] = {}

    def register(self, object_id: str, *, object_class: str | None = None) -> str:
        """Register an object identity and return its current canonical form."""

        if not isinstance(object_id, str) or not object_id.strip():
            raise ValueError("object_id must be a non-empty string")
        if object_id not in self._parent:
            self._parent[object_id] = object_id
            self._rank[object_id] = 0
            self._first_seen_order[object_id] = self._next_order
            self._next_order += 1
            self._classes[object_id] = set()
        if object_class is not None:
            if not isinstance(object_class, str) or not object_class.strip():
                raise ValueError("object_class must be None or a non-empty string")
            root = self._find(object_id)
            self._classes.setdefault(root, set()).add(object_class)
        return self.canonical(object_id)

    def canonical(self, object_id: str) -> str:
        """Return the canonical representative for one local/global identity."""

        if object_id not in self._parent:
            self.register(object_id)
        return self._find(object_id)

    def merge(self, left: str, right: str) -> str:
        """Record that two provider identities refer to the same physical object.

        A merge is rejected when previously observed class evidence has no
        compatible semantic class in the installed provider capability catalog.
        """

        self.register(left)
        self.register(right)
        left_root = self._find(left)
        right_root = self._find(right)
        if left_root == right_root:
            return left_root

        left_classes = self._classes.get(left_root, set())
        right_classes = self._classes.get(right_root, set())
        if left_classes and right_classes and not _class_sets_compatible(
            left_classes, right_classes
        ):
            raise ValueError(
                "cannot merge identities with incompatible observed classes: "
                f"{sorted(left_classes)} vs {sorted(right_classes)}"
            )

        # Preserve the earliest provider identity as the human-readable
        # canonical representative.  Rank is used only as a secondary tie-break.
        left_order = self._first_seen_order[left_root]
        right_order = self._first_seen_order[right_root]
        if (right_order, -self._rank[right_root]) < (left_order, -self._rank[left_root]):
            left_root, right_root = right_root, left_root

        self._parent[right_root] = left_root
        if self._rank[left_root] == self._rank[right_root]:
            self._rank[left_root] += 1
        self._classes.setdefault(left_root, set()).update(
            self._classes.pop(right_root, set())
        )
        left_seen = self._last_seen.get(left_root)
        right_seen = self._last_seen.pop(right_root, None)
        if right_seen is not None and (left_seen is None or right_seen[1] > left_seen[1]):
            self._last_seen[left_root] = right_seen
        return left_root

    def apply_associations(self, associations: Mapping[str, str]) -> dict[str, str]:
        """Apply a ReID provider's ``local_id -> matching_id`` associations."""

        resolved: dict[str, str] = {}
        for left, right in associations.items():
            resolved[left] = self.merge(left, right)
        return resolved

    def canonicalize_match(self, match: PredicateMatch) -> PredicateMatch:
        """Rewrite identity-bearing PredicateMatch arguments to canonical IDs.

        ``PredicateMatch.classes`` is the marker for identity-bearing visual
        arguments.  Literal values such as ``audio_event.class`` are therefore
        left untouched.
        """

        arguments = dict(match.arguments)
        source = match.source_ids[0] if len(match.source_ids) == 1 else None
        for arg_name, object_class in match.classes.items():
            raw = arguments.get(arg_name)
            if not isinstance(raw, str) or not raw:
                continue
            self.register(raw, object_class=object_class)
            canonical = self.canonical(raw)
            arguments[arg_name] = canonical
            if source is not None:
                previous = self._last_seen.get(canonical)
                if previous is None or match.event_time >= previous[1]:
                    self._last_seen[canonical] = (source, match.event_time)
        if arguments == dict(match.arguments):
            return match
        return replace(match, arguments=arguments)


    def last_seen(self, object_id: str) -> tuple[str, datetime] | None:
        """Return the most recent source/time known for a canonical object."""

        root = self.canonical(object_id)
        best = self._last_seen.get(root)
        for alias in self.aliases(root):
            value = self._last_seen.get(alias)
            if value is not None and (best is None or value[1] > best[1]):
                best = value
        return best

    def aliases(self, object_id: str) -> tuple[str, ...]:
        """Return every known provider identity in the same equivalence class."""

        root = self.canonical(object_id)
        return tuple(sorted(key for key in self._parent if self._find(key) == root))

    def _find(self, object_id: str) -> str:
        parent = self._parent[object_id]
        if parent != object_id:
            self._parent[object_id] = self._find(parent)
        return self._parent[object_id]


def _class_sets_compatible(left: set[str], right: set[str]) -> bool:
    if left & right:
        return True
    catalog = load_provider_capabilities()
    semantic_classes = supported_visual_classes(catalog)
    for semantic_class in semantic_classes:
        labels = set(native_labels_for_visual_class(catalog, semantic_class))
        labels.add(semantic_class)
        if labels & left and labels & right:
            return True
    return False
