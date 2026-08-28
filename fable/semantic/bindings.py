"""Canonical entity binding and alias management for the semantic runtime."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

from fable.common.schemas import EntityBinding, Hypothesis

from .compiled import CompiledSemanticGraph


class BindingError(ValueError):
    """Raised when a predicate result is incompatible with hypothesis bindings."""


@dataclass(frozen=True)
class CanonicalObservation:
    entity_type: str
    canonical_entity_id: str
    source_id: str
    local_entity_id: str


class CanonicalBindingManager:
    """Resolves provider-local identifiers to stable event-level identities.

    The first implementation uses an explicit alias table.  A real identity
    association provider can populate this table later without changing the
    hypothesis interface.
    """

    def __init__(self) -> None:
        self._aliases: dict[tuple[str, str, str], str] = {}
        self._reverse: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)

    def register_alias(
        self,
        *,
        entity_type: str,
        source_id: str,
        local_entity_id: str,
        canonical_entity_id: str,
    ) -> None:
        key = (entity_type, source_id, local_entity_id)
        existing = self._aliases.get(key)
        if existing is not None and existing != canonical_entity_id:
            raise BindingError(
                f"alias {key!r} is already assigned to canonical entity {existing!r}"
            )
        self._aliases[key] = canonical_entity_id
        self._reverse[(entity_type, canonical_entity_id)].add((source_id, local_entity_id))

    def resolve(
        self,
        *,
        entity_type: str,
        source_id: str,
        observed_entity_id: str,
    ) -> CanonicalObservation:
        canonical = self._aliases.get(
            (entity_type, source_id, observed_entity_id),
            observed_entity_id,
        )
        self._aliases.setdefault((entity_type, source_id, observed_entity_id), canonical)
        self._reverse[(entity_type, canonical)].add((source_id, observed_entity_id))
        return CanonicalObservation(
            entity_type=entity_type,
            canonical_entity_id=canonical,
            source_id=source_id,
            local_entity_id=observed_entity_id,
        )

    def aliases_for(self, entity_type: str, canonical_entity_id: str) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for source_id, local_id in sorted(self._reverse.get((entity_type, canonical_entity_id), set())):
            grouped[source_id].append(local_id)
        return {source_id: tuple(values) for source_id, values in grouped.items()}

    def make_binding(
        self,
        *,
        role_name: str,
        entity_type: str,
        source_id: str,
        observed_entity_id: str,
        occurrence_id: str | None,
    ) -> EntityBinding:
        observation = self.resolve(
            entity_type=entity_type,
            source_id=source_id,
            observed_entity_id=observed_entity_id,
        )
        return EntityBinding(
            role_name=role_name,
            entity_type=entity_type,
            canonical_entity_id=observation.canonical_entity_id,
            local_entity_ids=self.aliases_for(entity_type, observation.canonical_entity_id),
            established_by_occurrence_id=occurrence_id,
        )

    def enrich_binding(
        self,
        binding: EntityBinding,
        *,
        source_id: str,
        observed_entity_id: str,
    ) -> EntityBinding:
        observation = self.resolve(
            entity_type=binding.entity_type,
            source_id=source_id,
            observed_entity_id=observed_entity_id,
        )
        if observation.canonical_entity_id != binding.canonical_entity_id:
            raise BindingError(
                f"observed entity {observed_entity_id!r} resolves to "
                f"{observation.canonical_entity_id!r}, expected {binding.canonical_entity_id!r}"
            )
        return binding.model_copy(
            update={
                "local_entity_ids": self.aliases_for(
                    binding.entity_type,
                    binding.canonical_entity_id,
                )
            }
        )

    def canonicalize_delta(
        self,
        *,
        graph: CompiledSemanticGraph,
        hypothesis: Hypothesis | None,
        node_id: str,
        introduced: Mapping[str, str],
        validated: Mapping[str, str],
        source_id: str,
        occurrence_id: str,
    ) -> tuple[dict[str, EntityBinding], dict[str, EntityBinding]]:
        """Return canonical introduced and validated bindings.

        BindingDelta keys refer to semantic graph variables rather than the
        provider-facing role labels in a predicate contract.
        """

        variable_types = graph.predicate_variables(node_id)
        predicate = graph.nodes_by_id[node_id].predicate
        local_to_variable = {
            role.role_name: role.variable for role in (predicate.roles if predicate else ())
        }

        def normalize(values: Mapping[str, str]) -> dict[str, str]:
            return {
                (key if key in variable_types else local_to_variable.get(key, key)): value
                for key, value in values.items()
            }

        introduced = normalize(introduced)
        validated = normalize(validated)
        unknown = (set(introduced) | set(validated)) - set(variable_types)
        if unknown:
            raise BindingError(f"binding delta references unknown predicate variables: {sorted(unknown)}")

        introduced_bindings: dict[str, EntityBinding] = {}
        validated_bindings: dict[str, EntityBinding] = {}

        for role_name, observed_id in introduced.items():
            if role_name not in {role.role_name for role in graph.graph.roles}:
                raise BindingError(f"predicate attempted to bind undeclared graph role {role_name!r}")
            binding = self.make_binding(
                role_name=role_name,
                entity_type=variable_types[role_name],
                source_id=source_id,
                observed_entity_id=observed_id,
                occurrence_id=occurrence_id,
            )
            introduced_bindings[role_name] = binding

        for role_name, observed_id in validated.items():
            if hypothesis is None or role_name not in hypothesis.role_bindings:
                raise BindingError(f"cannot validate unbound role {role_name!r}")
            existing = hypothesis.role_bindings[role_name]
            if existing.entity_type != variable_types[role_name]:
                raise BindingError(
                    f"role {role_name!r} expects {existing.entity_type!r}, "
                    f"predicate returned {variable_types[role_name]!r}"
                )
            validated_binding = self.enrich_binding(
                existing,
                source_id=source_id,
                observed_entity_id=observed_id,
            )
            validated_bindings[role_name] = validated_binding

        prospective = dict(hypothesis.role_bindings if hypothesis else {})
        prospective.update(introduced_bindings)
        prospective.update(validated_bindings)
        self.validate_distinctness(graph, prospective)
        return introduced_bindings, validated_bindings

    @staticmethod
    def validate_distinctness(
        graph: CompiledSemanticGraph,
        bindings: Mapping[str, EntityBinding],
    ) -> None:
        for role in graph.graph.roles:
            left = bindings.get(role.role_name)
            if left is None:
                continue
            for other_name in role.distinct_from:
                right = bindings.get(other_name)
                if right is None:
                    continue
                if (
                    left.entity_type == right.entity_type
                    and left.canonical_entity_id == right.canonical_entity_id
                ):
                    raise BindingError(
                        f"roles {role.role_name!r} and {other_name!r} must bind distinct entities"
                    )
