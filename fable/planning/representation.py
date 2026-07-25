"""Representation compatibility and partial-order logic.

FABLE does not assign one scalar "information content" score to raw media,
tracks, embeddings, or symbolic results.  Two representations are compared by
the downstream consumers declared able to use them.  Neither dominates when
those consumer sets are incomparable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from fable.common.enums import ProviderPortKind

from .provider_registry import ProviderRegistry


class RepresentationCompatibility:
    """Declared compatible-consumer sets for typed representations."""

    def __init__(
        self,
        provider_registry: ProviderRegistry,
        *,
        consumer_overrides: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self.providers = provider_registry
        self._consumer_sets = self._derive_consumer_sets()
        for data_type, consumers in (consumer_overrides or {}).items():
            self._consumer_sets[data_type] = frozenset(
                {f"type:{data_type}", *tuple(str(item) for item in consumers)}
            )

    def _derive_consumer_sets(self) -> dict[str, frozenset[str]]:
        consumers: dict[str, set[str]] = {
            data_type: {f"type:{data_type}"}
            for data_type in self.providers.data_types
        }
        for provider in self.providers.providers.values():
            for port in provider.ports:
                if port.kind not in (ProviderPortKind.INPUT, ProviderPortKind.STATE_INPUT):
                    continue
                bucket = consumers.setdefault(port.data_type, {f"type:{port.data_type}"})
                bucket.add(f"provider:{provider.provider_id}")
                bucket.add(provider.provider_id)
        for chain in self.providers.chains.values():
            for external in chain.external_inputs:
                bucket = consumers.setdefault(external.data_type, {f"type:{external.data_type}"})
                bucket.add(f"chain:{chain.chain_id}")
                bucket.add(chain.chain_id)
        return {key: frozenset(value) for key, value in consumers.items()}

    def consumers_for(self, data_type: str) -> frozenset[str]:
        return self._consumer_sets.get(data_type, frozenset({f"type:{data_type}"}))

    def combined_consumers(self, data_types: Iterable[str]) -> frozenset[str]:
        combined: set[str] = set()
        for data_type in data_types:
            combined.update(self.consumers_for(data_type))
        return frozenset(combined)

    def supports(self, data_types: Iterable[str], required_consumers: Iterable[str]) -> bool:
        required = set(required_consumers)
        return required.issubset(self.combined_consumers(data_types))

    def dominates(
        self,
        left_data_types: Iterable[str],
        right_data_types: Iterable[str],
    ) -> bool:
        """Return true when left supports every consumer supported by right."""

        return self.combined_consumers(left_data_types).issuperset(
            self.combined_consumers(right_data_types)
        )

    def comparable(
        self,
        left_data_types: Iterable[str],
        right_data_types: Iterable[str],
    ) -> bool:
        left = self.combined_consumers(left_data_types)
        right = self.combined_consumers(right_data_types)
        return left.issuperset(right) or right.issuperset(left)
