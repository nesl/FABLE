"""Typed backward search over provider declarations.

The catalog remains ordinary YAML records.  This module builds only a temporary
producer index and performs bounded AND/OR backward chaining from a requested
semantic output to raw sensor inputs.  Adding a provider with compatible
``inputs``/``outputs`` therefore makes it discoverable without authoring a chain.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

from fable.providers.provider_capabilities import (
    load_provider_capabilities,
    provider_supports_predicate_call,
    visual_providers_for_class,
)
from fable.runtime.frontier import FrontierItem

RAW_INPUT_TYPES = frozenset({"video_frame", "audio_window", "multichannel_audio"})


@dataclass(frozen=True, slots=True)
class RawInput:
    data_type: str


@dataclass(frozen=True, slots=True)
class ProviderRecipe:
    provider_id: str
    output_type: str
    inputs: tuple["ProviderRecipe | RawInput", ...] = ()

    def provider_ids(self) -> tuple[str, ...]:
        rows: list[str] = []
        for child in self.inputs:
            if isinstance(child, ProviderRecipe):
                rows.extend(child.provider_ids())
        rows.append(self.provider_id)
        return tuple(rows)

    def raw_input_types(self) -> tuple[str, ...]:
        rows: list[str] = []
        for child in self.inputs:
            if isinstance(child, RawInput):
                rows.append(child.data_type)
            else:
                rows.extend(child.raw_input_types())
        return tuple(rows)


class ProviderSearcher:
    def __init__(
        self,
        catalog: Mapping[str, Any] | None = None,
        *,
        max_depth: int = 8,
        max_recipes: int = 64,
    ) -> None:
        self.catalog = catalog if catalog is not None else load_provider_capabilities()
        self.max_depth = max_depth
        self.max_recipes = max_recipes
        self._producers: dict[str, tuple[str, ...]] = {}
        for provider_id, spec in self.catalog["providers"].items():
            if not spec.get("enabled", True):
                continue
            for output_type in spec.get("outputs", ()):
                self._producers.setdefault(output_type, []).append(provider_id)
        self._producers = {
            output: tuple(sorted(provider_ids))
            for output, provider_ids in self._producers.items()
        }

    def recipes_for_frontier_item(self, item: FrontierItem) -> tuple[ProviderRecipe, ...]:
        target = f"predicate_match:{item.predicate}"
        recipes = self._search(target, item=item, depth=0, stack=())
        return tuple(_dedupe(recipes)[: self.max_recipes])

    def recipes_for_output(
        self,
        output_type: str,
        *,
        item: FrontierItem | None = None,
    ) -> tuple[ProviderRecipe, ...]:
        recipes = self._search(output_type, item=item, depth=0, stack=())
        return tuple(_dedupe(recipes)[: self.max_recipes])

    def _search(
        self,
        target: str,
        *,
        item: FrontierItem | None,
        depth: int,
        stack: tuple[str, ...],
    ) -> list[ProviderRecipe | RawInput]:
        if target in RAW_INPUT_TYPES:
            return [RawInput(target)]
        if depth >= self.max_depth or target in stack:
            return []

        output: list[ProviderRecipe] = []
        for provider_id in self._producers.get(target, ()):
            spec = self.catalog["providers"][provider_id]
            if item is not None and not self._provider_compatible(provider_id, spec, item, target):
                continue
            inputs = tuple(spec.get("inputs", ()))
            child_choices: list[list[ProviderRecipe | RawInput]] = []
            failed = False
            for input_type in inputs:
                candidates = self._search(
                    input_type,
                    item=item,
                    depth=depth + 1,
                    stack=stack + (target,),
                )
                if not candidates:
                    failed = True
                    break
                child_choices.append(candidates[: self.max_recipes])
            if failed:
                continue
            combinations = product(*child_choices) if child_choices else [()]
            for children in combinations:
                output.append(ProviderRecipe(provider_id, target, tuple(children)))
                if len(output) >= self.max_recipes * 2:
                    break
        return output

    def _provider_compatible(
        self,
        provider_id: str,
        spec: Mapping[str, Any],
        item: FrontierItem,
        target: str,
    ) -> bool:
        if spec.get("kind") == "predicate_implementation" and target == f"predicate_match:{item.predicate}":
            return provider_supports_predicate_call(
                self.catalog,
                provider_id,
                item.predicate,
                argument_classes=item.classes,
                literal_arguments=item.parameters,
            )
        if spec.get("kind") == "visual_detector":
            # A single detector branch must be able to observe every semantic
            # visual class required by this predicate occurrence.
            supported = set(spec.get("semantic_classes", {}))
            return all(value in supported for value in set(item.classes.values()))
        accepted = spec.get("accepts_classes", "*")
        if accepted != "*" and item.classes:
            allowed = set(accepted)
            return all(value in allowed for value in set(item.classes.values()))
        return True


def recipe_signature(recipe: ProviderRecipe | RawInput) -> tuple:
    if isinstance(recipe, RawInput):
        return ("raw", recipe.data_type)
    return (
        "provider",
        recipe.provider_id,
        recipe.output_type,
        tuple(recipe_signature(child) for child in recipe.inputs),
    )


def _dedupe(rows: Sequence[ProviderRecipe | RawInput]) -> list:
    seen: set[tuple] = set()
    output = []
    for row in rows:
        signature = recipe_signature(row)
        if signature in seen:
            continue
        seen.add(signature)
        output.append(row)
    return output
