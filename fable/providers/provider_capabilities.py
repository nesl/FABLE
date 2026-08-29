"""Load and query FABLE's provider capability and label catalog.

This module is intentionally small.  The capability catalog answers only the
questions needed before runtime planning:

* Which semantic visual classes can the configured perception providers see?
* Which native model labels implement each semantic class?
* Which provider implementations can emit each public CE predicate?
* Which semantic literal values (for example audio classes) do those predicate
  implementations support?

It does **not** choose a provider, placement, model instance, or execution plan.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


ProviderCapabilityCatalog = dict[str, Any]
_SEMANTIC_CLASS_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ALLOWED_KINDS = frozenset({"visual_detector", "predicate_implementation", "intermediate"})


def _default_catalog_path() -> Path:
    return Path(__file__).with_name("provider_capabilities.yaml")


def canonical_semantic_class(native_label: str) -> str:
    """Convert a model-native label such as ``traffic light`` to ``traffic_light``."""

    text = native_label.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


@lru_cache(maxsize=8)
def _load_cached(path_string: str) -> ProviderCapabilityCatalog:
    path = Path(path_string)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return _validate_catalog(raw, source=str(path))


def load_provider_capabilities(path: str | Path | None = None) -> ProviderCapabilityCatalog:
    """Load a validated capability catalog.

    A deep copy is returned so tests/deployments may enable or disable providers
    without mutating the process-global cached source catalog.
    """

    catalog_path = Path(path) if path is not None else _default_catalog_path()
    return deepcopy(_load_cached(str(catalog_path.resolve())))


def _validate_catalog(raw: object, *, source: str) -> ProviderCapabilityCatalog:
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: provider capability catalog must be a mapping")
    if raw.get("version") != 1:
        raise ValueError(f"{source}: expected provider capability catalog version 1")

    label_spaces = raw.get("label_spaces", {})
    providers = raw.get("providers")
    if not isinstance(label_spaces, dict):
        raise ValueError(f"{source}.label_spaces: expected a mapping")
    if not isinstance(providers, dict) or not providers:
        raise ValueError(f"{source}.providers: expected a non-empty mapping")

    normalized_spaces: dict[str, dict[str, Any]] = {}
    for space_name, space_spec in label_spaces.items():
        if not isinstance(space_name, str) or not space_name.isidentifier():
            raise ValueError(f"{source}.label_spaces: invalid label-space name {space_name!r}")
        if not isinstance(space_spec, dict):
            raise ValueError(f"{source}.label_spaces.{space_name}: expected an object")
        unknown = set(space_spec) - {"description", "labels"}
        if unknown:
            raise ValueError(
                f"{source}.label_spaces.{space_name}: unknown fields {sorted(unknown)}"
            )
        labels = space_spec.get("labels")
        if not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x for x in labels):
            raise ValueError(f"{source}.label_spaces.{space_name}.labels: expected non-empty string list")
        if len(labels) != len(set(labels)):
            raise ValueError(f"{source}.label_spaces.{space_name}.labels: duplicate native labels")
        normalized_spaces[space_name] = {
            "description": str(space_spec.get("description", "")),
            "labels": tuple(labels),
        }

    normalized_providers: dict[str, dict[str, Any]] = {}
    for provider_id, provider_spec in providers.items():
        provider_path = f"{source}.providers.{provider_id}"
        if not isinstance(provider_id, str) or not provider_id.isidentifier():
            raise ValueError(f"{source}.providers: invalid provider ID {provider_id!r}")
        if not isinstance(provider_spec, dict):
            raise ValueError(f"{provider_path}: expected an object")
        kind = provider_spec.get("kind")
        if kind not in _ALLOWED_KINDS:
            raise ValueError(f"{provider_path}.kind: expected one of {sorted(_ALLOWED_KINDS)}")
        enabled = provider_spec.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"{provider_path}.enabled: expected boolean")

        if kind == "visual_detector":
            normalized_providers[provider_id] = _validate_visual_detector(
                provider_spec,
                provider_id=provider_id,
                provider_path=provider_path,
                label_spaces=normalized_spaces,
            )
        elif kind == "predicate_implementation":
            normalized_providers[provider_id] = _validate_predicate_provider(
                provider_spec,
                provider_id=provider_id,
                provider_path=provider_path,
            )
        else:
            unknown = set(provider_spec) - {"kind", "enabled", "inputs", "outputs", "accepts_classes"}
            if unknown:
                raise ValueError(f"{provider_path}: unknown fields {sorted(unknown)}")
            inputs, outputs = _validate_io(provider_spec, provider_path)
            accepts_raw = provider_spec.get("accepts_classes", "*")
            if accepts_raw == "*":
                accepts_classes: str | tuple[str, ...] = "*"
            elif isinstance(accepts_raw, list) and accepts_raw and all(
                isinstance(value, str) and _SEMANTIC_CLASS_RE.fullmatch(value) for value in accepts_raw
            ):
                accepts_classes = tuple(accepts_raw)
            else:
                raise ValueError(f"{provider_path}.accepts_classes: expected '*' or semantic-class list")
            normalized_providers[provider_id] = {
                "kind": kind,
                "enabled": enabled,
                "inputs": inputs,
                "outputs": outputs,
                "accepts_classes": accepts_classes,
            }

    return {
        "version": 1,
        "label_spaces": normalized_spaces,
        "providers": normalized_providers,
    }


def _validate_io(spec: Mapping[str, Any], provider_path: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    def normalize(field: str) -> tuple[str, ...]:
        raw = spec.get(field, [])
        if not isinstance(raw, list) or not all(isinstance(value, str) and value.strip() for value in raw):
            raise ValueError(f"{provider_path}.{field}: expected string list")
        return tuple(raw)
    return normalize("inputs"), normalize("outputs")


def _validate_visual_detector(
    spec: Mapping[str, Any],
    *,
    provider_id: str,
    provider_path: str,
    label_spaces: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    unknown = set(spec) - {
        "kind",
        "enabled",
        "native_label_space",
        "native_labels",
        "expose_native_labels_as_semantic_classes",
        "semantic_classes",
        "inputs",
        "outputs",
    }
    if unknown:
        raise ValueError(f"{provider_path}: unknown fields {sorted(unknown)}")

    label_space = spec.get("native_label_space")
    if label_space is not None and label_space not in label_spaces:
        raise ValueError(f"{provider_path}.native_label_space: unknown label space {label_space!r}")

    available_native: tuple[str, ...]
    if "native_labels" in spec:
        raw_labels = spec["native_labels"]
        if not isinstance(raw_labels, list) or not raw_labels or not all(isinstance(x, str) and x for x in raw_labels):
            raise ValueError(f"{provider_path}.native_labels: expected non-empty string list")
        available_native = tuple(raw_labels)
        if label_space is not None:
            declared = set(label_spaces[label_space]["labels"])
            unknown_labels = set(available_native) - declared
            if unknown_labels:
                raise ValueError(
                    f"{provider_path}.native_labels: labels not in {label_space!r}: {sorted(unknown_labels)}"
                )
    elif label_space is not None:
        available_native = tuple(label_spaces[label_space]["labels"])
    else:
        raise ValueError(
            f"{provider_path}: visual detector needs native_labels or native_label_space"
        )

    expose_native = spec.get("expose_native_labels_as_semantic_classes", False)
    if not isinstance(expose_native, bool):
        raise ValueError(
            f"{provider_path}.expose_native_labels_as_semantic_classes: expected boolean"
        )

    semantic_classes: dict[str, tuple[str, ...]] = {}
    if expose_native:
        for native_label in available_native:
            semantic_name = canonical_semantic_class(native_label)
            if semantic_name:
                semantic_classes[semantic_name] = (native_label,)

    raw_semantic = spec.get("semantic_classes", {})
    if not isinstance(raw_semantic, dict):
        raise ValueError(f"{provider_path}.semantic_classes: expected a mapping")
    for semantic_class, native_labels in raw_semantic.items():
        if not isinstance(semantic_class, str) or not _SEMANTIC_CLASS_RE.fullmatch(semantic_class):
            raise ValueError(
                f"{provider_path}.semantic_classes: invalid semantic class {semantic_class!r}"
            )
        if not isinstance(native_labels, list) or not native_labels or not all(isinstance(x, str) for x in native_labels):
            raise ValueError(
                f"{provider_path}.semantic_classes.{semantic_class}: expected non-empty native-label list"
            )
        unknown_labels = set(native_labels) - set(available_native)
        if unknown_labels:
            raise ValueError(
                f"{provider_path}.semantic_classes.{semantic_class}: unavailable native labels {sorted(unknown_labels)}"
            )
        semantic_classes[semantic_class] = tuple(native_labels)

    inputs, outputs = _validate_io(spec, provider_path)
    return {
        "kind": "visual_detector",
        "enabled": bool(spec.get("enabled", True)),
        "native_label_space": label_space,
        "native_labels": available_native,
        "semantic_classes": semantic_classes,
        "inputs": inputs,
        "outputs": outputs,
    }


def _validate_predicate_provider(
    spec: Mapping[str, Any],
    *,
    provider_id: str,
    provider_path: str,
) -> dict[str, Any]:
    unknown = set(spec) - {"kind", "enabled", "predicates", "inputs", "outputs"}
    if unknown:
        raise ValueError(f"{provider_path}: unknown fields {sorted(unknown)}")
    predicates = spec.get("predicates")
    if not isinstance(predicates, dict) or not predicates:
        raise ValueError(f"{provider_path}.predicates: expected a non-empty mapping")

    normalized_predicates: dict[str, dict[str, Any]] = {}
    for predicate_name, predicate_spec in predicates.items():
        predicate_path = f"{provider_path}.predicates.{predicate_name}"
        if not isinstance(predicate_name, str) or not predicate_name.isidentifier() or predicate_name.lower() != predicate_name:
            raise ValueError(f"{predicate_path}: invalid predicate name")
        if predicate_spec is None:
            predicate_spec = {}
        if not isinstance(predicate_spec, dict):
            raise ValueError(f"{predicate_path}: expected an object")
        unknown_pred = set(predicate_spec) - {"visual_arguments", "semantic_literals"}
        if unknown_pred:
            raise ValueError(f"{predicate_path}: unknown fields {sorted(unknown_pred)}")

        visual_arguments: dict[str, str | tuple[str, ...]] = {}
        raw_visual = predicate_spec.get("visual_arguments", {})
        if not isinstance(raw_visual, dict):
            raise ValueError(f"{predicate_path}.visual_arguments: expected a mapping")
        for arg_name, allowed in raw_visual.items():
            arg_path = f"{predicate_path}.visual_arguments.{arg_name}"
            if allowed == "*":
                visual_arguments[arg_name] = "*"
            elif isinstance(allowed, list) and allowed and all(
                isinstance(x, str) and _SEMANTIC_CLASS_RE.fullmatch(x) for x in allowed
            ):
                visual_arguments[arg_name] = tuple(allowed)
            else:
                raise ValueError(f"{arg_path}: expected '*' or non-empty semantic-class list")

        semantic_literals: dict[str, dict[str, tuple[str, ...]]] = {}
        raw_literals = predicate_spec.get("semantic_literals", {})
        if not isinstance(raw_literals, dict):
            raise ValueError(f"{predicate_path}.semantic_literals: expected a mapping")
        for arg_name, value_map in raw_literals.items():
            literal_path = f"{predicate_path}.semantic_literals.{arg_name}"
            if not isinstance(value_map, dict) or not value_map:
                raise ValueError(f"{literal_path}: expected a non-empty value mapping")
            normalized_values: dict[str, tuple[str, ...]] = {}
            for semantic_value, value_spec in value_map.items():
                if not isinstance(semantic_value, str) or not _SEMANTIC_CLASS_RE.fullmatch(semantic_value):
                    raise ValueError(f"{literal_path}: invalid semantic value {semantic_value!r}")
                if not isinstance(value_spec, dict) or set(value_spec) != {"native_labels"}:
                    raise ValueError(
                        f"{literal_path}.{semantic_value}: expected only 'native_labels'"
                    )
                labels = value_spec["native_labels"]
                if not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x for x in labels):
                    raise ValueError(
                        f"{literal_path}.{semantic_value}.native_labels: expected non-empty string list"
                    )
                normalized_values[semantic_value] = tuple(labels)
            semantic_literals[arg_name] = normalized_values

        normalized_predicates[predicate_name] = {
            "visual_arguments": visual_arguments,
            "semantic_literals": semantic_literals,
        }

    inputs, outputs = _validate_io(spec, provider_path)
    return {
        "kind": "predicate_implementation",
        "enabled": bool(spec.get("enabled", True)),
        "predicates": normalized_predicates,
        "inputs": inputs,
        "outputs": outputs,
    }


def visual_providers_for_class(
    catalog: Mapping[str, Any], semantic_class: str
) -> tuple[str, ...]:
    """Return enabled detector providers that can observe ``semantic_class``."""

    matches = []
    for provider_id, spec in catalog["providers"].items():
        if spec["kind"] != "visual_detector" or not spec["enabled"]:
            continue
        if semantic_class in spec["semantic_classes"]:
            matches.append(provider_id)
    return tuple(sorted(matches))


def native_labels_for_visual_class(
    catalog: Mapping[str, Any], semantic_class: str
) -> dict[str, tuple[str, ...]]:
    """Return ``provider_id -> native labels`` for one semantic visual class."""

    result: dict[str, tuple[str, ...]] = {}
    for provider_id in visual_providers_for_class(catalog, semantic_class):
        result[provider_id] = tuple(
            catalog["providers"][provider_id]["semantic_classes"][semantic_class]
        )
    return result


def supported_visual_classes(catalog: Mapping[str, Any]) -> tuple[str, ...]:
    classes: set[str] = set()
    for provider_id, spec in catalog["providers"].items():
        if spec["kind"] == "visual_detector" and spec["enabled"]:
            classes.update(spec["semantic_classes"])
    return tuple(sorted(classes))


def predicate_providers(
    catalog: Mapping[str, Any], predicate_name: str
) -> tuple[str, ...]:
    """Return enabled providers that implement the named semantic predicate."""

    matches = []
    for provider_id, spec in catalog["providers"].items():
        if spec["kind"] != "predicate_implementation" or not spec["enabled"]:
            continue
        if predicate_name in spec["predicates"]:
            matches.append(provider_id)
    return tuple(sorted(matches))


def semantic_literal_values(
    catalog: Mapping[str, Any], predicate_name: str, argument_name: str
) -> tuple[str, ...]:
    """Return semantic literal values supported by at least one predicate provider."""

    values: set[str] = set()
    for provider_id in predicate_providers(catalog, predicate_name):
        predicate_spec = catalog["providers"][provider_id]["predicates"][predicate_name]
        values.update(predicate_spec["semantic_literals"].get(argument_name, {}))
    return tuple(sorted(values))


def provider_supports_predicate_call(
    catalog: Mapping[str, Any],
    provider_id: str,
    predicate_name: str,
    *,
    argument_classes: Mapping[str, str],
    literal_arguments: Mapping[str, Any],
) -> bool:
    """Return whether one predicate provider can implement a concrete CE leaf."""

    provider = catalog["providers"].get(provider_id)
    if not provider or not provider.get("enabled") or provider.get("kind") != "predicate_implementation":
        return False
    predicate_spec = provider["predicates"].get(predicate_name)
    if predicate_spec is None:
        return False

    restrictions = predicate_spec["visual_arguments"]
    for arg_name, semantic_class in argument_classes.items():
        if arg_name not in restrictions:
            # No class restriction was declared for this visual argument.
            continue
        allowed = restrictions[arg_name]
        if allowed != "*" and semantic_class not in allowed:
            return False

    literal_support = predicate_spec["semantic_literals"]
    for arg_name, allowed_values in literal_support.items():
        if arg_name in literal_arguments and literal_arguments[arg_name] not in allowed_values:
            return False

    return True


def visual_class_matches(
    catalog: Mapping[str, Any],
    expected_semantic_class: str,
    observed_class: str,
) -> bool:
    """Return whether an observed/native class can satisfy a semantic CE class.

    Predicate implementations may report the native class carried by their
    tracked objects (for example ``car``), while the CE can intentionally use a
    broader provider-independent class such as ``vehicle``.  The capability
    catalog is the single source of truth for that mapping.
    """

    if expected_semantic_class == observed_class:
        return True
    for provider_id in visual_providers_for_class(catalog, expected_semantic_class):
        native_labels = catalog["providers"][provider_id]["semantic_classes"][expected_semantic_class]
        if observed_class in native_labels:
            return True
        if canonical_semantic_class(observed_class) == expected_semantic_class:
            return True
    return False
