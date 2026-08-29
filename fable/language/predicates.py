"""Predicate catalog loading and validation for the FABLE CE language.

A predicate catalog entry contains only a semantic signature: named arguments
and their small language-level types.  It deliberately contains no provider,
model, artifact, camera, placement, or binding-policy information.

Visual-object-valued arguments refer to UPPERCASE logical roles declared by
the CE.  The role itself carries a provider-independent semantic class such as
``dog`` or ``vehicle``.  Mapping that class to a model-specific label set (for
example COCO labels) belongs to the later provider layer.

Literal-valued arguments (for example ``audio_class`` or ``number``) are written
directly in the predicate call.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


PredicateCatalog = dict[str, dict[str, Any]]
_SUPPORTED_TYPES = frozenset({"visual_object", "audio_class", "string", "number", "integer", "boolean"})
_CLASS_LITERAL_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _default_catalog_path() -> Path:
    return Path(__file__).with_name("predicates.yaml")


@lru_cache(maxsize=8)
def _load_catalog_cached(path_string: str) -> PredicateCatalog:
    path = Path(path_string)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: predicate catalog must be a mapping")
    if raw.get("version") != 1:
        raise ValueError(f"{path}: expected predicate catalog version 1")

    predicates = raw.get("predicates")
    if not isinstance(predicates, dict) or not predicates:
        raise ValueError(f"{path}: 'predicates' must be a non-empty mapping")

    validated: PredicateCatalog = {}
    for name, spec in predicates.items():
        if not isinstance(name, str) or not name.isidentifier() or name.lower() != name:
            raise ValueError(
                f"{path}: predicate names must be lowercase identifiers; got {name!r}"
            )
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: predicate {name!r} must map to an object")

        unknown = set(spec) - {"description", "arguments"}
        if unknown:
            raise ValueError(
                f"{path}: predicate {name!r} has unknown fields {sorted(unknown)}"
            )

        arguments = spec.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError(f"{path}: predicate {name!r}.arguments must be a mapping")

        normalized_arguments: dict[str, dict[str, Any]] = {}
        for arg_name, arg_spec in arguments.items():
            if not isinstance(arg_name, str) or not arg_name.isidentifier() or arg_name.lower() != arg_name:
                raise ValueError(
                    f"{path}: predicate {name!r} argument names must be lowercase "
                    f"identifiers; got {arg_name!r}"
                )
            normalized_arguments[arg_name] = _normalize_argument_spec(
                arg_spec,
                path=f"{path}: predicate {name!r}.arguments.{arg_name}",
            )

        validated[name] = {
            "description": str(spec.get("description", "")),
            "arguments": normalized_arguments,
        }

    return validated


def _normalize_argument_spec(value: object, *, path: str) -> dict[str, Any]:
    """Normalize shorthand and full predicate-argument specifications."""

    if isinstance(value, str):
        spec: dict[str, Any] = {"type": value, "required": True}
    elif isinstance(value, dict):
        unknown = set(value) - {"type", "required", "minimum", "maximum", "enum"}
        if unknown:
            raise ValueError(f"{path}: unknown fields {sorted(unknown)}")
        if "type" not in value:
            raise ValueError(f"{path}: missing required field 'type'")
        spec = dict(value)
        spec.setdefault("required", True)
    else:
        raise ValueError(f"{path}: expected a type string or argument object")

    expected = spec.get("type")
    if expected not in _SUPPORTED_TYPES:
        raise ValueError(f"{path}: unsupported argument type {expected!r}")
    if not isinstance(spec.get("required"), bool):
        raise ValueError(f"{path}.required: expected boolean")

    if expected == "visual_object" and any(key in spec for key in ("minimum", "maximum", "enum")):
        raise ValueError(f"{path}: visual_object arguments cannot use minimum/maximum/enum")

    return spec


def load_predicates(path: str | Path | None = None) -> PredicateCatalog:
    """Load and validate the semantic predicate catalog."""

    catalog_path = Path(path) if path is not None else _default_catalog_path()
    return _load_catalog_cached(str(catalog_path.resolve()))


def validate_predicate_call(
    name: str,
    values: object,
    *,
    event_roles: Mapping[str, str],
    catalog: Mapping[str, Mapping[str, Any]],
    path: str,
) -> dict[str, Any]:
    """Validate one predicate invocation and return a normalized plain mapping."""

    if name not in catalog:
        raise ValueError(f"{path}: unknown predicate {name!r}")
    if not isinstance(values, dict):
        raise ValueError(f"{path}: predicate {name!r} must map to an object")

    arg_specs = catalog[name].get("arguments", {})
    allowed = set(arg_specs)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"{path}: predicate {name!r} has unknown fields {sorted(unknown)}"
        )

    missing = [
        arg_name
        for arg_name, arg_spec in arg_specs.items()
        if arg_spec.get("required", True) and arg_name not in values
    ]
    if missing:
        raise ValueError(
            f"{path}: predicate {name!r} is missing required arguments {missing}"
        )

    normalized: dict[str, Any] = {}
    for arg_name, arg_spec in arg_specs.items():
        if arg_name not in values:
            continue
        value = values[arg_name]
        arg_path = f"{path}.{name}.{arg_name}"
        if arg_spec["type"] == "visual_object":
            _validate_visual_object_reference(value, event_roles=event_roles, path=arg_path)
        else:
            _validate_literal(value, arg_spec, path=arg_path)
        normalized[arg_name] = value

    return normalized


def _validate_visual_object_reference(
    value: object,
    *,
    event_roles: Mapping[str, str],
    path: str,
) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{path}: visual_object argument must be a role name")
    if value not in event_roles:
        if value.isupper():
            raise ValueError(f"{path}: unknown role {value!r}")
        raise ValueError(
            f"{path}: expected an UPPERCASE visual-object role name, got literal {value!r}"
        )


def _validate_literal(value: object, spec: Mapping[str, Any], *, path: str) -> None:
    expected = spec["type"]

    if expected == "audio_class":
        if not isinstance(value, str) or not _CLASS_LITERAL_RE.fullmatch(value):
            raise ValueError(
                f"{path}: expected lowercase audio class identifier such as 'gunshot'"
            )
    elif expected == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected string")
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: expected number")
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path}: expected integer")
    elif expected == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected boolean")
    else:
        raise ValueError(f"predicate catalog error: unsupported type {expected!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = spec.get("minimum")
        maximum = spec.get("maximum")
        if minimum is not None and value < minimum:
            raise ValueError(f"{path}: must be >= {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{path}: must be <= {maximum}")

    enum = spec.get("enum")
    if enum is not None:
        if not isinstance(enum, list):
            raise ValueError(f"predicate catalog error: {path}.enum must be a list")
        if value not in enum:
            raise ValueError(f"{path}: expected one of {enum}, got {value!r}")
