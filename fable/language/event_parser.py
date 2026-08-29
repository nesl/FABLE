"""Load and validate declarative complex-event definitions.

A CE file contains only:

* a language version,
* an event name and optional description,
* logical object roles with semantic class constraints, and
* one recursive pattern.

Provider selection, cameras, compute nodes, artifacts, network state, and
runtime scheduling are outside this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import yaml

from .pattern_parser import Expr, parse_pattern
from .predicates import PredicateCatalog, load_predicates


_EVENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ROLE_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_CLASS_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class Event:
    """Parsed static definition for one complex-event family.

    ``roles`` maps each logical role name to its semantic object-class
    constraint, e.g. ``{"VEHICLE": "vehicle", "DOG": "dog"}``.
    These class names are intentionally provider-independent; a later provider
    layer may map one semantic class to one or more model-specific labels.
    """

    name: str
    roles: Mapping[str, str]
    pattern: Expr
    description: str = ""
    version: int = 1


def load_event(
    path: str | Path,
    *,
    predicate_catalog: PredicateCatalog | None = None,
) -> Event:
    """Load one YAML (or JSON, which is valid YAML) CE definition from disk."""

    event_path = Path(path)
    with event_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_event(
        raw,
        predicate_catalog=predicate_catalog,
        source=str(event_path),
    )


def parse_event(
    raw: object,
    *,
    predicate_catalog: PredicateCatalog | None = None,
    source: str = "<memory>",
) -> Event:
    """Validate a loaded YAML/JSON object and return the compact CE AST."""

    if not isinstance(raw, dict):
        raise ValueError(f"{source}: CE definition must be a mapping")

    unknown = set(raw) - {"version", "event", "description", "roles", "pattern"}
    if unknown:
        raise ValueError(f"{source}: unknown top-level fields {sorted(unknown)}")

    version = raw.get("version", 1)
    if version != 1:
        raise ValueError(f"{source}: only CE language version 1 is supported")

    name = raw.get("event")
    if not isinstance(name, str) or not _EVENT_NAME_RE.fullmatch(name):
        raise ValueError(
            f"{source}.event: expected lowercase snake_case name, got {name!r}"
        )

    description = raw.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"{source}.description: expected string")

    roles = _parse_roles(raw.get("roles"), source=source)
    if "pattern" not in raw:
        raise ValueError(f"{source}: missing required top-level field 'pattern'")

    catalog = predicate_catalog if predicate_catalog is not None else load_predicates()
    pattern = parse_pattern(
        raw["pattern"],
        roles=roles,
        predicates=catalog,
        path=f"{source}.pattern",
    )

    return Event(
        name=name,
        roles=roles,
        pattern=pattern,
        description=description,
        version=version,
    )


def _parse_roles(raw: object, *, source: str) -> dict[str, str]:
    """Parse authored object roles into ``role -> semantic class`` mappings."""

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source}.roles: expected a mapping")

    roles: dict[str, str] = {}
    for role_name, role_spec in raw.items():
        role_path = f"{source}.roles.{role_name}"
        if not isinstance(role_name, str) or not _ROLE_NAME_RE.fullmatch(role_name):
            raise ValueError(
                f"{source}.roles: role names must be UPPERCASE identifiers; "
                f"got {role_name!r}"
            )
        if not isinstance(role_spec, dict):
            raise ValueError(
                f"{role_path}: expected an object such as {{'class': 'vehicle'}}"
            )
        unknown = set(role_spec) - {"class"}
        if unknown:
            raise ValueError(f"{role_path}: unknown fields {sorted(unknown)}")
        if "class" not in role_spec:
            raise ValueError(f"{role_path}: missing required field 'class'")

        class_name = role_spec["class"]
        if not isinstance(class_name, str) or not _CLASS_NAME_RE.fullmatch(class_name):
            raise ValueError(
                f"{role_path}.class: expected lowercase semantic class identifier, "
                f"got {class_name!r}"
            )
        roles[role_name] = class_name

    return roles
