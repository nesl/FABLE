"""Parse the recursive pattern grammar for FABLE complex-event definitions.

This module defines the compact expression-tree representation used by the
language layer and parses authored structure operators into that representation.
It does not maintain runtime hypothesis/frontier state.

The multi-branch operators ``all`` and ``k_of_n`` carry a default *join window*.
The runtime interpretation is: once the first direct child of the operator is
satisfied, the operator has ``window_ms`` to collect the remaining required
child satisfactions.  Resolving that default here means later runtime code can
read the value from the AST rather than duplicating a hidden five-minute
constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterator, Mapping

from .predicates import validate_predicate_call


STRUCTURE_OPS = frozenset({"seq", "all", "any", "k_of_n", "within", "for"})

# Default completion window for structure operators that join multiple branches.
# The clock starts when the first direct child becomes satisfied.
DEFAULT_JOIN_WINDOW_MS = 5 * 60 * 1_000

_DURATION_RE = re.compile(r"^(?P<value>(?:\d+(?:\.\d*)?|\.\d+))(?P<unit>ms|s|m|h)$")


@dataclass(frozen=True, slots=True)
class Expr:
    """One node in a complex-event expression tree.

    ``op`` is either a structure operator or a semantic predicate name.

    Only fields relevant to the selected operator are populated.  ``window_ms``
    is used by ``all`` and ``k_of_n``.  ``k`` is used only by ``k_of_n``.
    ``min_ms``/``max_ms`` belong to ``within`` and ``duration_ms`` belongs to
    ``for``.
    """

    op: str
    children: tuple["Expr", ...] = ()
    args: Mapping[str, Any] = field(default_factory=dict)
    k: int | None = None
    window_ms: int | None = None
    min_ms: int | None = None
    max_ms: int | None = None
    duration_ms: int | None = None

    @property
    def is_predicate(self) -> bool:
        return self.op not in STRUCTURE_OPS


def parse_pattern(
    raw: object,
    *,
    roles: Mapping[str, str],
    predicates: Mapping[str, Mapping[str, Any]],
    path: str = "pattern",
) -> Expr:
    """Parse one recursive pattern mapping into an :class:`Expr` tree."""

    if not isinstance(raw, dict) or len(raw) != 1:
        raise ValueError(
            f"{path}: every pattern must be a mapping with exactly one operator"
        )

    op, payload = next(iter(raw.items()))
    if not isinstance(op, str):
        raise ValueError(f"{path}: pattern operator must be a string")
    if op.lower() != op:
        raise ValueError(f"{path}: pattern/predicate names must be lowercase; got {op!r}")

    if op in {"seq", "all", "any"}:
        return _parse_group(op, payload, roles=roles, predicates=predicates, path=path)
    if op == "k_of_n":
        return _parse_k_of_n(payload, roles=roles, predicates=predicates, path=path)
    if op == "within":
        return _parse_within(payload, roles=roles, predicates=predicates, path=path)
    if op == "for":
        return _parse_for(payload, roles=roles, predicates=predicates, path=path)

    args = validate_predicate_call(
        op,
        payload,
        event_roles=roles,
        catalog=predicates,
        path=path,
    )
    return Expr(op=op, args=args)


def _parse_group(
    op: str,
    payload: object,
    *,
    roles: Mapping[str, str],
    predicates: Mapping[str, Mapping[str, Any]],
    path: str,
) -> Expr:
    if not isinstance(payload, list):
        raise ValueError(f"{path}.{op}: expected a list of child patterns")
    if len(payload) < 2:
        raise ValueError(f"{path}.{op}: requires at least two child patterns")

    children = tuple(
        parse_pattern(
            child,
            roles=roles,
            predicates=predicates,
            path=f"{path}.{op}[{index}]",
        )
        for index, child in enumerate(payload)
    )

    # ``all`` is a temporal join rather than an unbounded conjunction.  Once
    # the first direct child is satisfied, the rest must complete within the
    # resolved join window.  ``seq`` and ``any`` do not use this field.
    window_ms = DEFAULT_JOIN_WINDOW_MS if op == "all" else None
    return Expr(op=op, children=children, window_ms=window_ms)


def _parse_k_of_n(
    payload: object,
    *,
    roles: Mapping[str, str],
    predicates: Mapping[str, Mapping[str, Any]],
    path: str,
) -> Expr:
    """Parse a K-of-N join.

    Authored form::

        k_of_n:
          k: 2
          patterns:
            - ...
            - ...
            - ...

    As with ``all``, the join window begins when the first direct child is
    satisfied.  The operator succeeds once any ``k`` distinct children have
    been satisfied before the window expires.
    """

    node_path = f"{path}.k_of_n"
    if not isinstance(payload, dict):
        raise ValueError(f"{node_path}: expected an object")

    unknown = set(payload) - {"k", "patterns"}
    if unknown:
        raise ValueError(f"{node_path}: unknown fields {sorted(unknown)}")
    if "k" not in payload:
        raise ValueError(f"{node_path}: requires 'k'")
    if "patterns" not in payload:
        raise ValueError(f"{node_path}: requires 'patterns'")

    k = payload["k"]
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError(f"{node_path}.k: expected an integer")

    raw_children = payload["patterns"]
    if not isinstance(raw_children, list):
        raise ValueError(f"{node_path}.patterns: expected a list of child patterns")
    if len(raw_children) < 2:
        raise ValueError(f"{node_path}.patterns: requires at least two child patterns")
    if k < 1 or k > len(raw_children):
        raise ValueError(
            f"{node_path}.k: must be between 1 and the number of patterns "
            f"({len(raw_children)}); got {k}"
        )

    children = tuple(
        parse_pattern(
            child,
            roles=roles,
            predicates=predicates,
            path=f"{node_path}.patterns[{index}]",
        )
        for index, child in enumerate(raw_children)
    )
    return Expr(
        op="k_of_n",
        children=children,
        k=k,
        window_ms=DEFAULT_JOIN_WINDOW_MS,
    )


def _parse_within(
    payload: object,
    *,
    roles: Mapping[str, str],
    predicates: Mapping[str, Mapping[str, Any]],
    path: str,
) -> Expr:
    node_path = f"{path}.within"
    if not isinstance(payload, dict):
        raise ValueError(f"{node_path}: expected an object")
    unknown = set(payload) - {"min", "max", "pattern"}
    if unknown:
        raise ValueError(f"{node_path}: unknown fields {sorted(unknown)}")
    if "max" not in payload:
        raise ValueError(f"{node_path}: requires 'max'")
    if "pattern" not in payload:
        raise ValueError(f"{node_path}: requires 'pattern'")

    minimum = parse_duration_ms(payload["min"], path=f"{node_path}.min") if "min" in payload else None
    maximum = parse_duration_ms(payload["max"], path=f"{node_path}.max")
    if maximum <= 0:
        raise ValueError(f"{node_path}.max: must be greater than zero")
    if minimum is not None:
        if minimum < 0:
            raise ValueError(f"{node_path}.min: must not be negative")
        if minimum > maximum:
            raise ValueError(f"{node_path}: 'min' must be <= 'max'")

    child = parse_pattern(
        payload["pattern"],
        roles=roles,
        predicates=predicates,
        path=f"{node_path}.pattern",
    )
    return Expr(op="within", children=(child,), min_ms=minimum, max_ms=maximum)


def _parse_for(
    payload: object,
    *,
    roles: Mapping[str, str],
    predicates: Mapping[str, Mapping[str, Any]],
    path: str,
) -> Expr:
    node_path = f"{path}.for"
    if not isinstance(payload, dict):
        raise ValueError(f"{node_path}: expected an object")
    unknown = set(payload) - {"duration", "pattern"}
    if unknown:
        raise ValueError(f"{node_path}: unknown fields {sorted(unknown)}")
    if "duration" not in payload:
        raise ValueError(f"{node_path}: requires 'duration'")
    if "pattern" not in payload:
        raise ValueError(f"{node_path}: requires 'pattern'")

    duration = parse_duration_ms(payload["duration"], path=f"{node_path}.duration")
    if duration <= 0:
        raise ValueError(f"{node_path}.duration: must be greater than zero")

    child = parse_pattern(
        payload["pattern"],
        roles=roles,
        predicates=predicates,
        path=f"{node_path}.pattern",
    )
    return Expr(op="for", children=(child,), duration_ms=duration)


def parse_duration_ms(value: object, *, path: str) -> int:
    """Parse a human-readable duration (``500ms``, ``3s``, ``5m``, ``1h``)."""

    if not isinstance(value, str):
        raise ValueError(
            f"{path}: duration must be a string with units, e.g. '500ms', '3s', '5m'"
        )
    match = _DURATION_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(
            f"{path}: invalid duration {value!r}; use ms, s, m, or h (for example '30s')"
        )

    amount = float(match.group("value"))
    unit = match.group("unit")
    factor = {"ms": 1.0, "s": 1_000.0, "m": 60_000.0, "h": 3_600_000.0}[unit]
    milliseconds = amount * factor
    rounded = round(milliseconds)
    if abs(milliseconds - rounded) > 1e-9:
        raise ValueError(f"{path}: duration must resolve to a whole number of milliseconds")
    return int(rounded)


def walk_pattern(expr: Expr) -> Iterator[Expr]:
    """Yield an expression and all descendants in pre-order."""

    yield expr
    for child in expr.children:
        yield from walk_pattern(child)
