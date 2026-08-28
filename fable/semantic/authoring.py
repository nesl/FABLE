"""Readable public language for defining a FABLE complex event.

Input is a predicate vocabulary plus CE roles and logical/temporal operators.
Output is the same validated :class:`SemanticGraph` consumed by the semantic
runtime. Provider selection and deployment placement deliberately do not occur
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from fable.common.base import JSONValue
from fable.common.schemas import SemanticGraph
from fable.planning.predicate_registry import (
    PredicateSchemaError,
    PredicateSchemaRegistry,
    default_predicate_registry,
)

from .builder import AuthoredGraphBuilder, GraphCompileError, PredicateRoleSpec


@dataclass(frozen=True)
class EventNode:
    """Reference to one node authored through :class:`ComplexEvent`."""

    event: "ComplexEvent"
    key: str

    def build(self) -> SemanticGraph:
        """Select this node as the event root and compile the static graph."""

        return self.event.build(self)


class ComplexEvent:
    """Schema-aware facade for authoring one provider-independent event DAG.

    Authors declare CE roles once, bind predicate-local argument names to those
    roles, and combine returned :class:`EventNode` references. Predicate role
    names, entity types, parameters, and graph acyclicity are validated.
    """

    def __init__(
        self,
        family_id: str,
        *,
        namespace: str | None = None,
        name: str | None = None,
        description: str | None = None,
        graph_version: int = 1,
        predicate_registry: PredicateSchemaRegistry | None = None,
    ) -> None:
        if not family_id:
            raise GraphCompileError("family_id must be non-empty")
        self.family_id = family_id
        self.predicate_registry = predicate_registry or default_predicate_registry()
        self._builder = AuthoredGraphBuilder(
            namespace=namespace or f"fable.events.{family_id}",
            name=name or family_id.replace("_", " ").title(),
            description=description,
            graph_version=graph_version,
        )
        self._roles: dict[str, str] = {}
        self._counter = 0

    def role(
        self,
        role_name: str,
        entity_type: str,
        *,
        cardinality_min: int = 1,
        cardinality_max: int | None = 1,
        distinct_from: Iterable[str] = (),
    ) -> "ComplexEvent":
        """Declare one event-level entity variable and its type.

        Most roles use a concrete vocabulary type such as ``person``,
        ``vehicle``, ``zone``, or ``location``. ``entity`` is deliberately a
        generic identity slot: a predicate may narrow it to its concrete
        argument type. It does *not* mean arbitrary JSON or an untyped sensor
        object, and concrete incompatible types remain compile-time errors.
        """

        self._builder.role(
            role_name,
            entity_type,
            cardinality_min=cardinality_min,
            cardinality_max=cardinality_max,
            distinct_from=distinct_from,
        )
        self._roles[role_name] = entity_type
        return self

    def predicate(
        self,
        predicate_id: str,
        *,
        bind: Mapping[str, str] | None = None,
        parameters: Mapping[str, JSONValue] | None = None,
        key: str | None = None,
        name: str | None = None,
        checkpoint: bool = False,
        annotations: Mapping[str, object] | None = None,
    ) -> EventNode:
        """Add one predicate node with schema-checked argument bindings.

        ``bind`` maps predicate-local arguments to declared CE roles. For
        example, ``{"left": "vehicle_a"}`` fills the predicate's ``left``
        argument with the event role ``vehicle_a``. Entity types are inferred
        from the event-role declaration and predicate schema.
        """

        try:
            schema = self.predicate_registry.get(predicate_id)
        except PredicateSchemaError as exc:
            raise GraphCompileError(str(exc)) from exc
        bindings = dict(bind or {})
        expected = {role.role_name: role for role in schema.roles}
        if set(bindings) != set(expected):
            raise GraphCompileError(
                f"{predicate_id} bindings {sorted(bindings)} do not match predicate "
                f"arguments {sorted(expected)}"
            )
        role_specs: list[PredicateRoleSpec] = []
        for argument, event_role in bindings.items():
            if event_role not in self._roles:
                raise GraphCompileError(
                    f"{predicate_id}.{argument} references undeclared CE role {event_role!r}"
                )
            event_type = self._roles[event_role]
            predicate_type = expected[argument].entity_type
            if event_type != predicate_type and event_type != "entity":
                raise GraphCompileError(
                    f"{predicate_id}.{argument} expects {predicate_type!r}; "
                    f"CE role {event_role!r} is {event_type!r}"
                )
            role_specs.append(PredicateRoleSpec(argument, event_role, predicate_type))
        node_key = key or self._next_key(predicate_id.lower())
        self._builder.predicate(
            node_key,
            name=name or predicate_id.replace("_", " ").title(),
            predicate_id=predicate_id,
            roles=tuple(role_specs),
            result_kind=schema.result_kind,
            parameters=parameters,
            checkpoint=checkpoint,
            annotations=annotations,
        )
        # Validate parameters as soon as possible, rather than deferring the
        # error until demand compilation.
        node = self._builder._nodes[node_key]  # lower-level draft owned here
        assert node.predicate is not None
        try:
            self.predicate_registry.validate(node.predicate)
        except PredicateSchemaError as exc:
            raise GraphCompileError(str(exc)) from exc
        return EventNode(self, node_key)

    def all_of(self, *children: EventNode, key: str | None = None, name: str = "All") -> EventNode:
        """Require every child, without imposing an order."""

        return self._composite("all", children, key, name, self._builder.and_group)

    def any_of(self, *children: EventNode, key: str | None = None, name: str = "Any") -> EventNode:
        """Require any one labeled branch."""

        self._check_children(children)
        node_key = key or self._next_key("any")
        branches = {f"branch_{index + 1}": child.key for index, child in enumerate(children)}
        self._builder.or_group(node_key, branches, name=name)
        return EventNode(self, node_key)

    def sequence(self, *children: EventNode, key: str | None = None, name: str = "Sequence") -> EventNode:
        """Require children in event-time order."""

        return self._composite("sequence", children, key, name, self._builder.sequence)

    def k_of(self, k: int, *children: EventNode, key: str | None = None, name: str = "K of N") -> EventNode:
        """Require at least ``k`` of the supplied children."""

        self._check_children(children)
        node_key = key or self._next_key("k_of")
        self._builder.k_of_n(node_key, tuple(item.key for item in children), k=k, name=name)
        return EventNode(self, node_key)

    def duration(self, child: EventNode, *, minimum_ms: int, key: str | None = None, name: str = "Duration") -> EventNode:
        self._check_children((child,))
        node_key = key or self._next_key("duration")
        self._builder.duration(node_key, child.key, minimum_ms=minimum_ms, name=name)
        return EventNode(self, node_key)

    def within(self, child: EventNode, *, after: Iterable[EventNode], maximum_ms: int, minimum_ms: int | None = None, key: str | None = None, name: str = "Within") -> EventNode:
        predecessors = tuple(after)
        self._check_children((child, *predecessors))
        node_key = key or self._next_key("within")
        self._builder.within(
            node_key,
            child.key,
            after=tuple(item.key for item in predecessors),
            maximum_ms=maximum_ms,
            minimum_ms=minimum_ms,
            name=name,
        )
        return EventNode(self, node_key)

    def absent(self, child: EventNode, *, window_ms: int, required_source_ids: Iterable[str], key: str | None = None, name: str = "Absent") -> EventNode:
        self._check_children((child,))
        node_key = key or self._next_key("absent")
        self._builder.absent(
            node_key,
            child.key,
            window_ms=window_ms,
            required_source_ids=required_source_ids,
            name=name,
        )
        return EventNode(self, node_key)

    def build(self, root: EventNode | None = None) -> SemanticGraph:
        """Compile and validate this event with ``root`` as its terminal node."""

        if root is not None:
            self._check_children((root,))
            self._builder.root(root.key)
        return self._builder.compile()

    def _composite(self, prefix, children, key, name, method) -> EventNode:
        self._check_children(children)
        node_key = key or self._next_key(prefix)
        method(node_key, tuple(item.key for item in children), name=name)
        return EventNode(self, node_key)

    def _check_children(self, children: Iterable[EventNode]) -> None:
        values = tuple(children)
        if not values:
            raise GraphCompileError("a structural operator requires at least one child")
        if any(item.event is not self for item in values):
            raise GraphCompileError("cannot combine nodes from different complex events")

    def _next_key(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"


__all__ = ["ComplexEvent", "EventNode"]
