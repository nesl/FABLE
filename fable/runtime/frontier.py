"""Discovery and candidate-specific continuation frontiers.

The frontier is the semantic-to-physical boundary of the rebuilt FABLE.  A
frontier item is already resolved against the current candidate bindings: it
states which predicate result is useful, which object identities are fixed,
which roles are still unbound, their semantic classes, predicate parameters,
and any semantic expiry / sustained-duration constraint.

There is deliberately no separate ``EvidenceDemand`` object.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Mapping

from fable.language.event_parser import Event
from fable.language.pattern_parser import Expr
from fable.language.predicates import PredicateCatalog, load_predicates
from fable.providers.predicate_result import PredicateMatch
from fable.providers.provider_capabilities import load_provider_capabilities, visual_class_matches

from .ce_instance import CEInstance, PatternPath


@dataclass(frozen=True, slots=True)
class FrontierItem:
    """One currently useful semantic predicate requirement."""

    path: PatternPath
    predicate: str

    # Predicate argument -> already-bound object identity or None.
    arguments: Mapping[str, str | None] = field(default_factory=dict)

    # Predicate argument -> authored semantic object class.
    classes: Mapping[str, str] = field(default_factory=dict)

    # Predicate argument -> authored role name.  This is runtime metadata used
    # to bind an unbound match back into the CE instance.
    role_refs: Mapping[str, str] = field(default_factory=dict)

    # Literal/configuration arguments such as max_gap_m or class: gunshot.
    parameters: Mapping[str, Any] = field(default_factory=dict)

    # Semantic usefulness window.  This is intentionally named ``expires_at``
    # rather than "deadline" to distinguish it from later scheduling QoS.
    expires_at: datetime | None = None

    # Used by ``for:``.  Providers still emit ordinary PredicateMatch records;
    # the semantic runtime decides when the sustained duration has been met.
    required_duration_ms: int | None = None
    sustain_path: PatternPath | None = None


@dataclass(frozen=True, slots=True)
class ActiveFrontier:
    """Persistent discovery work plus per-instance continuation work."""

    discovery: tuple[FrontierItem, ...]
    continuation: Mapping[str, tuple[FrontierItem, ...]]


def derive_discovery_frontier(
    event: Event,
    *,
    predicate_catalog: PredicateCatalog | None = None,
) -> tuple[FrontierItem, ...]:
    """Return the predicates that can start a new candidate CE instance.

    This frontier is structural and persistent for the lifetime of the CE
    query.  It is not represented as a fake/root CE instance.
    """

    catalog = predicate_catalog if predicate_catalog is not None else load_predicates()
    return tuple(_discover(event.pattern, (), event=event, catalog=catalog))


def derive_continuation_frontier(
    event: Event,
    instance: CEInstance,
    now: datetime,
    *,
    predicate_catalog: PredicateCatalog | None = None,
) -> tuple[FrontierItem, ...]:
    """Return the evidence that can advance one candidate instance now.

    The function also performs the small amount of temporal bookkeeping needed
    by structure operators: activations, join windows, explicit ``within``
    expirations, completion propagation, and expiry propagation.
    """

    _require_aware(now, "now")
    if now < instance.matched_at:
        raise ValueError("now cannot be earlier than CEInstance.matched_at")
    if instance.completed or instance.failed:
        return ()

    catalog = predicate_catalog if predicate_catalog is not None else load_predicates()
    status, _, frontier = _visit(
        event.pattern,
        (),
        instance,
        now,
        event=event,
        catalog=catalog,
        activation=instance.matched_at,
        inherited_expiry=None,
    )
    if status == "satisfied":
        instance.failed_paths.pop((), None)
    return tuple(frontier)


def advance_instance(
    event: Event,
    instance: CEInstance,
    match: PredicateMatch,
    *,
    predicate_catalog: PredicateCatalog | None = None,
) -> tuple[CEInstance, ...]:
    """Return all candidate branches produced by applying one match.

    The input instance is not semantically consumed.  A higher-level instance
    manager may preserve it so that later observations can form alternative
    valid CE occurrences from the same prefix (skip-till-any-match behavior).
    """

    catalog = predicate_catalog if predicate_catalog is not None else load_predicates()
    frontier = derive_continuation_frontier(
        event, instance, match.event_time, predicate_catalog=catalog
    )
    if instance.failed or instance.completed:
        return ()

    branches: list[CEInstance] = []
    for item in frontier:
        candidate_bindings = _matched_bindings(item, match, instance, catalog=catalog)
        if candidate_bindings is None:
            continue
        branch = instance.clone(instance_id="")
        branch.bindings.clear()
        branch.bindings.update(candidate_bindings)
        _record_match(branch, item, match.event_time)
        # Propagate surrounding structure completion / expiry immediately.
        derive_continuation_frontier(
            event, branch, match.event_time, predicate_catalog=catalog
        )
        branches.append(branch)
    return tuple(branches)


def seed_instance_from_match(
    event: Event,
    item: FrontierItem,
    match: PredicateMatch,
    *,
    predicate_catalog: PredicateCatalog | None = None,
) -> CEInstance | None:
    """Create one candidate from a match against one discovery-frontier item."""

    catalog = predicate_catalog if predicate_catalog is not None else load_predicates()
    instance = CEInstance(
        event_name=event.name,
        matched_at=match.event_time,
        matched_source=_matched_source(match),
        matched_predicate=match.predicate,
        matched_path=item.path,
    )
    candidate_bindings = _matched_bindings(item, match, instance, catalog=catalog)
    if candidate_bindings is None:
        return None
    instance.bindings.update(candidate_bindings)
    _record_match(instance, item, match.event_time)
    derive_continuation_frontier(
        event, instance, match.event_time, predicate_catalog=catalog
    )
    return instance


def is_complete(instance: CEInstance) -> bool:
    return instance.completed


def is_failed(instance: CEInstance) -> bool:
    return instance.failed


def _discover(
    expr: Expr,
    path: PatternPath,
    *,
    event: Event,
    catalog: Mapping[str, Mapping[str, Any]],
) -> list[FrontierItem]:
    if expr.is_predicate:
        return [_make_item(expr, path, event=event, instance=None, catalog=catalog)]

    if expr.op == "seq":
        return _discover(expr.children[0], path + (0,), event=event, catalog=catalog)

    if expr.op in {"all", "any", "k_of_n"}:
        return [
            item
            for index, child in enumerate(expr.children)
            for item in _discover(child, path + (index,), event=event, catalog=catalog)
        ]

    if expr.op == "within":
        # Before a candidate exists there is no preceding semantic completion
        # from which to derive this wrapper's relative timing.  Its child can
        # still seed a candidate; timing is enforced once the candidate exists.
        return _discover(expr.children[0], path + (0,), event=event, catalog=catalog)

    if expr.op == "for":
        child = expr.children[0]
        if not child.is_predicate:
            raise NotImplementedError(
                "the minimal runtime currently supports 'for' around a predicate leaf only"
            )
        item = _make_item(
            child,
            path + (0,),
            event=event,
            instance=None,
            catalog=catalog,
            required_duration_ms=expr.duration_ms,
            sustain_path=path,
        )
        return [item]

    raise ValueError(f"unsupported runtime structure operator {expr.op!r}")


def _visit(
    expr: Expr,
    path: PatternPath,
    instance: CEInstance,
    now: datetime,
    *,
    event: Event,
    catalog: Mapping[str, Mapping[str, Any]],
    activation: datetime,
    inherited_expiry: datetime | None,
) -> tuple[str, datetime | None, list[FrontierItem]]:
    if path in instance.satisfied_at:
        return "satisfied", instance.satisfied_at[path], []
    if path in instance.failed_paths:
        return "failed", None, []

    instance.activated_at.setdefault(path, activation)

    if expr.is_predicate:
        return "pending", None, [
            _make_item(
                expr,
                path,
                event=event,
                instance=instance,
                catalog=catalog,
                expires_at=inherited_expiry,
            )
        ]

    if expr.op == "seq":
        child_activation = activation
        for index, child in enumerate(expr.children):
            child_path = path + (index,)
            status, completed, frontier = _visit(
                child,
                child_path,
                instance,
                now,
                event=event,
                catalog=catalog,
                activation=child_activation,
                inherited_expiry=inherited_expiry,
            )
            if status == "failed":
                return _fail(instance, path, f"sequence child {index} failed")
            if status == "pending":
                return "pending", None, frontier
            assert completed is not None
            child_activation = completed

        instance.satisfied_at[path] = child_activation
        return "satisfied", child_activation, []

    if expr.op == "all":
        results = [
            _visit(
                child,
                path + (index,),
                instance,
                now,
                event=event,
                catalog=catalog,
                activation=activation,
                inherited_expiry=inherited_expiry,
            )
            for index, child in enumerate(expr.children)
        ]
        if any(status == "failed" for status, _, _ in results):
            return _fail(instance, path, "required all-branch failed")

        completed_times = [
            completed
            for status, completed, _ in results
            if status == "satisfied" and completed is not None
        ]
        if len(completed_times) == len(expr.children):
            completed = max(completed_times)
            instance.satisfied_at[path] = completed
            return "satisfied", completed, []

        join_expiry = instance.expires_at.get(path)
        if completed_times and join_expiry is None:
            first = min(completed_times)
            join_expiry = first + _ms(expr.window_ms)
            instance.expires_at[path] = join_expiry
        effective_expiry = _earliest(inherited_expiry, join_expiry)
        if effective_expiry is not None and now > effective_expiry:
            return _fail(instance, path, "all join window expired")

        frontier = _with_expiry(
            [item for status, _, items in results if status == "pending" for item in items],
            effective_expiry,
        )
        return "pending", None, frontier

    if expr.op == "any":
        results = [
            _visit(
                child,
                path + (index,),
                instance,
                now,
                event=event,
                catalog=catalog,
                activation=activation,
                inherited_expiry=inherited_expiry,
            )
            for index, child in enumerate(expr.children)
        ]
        completed_times = [
            completed
            for status, completed, _ in results
            if status == "satisfied" and completed is not None
        ]
        if completed_times:
            completed = min(completed_times)
            instance.satisfied_at[path] = completed
            return "satisfied", completed, []
        if all(status == "failed" for status, _, _ in results):
            return _fail(instance, path, "all any-branches failed")
        return "pending", None, [
            item for status, _, items in results if status == "pending" for item in items
        ]

    if expr.op == "k_of_n":
        assert expr.k is not None
        results = [
            _visit(
                child,
                path + (index,),
                instance,
                now,
                event=event,
                catalog=catalog,
                activation=activation,
                inherited_expiry=inherited_expiry,
            )
            for index, child in enumerate(expr.children)
        ]
        completed_times = sorted(
            completed
            for status, completed, _ in results
            if status == "satisfied" and completed is not None
        )
        if len(completed_times) >= expr.k:
            completed = completed_times[expr.k - 1]
            instance.satisfied_at[path] = completed
            return "satisfied", completed, []

        pending_count = sum(status == "pending" for status, _, _ in results)
        if len(completed_times) + pending_count < expr.k:
            return _fail(instance, path, "too few k-of-n branches remain possible")

        join_expiry = instance.expires_at.get(path)
        if completed_times and join_expiry is None:
            join_expiry = completed_times[0] + _ms(expr.window_ms)
            instance.expires_at[path] = join_expiry
        effective_expiry = _earliest(inherited_expiry, join_expiry)
        if effective_expiry is not None and now > effective_expiry:
            return _fail(instance, path, "k-of-n join window expired")

        frontier = _with_expiry(
            [item for status, _, items in results if status == "pending" for item in items],
            effective_expiry,
        )
        return "pending", None, frontier

    if expr.op == "within":
        assert expr.max_ms is not None and len(expr.children) == 1
        minimum_time = activation + _ms(expr.min_ms or 0)
        maximum_time = activation + _ms(expr.max_ms)
        instance.expires_at[path] = maximum_time
        effective_expiry = _earliest(inherited_expiry, maximum_time)

        if now < minimum_time:
            return "pending", None, []

        status, completed, frontier = _visit(
            expr.children[0],
            path + (0,),
            instance,
            now,
            event=event,
            catalog=catalog,
            activation=minimum_time,
            inherited_expiry=effective_expiry,
        )
        if status == "satisfied" and completed is not None:
            if minimum_time <= completed <= maximum_time:
                instance.satisfied_at[path] = completed
                return "satisfied", completed, []
        if status == "failed":
            return _fail(instance, path, "within child failed")
        if now > maximum_time:
            return _fail(instance, path, "within window expired")
        return "pending", None, _with_expiry(frontier, effective_expiry)

    if expr.op == "for":
        assert expr.duration_ms is not None and len(expr.children) == 1
        child = expr.children[0]
        if not child.is_predicate:
            raise NotImplementedError(
                "the minimal runtime currently supports 'for' around a predicate leaf only"
            )
        if inherited_expiry is not None and now > inherited_expiry:
            return _fail(instance, path, "ancestor window expired during sustained predicate")
        return "pending", None, [
            _make_item(
                child,
                path + (0,),
                event=event,
                instance=instance,
                catalog=catalog,
                expires_at=inherited_expiry,
                required_duration_ms=expr.duration_ms,
                sustain_path=path,
            )
        ]

    raise ValueError(f"unsupported runtime structure operator {expr.op!r}")


def _make_item(
    expr: Expr,
    path: PatternPath,
    *,
    event: Event,
    instance: CEInstance | None,
    catalog: Mapping[str, Mapping[str, Any]],
    expires_at: datetime | None = None,
    required_duration_ms: int | None = None,
    sustain_path: PatternPath | None = None,
) -> FrontierItem:
    specs = catalog[expr.op]["arguments"]
    arguments: dict[str, str | None] = {}
    classes: dict[str, str] = {}
    role_refs: dict[str, str] = {}
    parameters: dict[str, Any] = {}

    for arg_name, authored_value in expr.args.items():
        spec = specs[arg_name]
        if spec["type"] == "visual_object":
            role = str(authored_value)
            role_refs[arg_name] = role
            arguments[arg_name] = None if instance is None else instance.bindings.get(role)
            classes[arg_name] = event.roles[role]
        else:
            parameters[arg_name] = authored_value

    return FrontierItem(
        path=path,
        predicate=expr.op,
        arguments=arguments,
        classes=classes,
        role_refs=role_refs,
        parameters=parameters,
        expires_at=expires_at,
        required_duration_ms=required_duration_ms,
        sustain_path=sustain_path,
    )


def _matched_bindings(
    item: FrontierItem,
    match: PredicateMatch,
    instance: CEInstance,
    *,
    catalog: Mapping[str, Mapping[str, Any]],
) -> dict[str, str] | None:
    if match.predicate != item.predicate:
        return None
    if item.expires_at is not None and match.event_time > item.expires_at:
        return None

    bindings = dict(instance.bindings)
    for arg_name, role in item.role_refs.items():
        actual = match.arguments.get(arg_name)
        if not isinstance(actual, str) or not actual:
            return None
        observed_class = match.classes.get(arg_name)
        expected_class = item.classes.get(arg_name)
        if observed_class is not None and expected_class is not None:
            capabilities = load_provider_capabilities()
            if not visual_class_matches(capabilities, expected_class, observed_class):
                return None
        expected = item.arguments.get(arg_name)
        if expected is not None:
            if actual != expected:
                return None
        else:
            existing = bindings.get(role)
            if existing is not None:
                if existing != actual:
                    return None
            else:
                # Different role names denote distinct logical identities.
                if actual in bindings.values():
                    return None
                bindings[role] = actual

    # Literal/configuration arguments need not be echoed by every provider; if
    # they are echoed, they must agree with the authored value.
    for arg_name, expected in item.parameters.items():
        if arg_name in match.arguments and match.arguments[arg_name] != expected:
            return None

    minimum_confidence = item.parameters.get("minimum_confidence")
    if minimum_confidence is not None and match.confidence < float(minimum_confidence):
        return None
    return bindings


def _record_match(instance: CEInstance, item: FrontierItem, event_time: datetime) -> None:
    if item.sustain_path is None:
        instance.satisfied_at[item.path] = event_time
        return

    assert item.required_duration_ms is not None
    start = instance.sustain_started_at.get(item.sustain_path)
    if start is None:
        instance.sustain_started_at[item.sustain_path] = event_time
        instance.sustain_last_match_at[item.sustain_path] = event_time
        return
    if event_time < start:
        return

    instance.sustain_last_match_at[item.sustain_path] = event_time
    if event_time - start >= _ms(item.required_duration_ms):
        instance.satisfied_at[item.sustain_path] = event_time


def _with_expiry(
    items: list[FrontierItem],
    expires_at: datetime | None,
) -> list[FrontierItem]:
    if expires_at is None:
        return items
    return [
        replace(item, expires_at=_earliest(item.expires_at, expires_at))
        for item in items
    ]


def _fail(
    instance: CEInstance,
    path: PatternPath,
    reason: str,
) -> tuple[str, None, list[FrontierItem]]:
    instance.failed_paths.setdefault(path, reason)
    return "failed", None, []


def _matched_source(match: PredicateMatch) -> str | None:
    if not match.source_ids:
        return None
    if len(match.source_ids) == 1:
        return match.source_ids[0]
    return ",".join(match.source_ids)


def _earliest(a: datetime | None, b: datetime | None) -> datetime | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _ms(milliseconds: int | None) -> timedelta:
    if milliseconds is None:
        raise ValueError("missing required duration in AST")
    return timedelta(milliseconds=milliseconds)


def _require_aware(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
