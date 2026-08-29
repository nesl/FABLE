"""Compile a parsed CE against the configured provider capability catalog.

Parsing answers: "Is this valid FABLE CE syntax?"
Compilation answers: "Can this FABLE installation execute every semantic leaf?"

The compiler deliberately does not choose a provider or construct an execution
plan.  It only proves that at least one declared implementation exists for each
predicate call and that authored semantic classes/literals are supported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

from .event_parser import Event, load_event
from .pattern_parser import Expr
from .predicates import load_predicates
from ..providers.provider_capabilities import (
    ProviderCapabilityCatalog,
    load_provider_capabilities,
    predicate_providers,
    provider_supports_predicate_call,
    semantic_literal_values,
    supported_visual_classes,
    visual_providers_for_class,
)


class EventCompilationError(ValueError):
    """Raised when a syntactically valid CE cannot be executed by the catalog."""


def compile_event(
    event: Event,
    *,
    provider_capabilities: Mapping[str, Any] | None = None,
) -> Event:
    """Validate provider support for ``event`` and return the same compact AST.

    Returning the :class:`Event` unchanged is intentional: compilation does not
    pin a provider.  Future planning is free to choose among all providers that
    advertised support at compile time.
    """

    capabilities = (
        provider_capabilities
        if provider_capabilities is not None
        else load_provider_capabilities()
    )
    predicate_catalog = load_predicates()
    errors: list[str] = []

    # First validate every declared role class.  This catches unsupported terms
    # such as ``class: dragon`` before any runtime state is created.
    for role_name, semantic_class in sorted(event.roles.items()):
        if not visual_providers_for_class(capabilities, semantic_class):
            supported = supported_visual_classes(capabilities)
            errors.append(
                f"roles.{role_name}.class: unsupported visual class {semantic_class!r}; "
                f"no enabled visual detector advertises it. "
                f"Supported classes include: {_preview(supported)}"
            )

    for expr, path in _walk_with_paths(event.pattern, "pattern"):
        if not expr.is_predicate:
            continue

        implementations = predicate_providers(capabilities, expr.op)
        if not implementations:
            errors.append(
                f"{path}: predicate {expr.op!r} has no enabled implementation provider"
            )
            continue

        predicate_spec = predicate_catalog[expr.op]["arguments"]
        argument_classes: dict[str, str] = {}
        literal_arguments: dict[str, Any] = {}
        for arg_name, value in expr.args.items():
            arg_spec = predicate_spec[arg_name]
            if arg_spec["type"] == "visual_object":
                argument_classes[arg_name] = event.roles[str(value)]
            else:
                literal_arguments[arg_name] = value

        candidates = tuple(
            provider_id
            for provider_id in implementations
            if provider_supports_predicate_call(
                capabilities,
                provider_id,
                expr.op,
                argument_classes=argument_classes,
                literal_arguments=literal_arguments,
            )
        )
        if candidates:
            continue

        # Produce a focused error instead of just saying "no provider".
        specific = _predicate_support_error(
            capabilities,
            expr,
            path=path,
            argument_classes=argument_classes,
            literal_arguments=literal_arguments,
            implementations=implementations,
        )
        errors.append(specific)

    if errors:
        joined = "\n  - ".join(errors)
        raise EventCompilationError(
            f"Complex event {event.name!r} cannot be compiled against the current "
            f"provider capabilities:\n  - {joined}"
        )
    return event


def load_and_compile_event(
    path: str | Path,
    *,
    provider_capabilities: Mapping[str, Any] | None = None,
) -> Event:
    """Convenience function: parse one CE file and immediately compile it."""

    return compile_event(
        load_event(path),
        provider_capabilities=provider_capabilities,
    )


def _predicate_support_error(
    capabilities: Mapping[str, Any],
    expr: Expr,
    *,
    path: str,
    argument_classes: Mapping[str, str],
    literal_arguments: Mapping[str, Any],
    implementations: tuple[str, ...],
) -> str:
    # Semantic literal support (currently most important for audio_event.class).
    for arg_name, value in literal_arguments.items():
        supported_values = semantic_literal_values(capabilities, expr.op, arg_name)
        if supported_values and value not in supported_values:
            return (
                f"{path}.{expr.op}.{arg_name}: unsupported semantic value {value!r}; "
                f"enabled implementations support: {', '.join(supported_values)}"
            )

    # Visual argument class restrictions of specialized predicate providers.
    restrictions: list[str] = []
    for provider_id in implementations:
        predicate_spec = capabilities["providers"][provider_id]["predicates"][expr.op]
        visual_restrictions = predicate_spec["visual_arguments"]
        rendered = []
        for arg_name, semantic_class in argument_classes.items():
            allowed = visual_restrictions.get(arg_name)
            if allowed == "*" or allowed is None:
                rendered.append(f"{arg_name}=*")
            else:
                rendered.append(f"{arg_name}={{{', '.join(allowed)}}}")
        restrictions.append(f"{provider_id}({'; '.join(rendered)})")

    actual = ", ".join(f"{name}={value}" for name, value in argument_classes.items())
    return (
        f"{path}: predicate {expr.op!r} has implementations, but none accepts the "
        f"authored visual classes ({actual}). Provider constraints: "
        f"{'; '.join(restrictions)}"
    )


def _walk_with_paths(expr: Expr, path: str) -> Iterator[tuple[Expr, str]]:
    yield expr, path
    if not expr.children:
        return

    if expr.op in {"seq", "all", "any"}:
        for index, child in enumerate(expr.children):
            yield from _walk_with_paths(child, f"{path}.{expr.op}[{index}]")
    elif expr.op == "k_of_n":
        for index, child in enumerate(expr.children):
            yield from _walk_with_paths(child, f"{path}.k_of_n.patterns[{index}]")
    elif expr.op in {"within", "for"}:
        yield from _walk_with_paths(expr.children[0], f"{path}.{expr.op}.pattern")
    else:  # defensive; predicates do not have children in CE-v1
        for index, child in enumerate(expr.children):
            yield from _walk_with_paths(child, f"{path}.children[{index}]")


def _preview(values: tuple[str, ...], *, limit: int = 18) -> str:
    if len(values) <= limit:
        return ", ".join(values)
    return ", ".join(values[:limit]) + f", ... ({len(values)} total)"
