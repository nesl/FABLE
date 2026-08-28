"""Canonical registry of production complex-event families.

This is the one mapping from a public ``family_id`` to its definition factory.
Request parsing may supply parameters, but it cannot select executable code
outside this registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Mapping

from fable.common.base import JSONValue
from fable.common.schemas import SemanticGraph


ParameterizedFactory = Callable[[Mapping[str, JSONValue]], SemanticGraph]


@dataclass(frozen=True)
class ComplexEventDefinition:
    """One registered production CE and its canonical source location."""

    family_id: str
    module: str
    factory_name: str
    aliases: tuple[str, ...] = ()
    warning: str = ""

    @property
    def factory(self) -> Callable[..., SemanticGraph]:
        return getattr(import_module(self.module), self.factory_name)


PRODUCTION_DEFINITIONS: tuple[ComplexEventDefinition, ...] = (
    ComplexEventDefinition(
        "convoy",
        "fable.semantic.definitions.route_convoy",
        "route_convoy_graph",
        ("detect convoy", "detect a convoy", "monitor convoy", "route convoy"),
        "The default convoy is the authored sequential pass-follow-clear variant.",
    ),
    ComplexEventDefinition(
        "robbery",
        "fable.semantic.definitions.robbery_with_alarm",
        "robbery_with_alarm_graph",
        ("detect robbery", "detect a robbery", "robbery with alarm"),
    ),
    ComplexEventDefinition(
        "package_exchange",
        "fable.semantic.definitions.package_exchange",
        "package_exchange_graph",
        ("detect package exchange", "package exchange"),
    ),
    ComplexEventDefinition(
        "drive_up_shooting",
        "fable.semantic.definitions.drive_up_shooting",
        "drive_up_shooting_graph",
        ("detect drive up shooting", "drive-up shooting"),
    ),
    ComplexEventDefinition(
        "repeated_visit",
        "fable.semantic.definitions.repeated_visit",
        "repeated_visit_graph",
        ("detect repeated visit", "detect stalking", "repeated vehicle visit"),
    ),
    ComplexEventDefinition(
        "rendezvous",
        "fable.semantic.definitions.talking_rendezvous",
        "talking_rendezvous_graph",
        ("detect rendezvous", "talking rendezvous"),
    ),
    ComplexEventDefinition(
        "vehicle_convergence",
        "fable.semantic.definitions.vehicle_convergence",
        "vehicle_convergence_graph",
        ("detect vehicle convergence", "vehicle rendezvous"),
    ),
    ComplexEventDefinition(
        "two_vehicle_chase",
        "fable.semantic.definitions.two_vehicle_chase",
        "two_vehicle_chase_graph",
        ("detect two vehicle chase", "two vehicle chase"),
    ),
)

DEFINITION_BY_FAMILY = {item.family_id: item for item in PRODUCTION_DEFINITIONS}


def get_definition(family_id: str) -> ComplexEventDefinition:
    """Return canonical metadata for an exact normalized family identifier."""

    try:
        return DEFINITION_BY_FAMILY[family_id]
    except KeyError as exc:
        raise KeyError(f"unknown production CE family {family_id!r}") from exc


__all__ = [
    "ComplexEventDefinition",
    "DEFINITION_BY_FAMILY",
    "PRODUCTION_DEFINITIONS",
    "get_definition",
]
