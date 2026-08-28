"""Request front end for authored FABLE event families.

The deterministic compiler accepts a typed request or a small set of exact
natural-language aliases.  An optional interpreter may translate broader text
to the same typed request, but it cannot return an executable graph directly.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Protocol

from pydantic import Field

from fable.common.base import FableModel, JSONValue
from fable.common.schemas import SemanticGraph

from .definitions import (
    alarm_departure_graph,
    drive_up_shooting_graph,
    multimodal_robbery_graph,
    package_exchange_graph,
    repeated_visit_graph,
    sequential_vehicle_pass_graph,
    talking_rendezvous_graph,
    two_vehicle_chase_graph,
    uncalibrated_repeated_pass_graph,
    vehicle_convergence_graph,
)
from .definitions.registry import PRODUCTION_DEFINITIONS


class RequestCompileError(ValueError):
    """Raised when text cannot be grounded to an authored event family."""


class RequestCompilationMode(StrEnum):
    STRUCTURED = "STRUCTURED"
    NATURAL_LANGUAGE_ALIAS = "NATURAL_LANGUAGE_ALIAS"
    LLM_ASSISTED = "LLM_ASSISTED"


class StructuredEventRequest(FableModel):
    family_id: str = Field(min_length=1)
    parameters: dict[str, JSONValue] = Field(default_factory=dict)


class InterpretedEventRequest(StructuredEventRequest):
    rationale: str = ""
    inferred_fields: tuple[str, ...] = ()


class NaturalLanguageRequestInterpreter(Protocol):
    def interpret(
        self,
        text: str,
        *,
        available_family_ids: tuple[str, ...],
    ) -> InterpretedEventRequest:
        """Return a typed request, never a SemanticGraph or provider plan."""


class RequestCompilationResult(FableModel):
    graph: SemanticGraph
    family_id: str = Field(min_length=1)
    mode: RequestCompilationMode
    original_request: str | None = None
    grounded_parameters: dict[str, JSONValue] = Field(default_factory=dict)
    inferred_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


GraphFactory = Callable[[Mapping[str, JSONValue]], SemanticGraph]


class AuthoredEventFamilyRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, GraphFactory] = {}
        self._aliases: dict[str, str] = {}
        self._warnings: dict[str, str] = {}

    def register(
        self,
        family_id: str,
        factory: GraphFactory,
        *,
        aliases: tuple[str, ...] = (),
        warning: str = "",
    ) -> None:
        normalized_id = _normalize_phrase(family_id)
        if normalized_id in self._factories:
            raise RequestCompileError(f"duplicate event family: {family_id}")
        self._factories[normalized_id] = factory
        self._aliases[normalized_id] = normalized_id
        for alias in aliases:
            normalized_alias = _normalize_phrase(alias)
            previous = self._aliases.get(normalized_alias)
            if previous is not None and previous != normalized_id:
                raise RequestCompileError(f"ambiguous event-family alias: {alias}")
            self._aliases[normalized_alias] = normalized_id
        if warning:
            self._warnings[normalized_id] = warning

    @property
    def family_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def resolve(self, value: str) -> str | None:
        return self._aliases.get(_normalize_phrase(value))

    def build(self, request: StructuredEventRequest) -> SemanticGraph:
        family_id = self.resolve(request.family_id)
        if family_id is None:
            raise RequestCompileError(
                f"unknown event family {request.family_id!r}; available families: "
                f"{', '.join(self.family_ids)}"
            )
        return self._factories[family_id](request.parameters)

    def warning(self, family_id: str) -> str | None:
        resolved = self.resolve(family_id)
        return self._warnings.get(resolved or "")


class EventRequestCompiler:
    def __init__(
        self,
        *,
        registry: AuthoredEventFamilyRegistry | None = None,
        interpreter: NaturalLanguageRequestInterpreter | None = None,
    ) -> None:
        self.registry = registry or default_event_family_registry()
        self.interpreter = interpreter

    def compile(
        self,
        request: str | StructuredEventRequest | Mapping[str, object],
    ) -> RequestCompilationResult:
        if isinstance(request, Mapping):
            structured = StructuredEventRequest.model_validate(request)
            return self._compile_structured(
                structured,
                mode=RequestCompilationMode.STRUCTURED,
            )
        if isinstance(request, StructuredEventRequest):
            return self._compile_structured(
                request,
                mode=RequestCompilationMode.STRUCTURED,
            )
        text = request.strip()
        if not text:
            raise RequestCompileError("request text cannot be empty")
        family_id = self.registry.resolve(text)
        if family_id is not None:
            return self._compile_structured(
                StructuredEventRequest(family_id=family_id),
                mode=RequestCompilationMode.NATURAL_LANGUAGE_ALIAS,
                original_request=text,
            )
        if self.interpreter is None:
            raise RequestCompileError(
                "free-form request is not an authored alias. Use a structured request "
                "such as {'family_id': 'convoy', 'parameters': {...}} or configure "
                "a typed natural-language interpreter."
            )
        interpreted = self.interpreter.interpret(
            text,
            available_family_ids=self.registry.family_ids,
        )
        family_id = self.registry.resolve(interpreted.family_id)
        if family_id is None:
            raise RequestCompileError(
                "interpreter selected an unknown authored event family: "
                f"{interpreted.family_id}"
            )
        return self._compile_structured(
            StructuredEventRequest(
                family_id=family_id,
                parameters=interpreted.parameters,
            ),
            mode=RequestCompilationMode.LLM_ASSISTED,
            original_request=text,
            inferred_fields=interpreted.inferred_fields,
            extra_warning=interpreted.rationale,
        )

    def _compile_structured(
        self,
        request: StructuredEventRequest,
        *,
        mode: RequestCompilationMode,
        original_request: str | None = None,
        inferred_fields: tuple[str, ...] = (),
        extra_warning: str = "",
    ) -> RequestCompilationResult:
        family_id = self.registry.resolve(request.family_id)
        if family_id is None:
            raise RequestCompileError(
                f"unknown event family {request.family_id!r}; available families: "
                f"{', '.join(self.registry.family_ids)}"
            )
        graph = self.registry.build(
            StructuredEventRequest(
                family_id=family_id,
                parameters=request.parameters,
            )
        )
        warnings = [value for value in (self.registry.warning(family_id), extra_warning) if value]
        return RequestCompilationResult(
            graph=graph,
            family_id=family_id,
            mode=mode,
            original_request=original_request,
            grounded_parameters=request.parameters,
            inferred_fields=inferred_fields,
            warnings=tuple(warnings),
        )


def default_event_family_registry() -> AuthoredEventFamilyRegistry:
    """Build the request registry from the canonical definition registry.

    This is the only selection path used by :class:`EventRequestCompiler`:
    normalize an exact family/alias, validate its family-specific parameters,
    then invoke the canonical factory recorded in ``definitions.registry``.
    """

    registry = AuthoredEventFamilyRegistry()
    adapters: dict[str, GraphFactory] = {
        "convoy": _convoy_factory,
        "robbery": _robbery_factory,
        "package_exchange": lambda parameters: _no_parameters(
            "package_exchange", parameters, package_exchange_graph
        ),
        "drive_up_shooting": _drive_up_shooting_factory,
        "repeated_visit": _repeated_visit_factory,
        "rendezvous": _rendezvous_factory,
        "vehicle_convergence": _vehicle_convergence_factory,
        "two_vehicle_chase": _two_vehicle_chase_factory,
    }
    for definition in PRODUCTION_DEFINITIONS:
        registry.register(
            definition.family_id,
            adapters[definition.family_id],
            aliases=definition.aliases,
            warning=definition.warning,
        )
    return registry


def _convoy_factory(parameters: Mapping[str, JSONValue]) -> SemanticGraph:
    supported = {"variant", "evaluation_profile", "maximum_gap_ms"}
    unknown = set(parameters) - supported
    if unknown:
        raise RequestCompileError(
            f"the current convoy family does not yet ground parameters {sorted(unknown)}"
        )
    variant = str(parameters.get("variant", "pass_follow_clear"))
    profile = str(parameters.get("evaluation_profile", "sequential_passes"))
    if variant != "pass_follow_clear" or profile != "sequential_passes":
        raise RequestCompileError(
            "convoy supports variant=pass_follow_clear and "
            "evaluation_profile=sequential_passes"
        )
    return sequential_vehicle_pass_graph(
        name="Pass-follow-clear convoy",
        namespace_suffix="pass_follow_clear_convoy",
        maximum_gap_ms=int(parameters.get("maximum_gap_ms", 60_000)),
    )


def _robbery_factory(parameters: Mapping[str, JSONValue]) -> SemanticGraph:
    profile = str(parameters.get("evaluation_profile", "cross_sensor"))
    supported = {
        "evaluation_profile",
        "lookback_ms",
        "alarm_confirmation_ms",
        "gunshot_confirmation_ms",
    }
    _reject_unknown("robbery", parameters, supported)
    if profile == "alarm_departure":
        return alarm_departure_graph(
            alarm_confirmation_ms=int(parameters.get("alarm_confirmation_ms", 10_000))
        )
    if profile not in {"cross_sensor", "default"}:
        raise RequestCompileError(f"unsupported robbery evaluation_profile: {profile}")
    return multimodal_robbery_graph(
        lookback_ms=int(parameters.get("lookback_ms", 120_000)),
        alarm_confirmation_ms=int(parameters.get("alarm_confirmation_ms", 45_000)),
        gunshot_confirmation_ms=int(parameters.get("gunshot_confirmation_ms", 15_000)),
    )


def _drive_up_shooting_factory(
    parameters: Mapping[str, JSONValue],
) -> SemanticGraph:
    _reject_unknown(
        "drive_up_shooting", parameters, {"lookback_ms", "require_boarding"}
    )
    require_boarding = parameters.get("require_boarding", True)
    if not isinstance(require_boarding, bool):
        raise RequestCompileError("require_boarding must be a boolean")
    return drive_up_shooting_graph(
        lookback_ms=int(parameters.get("lookback_ms", 15_000)),
        require_boarding=require_boarding,
    )


def _repeated_visit_factory(parameters: Mapping[str, JSONValue]) -> SemanticGraph:
    supported = {
        "return_window_ms",
        "minimum_return_gap_ms",
        "visit_count",
        "evaluation_profile",
        "identity_confirmation",
    }
    _reject_unknown("repeated_visit", parameters, supported)
    common = {
        "return_window_ms": int(parameters.get("return_window_ms", 300_000)),
        "minimum_return_gap_ms": int(parameters.get("minimum_return_gap_ms", 30_000)),
        "visit_count": int(parameters.get("visit_count", 2)),
    }
    if str(parameters.get("evaluation_profile", "calibrated")) == "uncalibrated_passes":
        return uncalibrated_repeated_pass_graph(
            **common,
            identity_confirmation=bool(parameters.get("identity_confirmation", False)),
        )
    return repeated_visit_graph(**common)


def _rendezvous_factory(parameters: Mapping[str, JSONValue]) -> SemanticGraph:
    _reject_unknown("rendezvous", parameters, {"evaluation_profile", "interaction"})
    interaction = str(parameters.get("interaction", "either"))
    if interaction not in {"either", "conversation", "transfer"}:
        raise RequestCompileError(f"unsupported rendezvous interaction: {interaction}")
    profile = str(parameters.get("evaluation_profile", "full_talking"))
    if profile != "full_talking":
        raise RequestCompileError(f"unsupported rendezvous evaluation_profile: {profile}")
    return talking_rendezvous_graph()


def _vehicle_convergence_factory(parameters: Mapping[str, JSONValue]) -> SemanticGraph:
    _reject_unknown("vehicle_convergence", parameters, {"departure_policy"})
    return vehicle_convergence_graph(
        departure_policy=str(parameters.get("departure_policy", "identity_bound"))
    )


def _two_vehicle_chase_factory(parameters: Mapping[str, JSONValue]) -> SemanticGraph:
    _reject_unknown(
        "two_vehicle_chase", parameters, {"evaluation_profile", "maximum_gap_ms"}
    )
    profile = str(parameters.get("evaluation_profile", "follows"))
    if profile == "sequential_passes":
        return sequential_vehicle_pass_graph(
            name="Two-vehicle chase",
            namespace_suffix="two_vehicle_chase_sequential",
            maximum_gap_ms=int(parameters.get("maximum_gap_ms", 60_000)),
        )
    if profile != "follows":
        raise RequestCompileError(f"unsupported two_vehicle_chase profile: {profile}")
    return two_vehicle_chase_graph()


def _reject_unknown(
    family_id: str,
    parameters: Mapping[str, JSONValue],
    supported: set[str],
) -> None:
    unknown = set(parameters) - supported
    if unknown:
        raise RequestCompileError(
            f"event family {family_id} does not expose parameters {sorted(unknown)}"
        )


def _no_parameters(
    family_id: str,
    parameters: Mapping[str, JSONValue],
    factory: Callable[[], SemanticGraph],
) -> SemanticGraph:
    if parameters:
        raise RequestCompileError(
            f"event family {family_id} does not currently expose parameters: {sorted(parameters)}"
        )
    return factory()


def _normalize_phrase(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()
    tokens = text.split()
    while tokens and tokens[0] in {"please", "can", "you"}:
        tokens.pop(0)
    if tokens and tokens[0] in {"detect", "monitor", "identify", "find", "watch"}:
        tokens.pop(0)
        if tokens and tokens[0] == "for":
            tokens.pop(0)
    if tokens and tokens[0] in {"a", "an", "the"}:
        tokens.pop(0)
    return "_".join(tokens)
