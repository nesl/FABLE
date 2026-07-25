"""Logical predicate schemas and deterministic validation."""

from __future__ import annotations

from collections.abc import Iterable

from fable.common.enums import BindingCapability, ResultKind
from fable.common.schemas import SemanticPredicate

from .models import (
    LogicalPredicateSchema,
    PredicateExpressionKind,
    PredicateParameterSchema,
    PredicateRoleSchema,
)


class PredicateSchemaError(ValueError):
    """Raised when a semantic predicate is not covered by its logical schema."""


class PredicateSchemaRegistry:
    def __init__(self, schemas: Iterable[LogicalPredicateSchema] = ()) -> None:
        self._schemas: dict[str, LogicalPredicateSchema] = {}
        for schema in schemas:
            self.register(schema)

    def register(self, schema: LogicalPredicateSchema, *, replace: bool = False) -> None:
        if schema.predicate_id in self._schemas and not replace:
            raise PredicateSchemaError(f"predicate schema already registered: {schema.predicate_id}")
        self._schemas[schema.predicate_id] = schema

    def get(self, predicate_id: str) -> LogicalPredicateSchema:
        try:
            return self._schemas[predicate_id]
        except KeyError as exc:
            raise PredicateSchemaError(f"no logical predicate schema for {predicate_id}") from exc

    def validate(self, predicate: SemanticPredicate) -> LogicalPredicateSchema:
        schema = self.get(predicate.predicate_id)
        if predicate.result_kind != schema.result_kind:
            raise PredicateSchemaError(
                f"{predicate.predicate_id} result kind {predicate.result_kind} does not match "
                f"schema {schema.result_kind}"
            )

        declared = {role.role_name: role for role in schema.roles}
        observed = {role.role_name: role for role in predicate.roles}
        if set(observed) != set(declared):
            raise PredicateSchemaError(
                f"{predicate.predicate_id} roles {sorted(observed)} do not match schema "
                f"{sorted(declared)}"
            )
        for role_name, role in observed.items():
            expected = declared[role_name]
            if role.entity_type != expected.entity_type:
                raise PredicateSchemaError(
                    f"{predicate.predicate_id}.{role_name} expects {expected.entity_type}, "
                    f"got {role.entity_type}"
                )

        unknown_parameters = set(predicate.parameters) - set(schema.parameters)
        if unknown_parameters:
            raise PredicateSchemaError(
                f"{predicate.predicate_id} has unknown parameters {sorted(unknown_parameters)}"
            )
        missing = [
            name
            for name, definition in schema.parameters.items()
            if definition.required and name not in predicate.parameters
        ]
        if missing:
            raise PredicateSchemaError(
                f"{predicate.predicate_id} is missing required parameters {sorted(missing)}"
            )
        for name, value in predicate.parameters.items():
            self._validate_parameter(predicate.predicate_id, name, value, schema.parameters[name])
        return schema

    @staticmethod
    def _validate_parameter(
        predicate_id: str,
        name: str,
        value: object,
        definition: PredicateParameterSchema,
    ) -> None:
        expected = definition.type
        valid = (
            (expected == "string" and isinstance(value, str))
            or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (expected == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
            or (expected == "boolean" and isinstance(value, bool))
        )
        if not valid:
            raise PredicateSchemaError(
                f"{predicate_id}.{name} expects {expected}, got {type(value).__name__}"
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if definition.minimum is not None and value < definition.minimum:
                raise PredicateSchemaError(f"{predicate_id}.{name} is below minimum")
            if definition.maximum is not None and value > definition.maximum:
                raise PredicateSchemaError(f"{predicate_id}.{name} is above maximum")
        if definition.enum and value not in definition.enum:
            raise PredicateSchemaError(f"{predicate_id}.{name} is not in the allowed enum")

    @property
    def predicate_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._schemas))


def _role(
    name: str,
    entity_type: str,
    *capabilities: BindingCapability,
    identity_required: bool = False,
) -> PredicateRoleSchema:
    return PredicateRoleSchema(
        role_name=name,
        entity_type=entity_type,
        binding_capabilities=capabilities,
        identity_required=identity_required,
    )


def default_predicate_registry() -> PredicateSchemaRegistry:
    """Return the minimum authored vocabulary used by Phase 1 examples.

    The provider family identifiers are semantic groupings.  The provider
    registry resolves them to concrete chains; they are not container names.
    """

    schemas = (
        LogicalPredicateSchema(
            predicate_id="PASSES",
            expression_kind=PredicateExpressionKind.TRANSITION,
            result_kind=ResultKind.INSTANT_MATCH,
            roles=(
                _role(
                    "vehicle",
                    "vehicle",
                    BindingCapability.INTRODUCE,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
                _role("reference", "location", BindingCapability.CONSUME),
            ),
            provider_family_ids=("pass_geometry",),
            required_capabilities=("tracking", "reference_geometry"),
        ),
        LogicalPredicateSchema(
            predicate_id="FOLLOWS",
            expression_kind=PredicateExpressionKind.DERIVED_BEHAVIOR,
            result_kind=ResultKind.INTERVAL_MATCH,
            roles=(
                _role(
                    "leader",
                    "vehicle",
                    BindingCapability.CONSUME,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
                _role(
                    "follower",
                    "vehicle",
                    BindingCapability.INTRODUCE,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
            ),
            parameters={
                "max_gap_m": PredicateParameterSchema(type="number", minimum=0),
                "min_duration_ms": PredicateParameterSchema(type="integer", minimum=0),
            },
            provider_family_ids=("follows_local", "follows_cross_sensor"),
            acceptable_output_types=("predicate_match.v1",),
            required_capabilities=("vehicle_identity", "relative_motion"),
            default_continuation_types=("pair_trajectory.v1", "track_summary.v1"),
            forkable_roles=("follower",),
        ),
        LogicalPredicateSchema(
            predicate_id="MOVING",
            expression_kind=PredicateExpressionKind.STATE,
            result_kind=ResultKind.STATE_OBSERVATION,
            roles=(
                _role(
                    "vehicle",
                    "vehicle",
                    BindingCapability.OBSERVE_ONLY,
                    BindingCapability.VALIDATE,
                    identity_required=False,
                ),
            ),
            provider_family_ids=("motion_state",),
            required_capabilities=("tracking",),
        ),
        LogicalPredicateSchema(
            predicate_id="AUDIO_EVENT",
            expression_kind=PredicateExpressionKind.SENSOR_EVENT,
            result_kind=ResultKind.INSTANT_MATCH,
            roles=(
                _role("location", "zone", BindingCapability.CONSUME, BindingCapability.VALIDATE),
            ),
            parameters={"label": PredicateParameterSchema(type="string", required=True)},
            provider_family_ids=("audio_event",),
            required_capabilities=("audio",),
            default_continuation_types=("audio_event_set.v1",),
        ),
        LogicalPredicateSchema(
            predicate_id="VEHICLE_PRESENT_BEFORE",
            expression_kind=PredicateExpressionKind.DERIVED_BEHAVIOR,
            result_kind=ResultKind.INTERVAL_MATCH,
            roles=(
                _role(
                    "vehicle",
                    "vehicle",
                    BindingCapability.INTRODUCE,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
                _role("location", "zone", BindingCapability.CONSUME, BindingCapability.VALIDATE),
            ),
            parameters={"lookback_ms": PredicateParameterSchema(type="integer", minimum=0)},
            provider_family_ids=("historical_vehicle_recovery",),
            acceptable_output_types=("predicate_match.v1",),
            required_capabilities=("tracking", "retrospective"),
            default_continuation_types=("track_summary.v1",),
            forkable_roles=("vehicle",),
        ),
        LogicalPredicateSchema(
            predicate_id="AUDIO_VISUAL_ASSOCIATION",
            expression_kind=PredicateExpressionKind.DERIVED_BEHAVIOR,
            result_kind=ResultKind.INSTANT_MATCH,
            roles=(
                _role("location", "zone", BindingCapability.CONSUME, BindingCapability.VALIDATE),
                _role(
                    "person",
                    "person",
                    BindingCapability.INTRODUCE,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
            ),
            provider_family_ids=("audio_visual_association",),
            acceptable_output_types=("predicate_match.v1",),
            required_capabilities=("audio_localization", "tracking"),
            default_continuation_types=("audio_visual_association_set.v1",),
            forkable_roles=("person",),
        ),
        LogicalPredicateSchema(
            predicate_id="DISEMBARKS",
            expression_kind=PredicateExpressionKind.TRANSITION,
            result_kind=ResultKind.INSTANT_MATCH,
            roles=(
                _role(
                    "person",
                    "person",
                    BindingCapability.INTRODUCE,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
                _role(
                    "vehicle",
                    "vehicle",
                    BindingCapability.CONSUME,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
            ),
            provider_family_ids=("person_vehicle_relation",),
            acceptable_output_types=("predicate_match.v1",),
            required_capabilities=("tracking", "person_vehicle_relation"),
            default_continuation_types=("track_summary.v1",),
            forkable_roles=("person",),
        ),
        LogicalPredicateSchema(
            predicate_id="BOARDS",
            expression_kind=PredicateExpressionKind.TRANSITION,
            result_kind=ResultKind.INSTANT_MATCH,
            roles=(
                _role(
                    "person",
                    "person",
                    BindingCapability.CONSUME,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
                _role(
                    "vehicle",
                    "vehicle",
                    BindingCapability.CONSUME,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
            ),
            provider_family_ids=("person_vehicle_relation",),
            acceptable_output_types=("predicate_match.v1",),
            required_capabilities=("tracking", "person_vehicle_relation"),
            default_continuation_types=("track_summary.v1",),
        ),
        LogicalPredicateSchema(
            predicate_id="CONVERSATION",
            expression_kind=PredicateExpressionKind.DERIVED_BEHAVIOR,
            result_kind=ResultKind.INTERVAL_MATCH,
            roles=(
                _role(
                    "participant_a",
                    "person",
                    BindingCapability.INTRODUCE,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
                _role(
                    "participant_b",
                    "person",
                    BindingCapability.INTRODUCE,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
            ),
            parameters={
                "maximum_distance_m": PredicateParameterSchema(type="number", minimum=0),
                "minimum_duration_s": PredicateParameterSchema(type="number", minimum=0),
                "required_terms": PredicateParameterSchema(type="string", required=False),
            },
            provider_family_ids=("conversation",),
            acceptable_output_types=("predicate_match.v1",),
            required_capabilities=("tracking", "voice_activity", "speaker_diarization"),
            default_continuation_types=("speaker_turn_set.v1", "track_summary.v1"),
            forkable_roles=("participant_a", "participant_b"),
        ),
        LogicalPredicateSchema(
            predicate_id="TRANSFER",
            expression_kind=PredicateExpressionKind.DERIVED_BEHAVIOR,
            result_kind=ResultKind.INTERVAL_MATCH,
            roles=(
                _role(
                    "object",
                    "package",
                    BindingCapability.INTRODUCE,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
                _role(
                    "source",
                    "entity",
                    BindingCapability.CONSUME,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
                _role(
                    "destination",
                    "entity",
                    BindingCapability.INTRODUCE,
                    BindingCapability.VALIDATE,
                    identity_required=True,
                ),
            ),
            provider_family_ids=("package_transfer",),
            acceptable_output_types=("predicate_match.v1",),
            required_capabilities=("package_detection", "custody_reasoning"),
            default_continuation_types=("custody_state.v1",),
            forkable_roles=("object", "destination"),
        ),
        LogicalPredicateSchema(
            predicate_id="SUSPICIOUS_ENTRY",
            expression_kind=PredicateExpressionKind.TRANSITION,
            result_kind=ResultKind.INSTANT_MATCH,
            roles=(
                _role("person", "person", BindingCapability.INTRODUCE, BindingCapability.VALIDATE),
                _role("location", "zone", BindingCapability.CONSUME, BindingCapability.VALIDATE),
            ),
            provider_family_ids=("entry",),
            forkable_roles=("person",),
        ),
        LogicalPredicateSchema(
            predicate_id="THREAT_EVENT",
            expression_kind=PredicateExpressionKind.DERIVED_BEHAVIOR,
            result_kind=ResultKind.INTERVAL_MATCH,
            roles=(
                _role("person", "person", BindingCapability.CONSUME, BindingCapability.VALIDATE),
                _role("location", "zone", BindingCapability.CONSUME, BindingCapability.VALIDATE),
            ),
            provider_family_ids=("threat",),
        ),
        LogicalPredicateSchema(
            predicate_id="FORCED_TRANSFER",
            expression_kind=PredicateExpressionKind.DERIVED_BEHAVIOR,
            result_kind=ResultKind.INTERVAL_MATCH,
            roles=(
                _role("person", "person", BindingCapability.CONSUME, BindingCapability.VALIDATE),
                _role("location", "zone", BindingCapability.CONSUME, BindingCapability.VALIDATE),
            ),
            provider_family_ids=("forced_transfer",),
        ),
        LogicalPredicateSchema(
            predicate_id="FAILED_ATTEMPT_RAPID_EXIT",
            expression_kind=PredicateExpressionKind.DERIVED_BEHAVIOR,
            result_kind=ResultKind.INSTANT_MATCH,
            roles=(
                _role("person", "person", BindingCapability.CONSUME, BindingCapability.VALIDATE),
                _role("location", "zone", BindingCapability.CONSUME, BindingCapability.VALIDATE),
            ),
            provider_family_ids=("failed_attempt",),
        ),
        LogicalPredicateSchema(
            predicate_id="DEPARTURE_OR_ESCAPE",
            expression_kind=PredicateExpressionKind.TRANSITION,
            result_kind=ResultKind.INSTANT_MATCH,
            roles=(
                _role("person", "person", BindingCapability.CONSUME, BindingCapability.VALIDATE),
                _role("vehicle", "vehicle", BindingCapability.INTRODUCE_OR_VALIDATE),
            ),
            provider_family_ids=("departure",),
            forkable_roles=("vehicle",),
        ),
        LogicalPredicateSchema(
            predicate_id="ENTERS",
            expression_kind=PredicateExpressionKind.TRANSITION,
            result_kind=ResultKind.INSTANT_MATCH,
            roles=(
                _role("vehicle", "vehicle", BindingCapability.INTRODUCE_OR_VALIDATE),
            ),
            provider_family_ids=("zone_transition",),
            forkable_roles=("vehicle",),
        ),
        LogicalPredicateSchema(
            predicate_id="EXITS",
            expression_kind=PredicateExpressionKind.TRANSITION,
            result_kind=ResultKind.INSTANT_MATCH,
            roles=(
                _role("vehicle", "vehicle", BindingCapability.CONSUME, BindingCapability.VALIDATE),
            ),
            provider_family_ids=("zone_transition",),
        ),
    )
    return PredicateSchemaRegistry(schemas)
