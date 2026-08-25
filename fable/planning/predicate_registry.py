"""Logical predicate schemas and deterministic validation."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path

import yaml

from fable.common.enums import ResultKind
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



def load_predicate_registry(path: str | Path) -> PredicateSchemaRegistry:
    """Load an authored logical-predicate vocabulary from YAML or JSON."""

    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() in {".yaml", ".yml"}:
        document = yaml.safe_load(text)
    else:
        document = json.loads(text)
    rows = document if isinstance(document, list) else (document or {}).get("predicates", [])
    return PredicateSchemaRegistry(
        LogicalPredicateSchema.model_validate(row) for row in rows
    )


def default_predicate_registry() -> PredicateSchemaRegistry:
    """Load the packaged authored semantic vocabulary."""

    return load_predicate_registry(
        Path(__file__).resolve().parents[1] / "catalog" / "default_predicates.yaml"
    )
