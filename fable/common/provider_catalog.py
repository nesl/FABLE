"""Adapter from the existing provider YAML catalog to Phase-0 contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .enums import ArtifactAccessMode, BindingCapability, ExecutionMode, ProviderPortKind
from .schemas import (
    CompatibilityGroup,
    ParameterSpec,
    ProviderContract,
    ProviderExecutionCapabilities,
    ProviderEvaluationContract,
    ProviderPort,
    ProviderRoleCapability,
    ProviderSemanticCapabilities,
)


_MODE_MAP = {
    "live": ExecutionMode.LIVE,
    "retrospective": ExecutionMode.RETROSPECTIVE,
}
_ACCESS_MAP = {
    "local": ArtifactAccessMode.LOCAL,
    "remote_reference": ArtifactAccessMode.REMOTE_REFERENCE,
    "transferred": ArtifactAccessMode.TRANSFERRED,
    "inline": ArtifactAccessMode.INLINE,
}

_BINDING_CAPABILITY_MAP = {
    "consume": BindingCapability.CONSUME,
    "introduce": BindingCapability.INTRODUCE,
    "validate": BindingCapability.VALIDATE,
    "introduce_or_validate": BindingCapability.INTRODUCE_OR_VALIDATE,
    "observe_only": BindingCapability.OBSERVE_ONLY,
    "aggregate": BindingCapability.AGGREGATE,
}

_PORT_SECTIONS = {
    "inputs": ProviderPortKind.INPUT,
    "state_inputs": ProviderPortKind.STATE_INPUT,
    "outputs": ProviderPortKind.OUTPUT,
    "state_outputs": ProviderPortKind.STATE_OUTPUT,
}


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping at root of {path}")
    return data


def provider_contract_from_catalog_entry(provider_id: str, raw: Mapping[str, Any]) -> ProviderContract:
    implements = raw.get("implements", {})
    predicate_ids = tuple(implements.get("predicates", ()))
    role_capabilities = tuple(
        ProviderRoleCapability(
            role_name=role_name,
            capabilities=tuple(
                _BINDING_CAPABILITY_MAP[str(value).lower()]
                for value in values
            ),
        )
        for role_name, values in (implements.get("role_capabilities", {}) or {}).items()
    )

    ports: list[ProviderPort] = []
    for section, kind in _PORT_SECTIONS.items():
        for port in raw.get("ports", {}).get(section, ()) or ():
            ports.append(
                ProviderPort(
                    name=port["name"],
                    kind=kind,
                    data_type=port["type"],
                    required=bool(port.get("required", kind in (ProviderPortKind.INPUT, ProviderPortKind.STATE_INPUT))),
                    purpose=port.get("purpose"),
                )
            )

    parameters = {
        name: ParameterSpec(
            type=spec["type"],
            required=bool(spec.get("required", False)),
            minimum=spec.get("minimum"),
            maximum=spec.get("maximum"),
            enum=tuple(spec.get("enum", ())),
            default=spec.get("default"),
        )
        for name, spec in (raw.get("parameters", {}) or {}).items()
    }

    execution = raw.get("execution_capabilities", {})
    capabilities = ProviderExecutionCapabilities(
        modes=tuple(_MODE_MAP[value] for value in execution.get("modes", ())),
        supports_shared_execution=bool(execution.get("supports_shared_execution", False)),
        accepted_input_access=tuple(
            _ACCESS_MAP[value] for value in execution.get("accepted_input_access", ())
        ),
        state_operations=tuple(execution.get("state_operations", ())),
    )

    compatibility_groups = tuple(
        CompatibilityGroup(
            ports=tuple(group.get("ports", ())),
            require_same_runtime_keys=tuple(group.get("require_same_runtime_keys", ())),
        )
        for group in raw.get("compatibility_groups", ()) or ()
    )

    return ProviderContract(
        provider_id=provider_id,
        description=raw["description"],
        semantic_capabilities=ProviderSemanticCapabilities(
            predicate_ids=predicate_ids,
            role_capabilities=role_capabilities,
            result_kinds=(),
        ),
        ports=tuple(ports),
        parameters=parameters,
        execution_capabilities=capabilities,
        compatibility_groups=compatibility_groups,
        eligible_node_classes=tuple(raw.get("eligible_node_classes", ())),
        required_node_capabilities=tuple(raw.get("required_node_capabilities", ())),
        evaluation_contract=ProviderEvaluationContract.model_validate(
            raw.get("evaluation_contract", {}) or {}
        ),
    )


def load_provider_contracts(path: str | Path) -> dict[str, ProviderContract]:
    document = load_yaml_mapping(path)
    providers = document.get("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("provider catalog 'providers' field must be a mapping")
    return {
        provider_id: provider_contract_from_catalog_entry(provider_id, raw)
        for provider_id, raw in providers.items()
    }
