"""Provider, chain, data-type, and runtime-profile registry for physical planning."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from fable.common.enums import ProviderPortKind
from fable.common.provider_catalog import (
    load_yaml_mapping,
    provider_contract_from_catalog_entry,
)
from fable.common.schemas import ArtifactRef, PredicateDemand, ProviderContract, ProviderPort
from fable.catalog.chain_validator import ChainValidator

from .models import (
    ChainExternalInput,
    ChainStep,
    DataTypeDefinition,
    ProviderChainContract,
    ProviderResourceProfile,
)


class ProviderRegistryError(ValueError):
    """Raised for invalid provider, chain, or data-type metadata."""


class ProviderRegistry:
    def __init__(
        self,
        *,
        data_types: Mapping[str, DataTypeDefinition],
        providers: Mapping[str, ProviderContract],
        chains: Mapping[str, ProviderChainContract],
        profiles: Iterable[ProviderResourceProfile] = (),
    ) -> None:
        self.data_types = dict(data_types)
        self.providers = dict(providers)
        self.chains = dict(chains)
        self._profiles: dict[tuple[str, str], ProviderResourceProfile] = {
            (profile.provider_id, profile.node_class): profile for profile in profiles
        }
        self._validate_references()

    @classmethod
    def from_files(
        cls,
        *,
        catalog_path: str | Path,
        data_types_path: str | Path,
        profiles: Iterable[ProviderResourceProfile] | None = None,
        profiles_path: str | Path | None = None,
    ) -> "ProviderRegistry":
        catalog_document = load_yaml_mapping(catalog_path)
        data_types_document = load_yaml_mapping(data_types_path)

        report = ChainValidator(data_types_document, catalog_document).validate_all()
        if not report.ok:
            report.raise_for_errors()

        raw_types = data_types_document.get("data_types", {})
        data_types = {
            data_type: _data_type_from_raw(data_type, raw)
            for data_type, raw in raw_types.items()
        }
        raw_providers = catalog_document.get("providers", {})
        providers = {
            provider_id: provider_contract_from_catalog_entry(provider_id, raw)
            for provider_id, raw in raw_providers.items()
        }
        chains = {
            chain_id: _chain_from_raw(
                chain_id,
                raw,
                providers=providers,
                data_types=data_types,
            )
            for chain_id, raw in catalog_document.get("chains", {}).items()
        }
        if profiles is not None and profiles_path is not None:
            raise ProviderRegistryError("provide profiles or profiles_path, not both")
        resolved_profiles: Iterable[ProviderResourceProfile]
        if profiles_path is not None:
            from fable.catalog.profiles import load_profile_records

            resolved_profiles = tuple(
                record.to_planner_profile()
                for record in load_profile_records(profiles_path)
            )
        else:
            resolved_profiles = profiles or default_provider_profiles()
        return cls(
            data_types=data_types,
            providers=providers,
            chains=chains,
            profiles=resolved_profiles,
        )

    def _validate_references(self) -> None:
        for provider in self.providers.values():
            for port in provider.ports:
                if port.data_type not in self.data_types:
                    raise ProviderRegistryError(
                        f"provider {provider.provider_id} references unknown data type {port.data_type}"
                    )
        for chain in self.chains.values():
            for step in chain.steps:
                if step.provider_id not in self.providers:
                    raise ProviderRegistryError(
                        f"chain {chain.chain_id} references unknown provider {step.provider_id}"
                    )
            for external in chain.external_inputs:
                if external.data_type not in self.data_types:
                    raise ProviderRegistryError(
                        f"chain {chain.chain_id} references unknown external type {external.data_type}"
                    )
            for data_type in chain.output_types.values():
                if data_type not in self.data_types:
                    raise ProviderRegistryError(
                        f"chain {chain.chain_id} exposes unknown output type {data_type}"
                    )

    def provider(self, provider_id: str) -> ProviderContract:
        try:
            return self.providers[provider_id]
        except KeyError as exc:
            raise ProviderRegistryError(f"unknown provider: {provider_id}") from exc

    def chain(self, chain_id: str) -> ProviderChainContract:
        try:
            return self.chains[chain_id]
        except KeyError as exc:
            raise ProviderRegistryError(f"unknown provider chain: {chain_id}") from exc

    def data_type(self, data_type: str) -> DataTypeDefinition:
        try:
            return self.data_types[data_type]
        except KeyError as exc:
            raise ProviderRegistryError(f"unknown data type: {data_type}") from exc

    def profile(self, provider_id: str, node_class: str) -> ProviderResourceProfile:
        exact = self._profiles.get((provider_id, node_class))
        if exact is not None:
            return exact.model_copy(update={"node_class": node_class})
        generic = self._profiles.get((provider_id, "*"))
        if generic is not None:
            return generic.model_copy(update={"node_class": node_class})
        raise ProviderRegistryError(
            f"provider {provider_id} has no measured/default profile for node class {node_class}"
        )

    def candidate_chains(self, demand: PredicateDemand) -> tuple[ProviderChainContract, ...]:
        predicate_id = demand.semantic_predicate.predicate_id
        acceptable = set(demand.acceptable_output_types)
        candidates = [
            chain
            for chain in self.chains.values()
            if predicate_id in chain.predicate_ids
            and bool(acceptable & set(chain.output_types.values()))
        ]
        return tuple(sorted(candidates, key=lambda chain: chain.chain_id))

    def provider_input_ports(self, provider_id: str) -> dict[str, ProviderPort]:
        return {
            port.name: port
            for port in self.provider(provider_id).ports
            if port.kind in (ProviderPortKind.INPUT, ProviderPortKind.STATE_INPUT)
        }

    def provider_output_ports(self, provider_id: str) -> dict[str, ProviderPort]:
        return {
            port.name: port
            for port in self.provider(provider_id).ports
            if port.kind in (ProviderPortKind.OUTPUT, ProviderPortKind.STATE_OUTPUT)
        }

    def validate_runtime_input_compatibility(
        self,
        provider_id: str,
        artifacts_by_port: Mapping[str, ArtifactRef],
    ) -> None:
        """Validate typed artifacts against one provider invocation.

        Static chain wiring is checked by ``ChainValidator``.  This method checks
        runtime schema and model/feature-space compatibility for concrete
        artifacts selected by the planner.
        """

        provider = self.provider(provider_id)
        input_ports = self.provider_input_ports(provider_id)
        for port_name, artifact in artifacts_by_port.items():
            if port_name not in input_ports:
                raise ProviderRegistryError(
                    f"provider {provider_id} has no input port {port_name}"
                )
            expected_type = input_ports[port_name].data_type
            if artifact.artifact_type != expected_type:
                raise ProviderRegistryError(
                    f"provider {provider_id}.{port_name} expects {expected_type}, "
                    f"got {artifact.artifact_type}"
                )
            if artifact.artifact_schema_version != expected_type:
                raise ProviderRegistryError(
                    f"artifact schema {artifact.artifact_schema_version} is incompatible with {expected_type}"
                )
            producer_contract = self.providers.get(artifact.producer.provider_id)
            if (
                producer_contract is not None
                and artifact.producer.provider_contract_version
                != producer_contract.contract_version
            ):
                raise ProviderRegistryError(
                    f"artifact was produced by contract version "
                    f"{artifact.producer.provider_contract_version}, but registry has "
                    f"{producer_contract.contract_version}"
                )

        for group in provider.compatibility_groups:
            selected = [
                artifacts_by_port[port_name]
                for port_name in group.ports
                if port_name in artifacts_by_port
            ]
            if len(selected) < 2:
                continue
            for key in group.require_same_runtime_keys:
                values = [artifact.compatibility_keys.get(key) for artifact in selected]
                if any(value is None for value in values):
                    raise ProviderRegistryError(
                        f"provider {provider_id} requires compatibility key {key} on ports {group.ports}"
                    )
                if len({repr(value) for value in values}) != 1:
                    raise ProviderRegistryError(
                        f"provider {provider_id} requires equal {key} across ports {group.ports}"
                    )

    def provider_families_for_predicate(self, predicate_id: str) -> tuple[str, ...]:
        families: set[str] = set()
        for chain in self.chains.values():
            if predicate_id not in chain.predicate_ids:
                continue
            if "cross" in chain.chain_id:
                families.add(f"{predicate_id.lower()}_cross_sensor")
            elif "local" in chain.chain_id:
                families.add(f"{predicate_id.lower()}_local")
            else:
                families.add(predicate_id.lower())
        return tuple(sorted(families))


def _data_type_from_raw(data_type: str, raw: Mapping[str, Any]) -> DataTypeDefinition:
    transport = raw.get("transport", {}) or {}
    continuation = raw.get("continuation", {}) or {}
    transferable = transport.get("transferable", True)
    if transferable not in (True, False, "policy_dependent"):
        raise ProviderRegistryError(
            f"data type {data_type} has invalid transferable value {transferable!r}"
        )
    return DataTypeDefinition(
        data_type=data_type,
        description=str(raw.get("description", "")),
        kind=str(raw.get("kind", "unknown")),
        modality=raw.get("modality"),
        compatibility_keys=tuple(raw.get("compatibility_keys", ()) or ()),
        transferable=transferable,
        remote_reference_allowed=bool(transport.get("remote_reference_allowed", False)),
        inline_allowed=bool(transport.get("inline_allowed", False)),
        continuation_eligible=bool(continuation.get("eligible", False)),
        continuation_category=continuation.get("category"),
    )


def _chain_from_raw(
    chain_id: str,
    raw: Mapping[str, Any],
    *,
    providers: Mapping[str, ProviderContract],
    data_types: Mapping[str, DataTypeDefinition],
) -> ProviderChainContract:
    external_inputs = tuple(
        ChainExternalInput(
            name=name,
            data_type=spec["type"] if isinstance(spec, Mapping) else str(spec),
            optional=bool(spec.get("optional", False)) if isinstance(spec, Mapping) else False,
        )
        for name, spec in (raw.get("external_inputs", {}) or {}).items()
    )
    steps = tuple(
        ChainStep(
            step_id=step["id"],
            provider_id=step["provider"],
            bindings=dict(step.get("bind", {}) or {}),
        )
        for step in raw.get("steps", ())
    )
    outputs = dict(raw.get("outputs", {}) or {})

    provider_outputs: dict[str, dict[str, str]] = {}
    predicate_ids: set[str] = set()
    for step in steps:
        provider = providers[step.provider_id]
        predicate_ids.update(provider.semantic_capabilities.predicate_ids)
        provider_outputs[step.step_id] = {
            port.name: port.data_type
            for port in provider.ports
            if port.kind in (ProviderPortKind.OUTPUT, ProviderPortKind.STATE_OUTPUT)
        }

    output_types: dict[str, str] = {}
    for output_name, reference in outputs.items():
        step_id, port_name = reference.split(".", 1)
        try:
            output_types[output_name] = provider_outputs[step_id][port_name]
        except KeyError as exc:
            raise ProviderRegistryError(
                f"chain {chain_id} output {output_name} has unresolved reference {reference}"
            ) from exc

    continuation_types = tuple(
        sorted(
            {
                output_type
                for output_name, output_type in output_types.items()
                if output_name != "result" and data_types[output_type].continuation_eligible
            }
        )
    )
    return ProviderChainContract(
        chain_id=chain_id,
        description=str(raw.get("description", "")),
        predicate_ids=tuple(sorted(predicate_ids)),
        external_inputs=external_inputs,
        steps=steps,
        outputs=outputs,
        output_types=output_types,
        continuation_output_types=continuation_types,
        capability_tags=tuple(sorted(set(raw.get("capabilities", ()) or ()))),
    )



def default_provider_profiles() -> tuple[ProviderResourceProfile, ...]:
    """Load deterministic fallback profiles from the packaged catalog data."""

    from fable.catalog.profiles import default_planner_profiles

    return default_planner_profiles()
