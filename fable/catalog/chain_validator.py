#!/usr/bin/env python3
"""Validate FABLE provider catalogs and explicitly wired provider chains.

The validator checks static compatibility only:

* referenced data types, providers, ports, steps, and outputs exist;
* every required provider input is bound;
* a binding's produced type is accepted by the destination input port;
* producer steps appear before consumer steps;
* continuation/state ports use registered types;
* data types marked as provider state are used for state ports;
* named compatibility groups reference valid provider input ports.

Runtime checks such as artifact model-version compatibility, current artifact
location, data-movement policy, node capacity, and measured deadline
feasibility belong in the runtime planner and scheduler.
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required. Install it with: python -m pip install PyYAML"
    ) from exc


@dataclass(frozen=True)
class ValidationIssue:
    """One catalog or chain validation issue."""

    code: str
    message: str
    chain_id: Optional[str] = None
    step_id: Optional[str] = None

    def format(self) -> str:
        location: List[str] = []
        if self.chain_id:
            location.append(f"chain={self.chain_id}")
        if self.step_id:
            location.append(f"step={self.step_id}")
        prefix = f"[{', '.join(location)}] " if location else ""
        return f"{self.code}: {prefix}{self.message}"


@dataclass
class ValidationReport:
    """Collection of validation issues."""

    issues: List[ValidationIssue]

    @property
    def ok(self) -> bool:
        return not self.issues

    def extend(self, other: "ValidationReport") -> None:
        self.issues.extend(other.issues)

    def raise_for_errors(self) -> None:
        if self.issues:
            rendered = "\n".join(f"  - {issue.format()}" for issue in self.issues)
            raise ChainValidationError(f"Validation failed:\n{rendered}")

    def format(self) -> str:
        if self.ok:
            return "OK"
        return "\n".join(issue.format() for issue in self.issues)


class ChainValidationError(ValueError):
    """Raised when a chain fails static validation."""


def load_yaml(path: Path | str) -> Dict[str, Any]:
    """Load one YAML document and require a mapping at its root."""

    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ChainValidationError(f"YAML file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ChainValidationError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ChainValidationError(f"Expected a mapping at the root of {path}")
    return loaded


class ChainValidator:
    """Validate provider contracts and named provider chains."""

    PORT_SECTIONS: Tuple[str, ...] = (
        "inputs",
        "state_inputs",
        "outputs",
        "state_outputs",
    )
    INPUT_PORT_SECTIONS: Tuple[str, ...] = ("inputs", "state_inputs")
    OUTPUT_PORT_SECTIONS: Tuple[str, ...] = ("outputs", "state_outputs")

    def __init__(
        self,
        data_types_document: Mapping[str, Any],
        catalog_document: Mapping[str, Any],
    ) -> None:
        self.data_types_document = copy.deepcopy(dict(data_types_document))
        self.catalog_document = copy.deepcopy(dict(catalog_document))
        self.data_types: Dict[str, Dict[str, Any]] = dict(
            self.data_types_document.get("data_types", {})
        )
        self.providers: Dict[str, Dict[str, Any]] = dict(
            self.catalog_document.get("providers", {})
        )
        self.chains: Dict[str, Dict[str, Any]] = dict(
            self.catalog_document.get("chains", {})
        )

    @classmethod
    def from_files(
        cls,
        data_types_path: Path | str,
        catalog_path: Path | str,
    ) -> "ChainValidator":
        return cls(load_yaml(data_types_path), load_yaml(catalog_path))

    def validate_all(self) -> ValidationReport:
        report = self.validate_catalog()
        for chain_id in self.chains:
            report.extend(self.validate_chain(chain_id))
        return report

    def validate_catalog(self) -> ValidationReport:
        issues: List[ValidationIssue] = []

        if not isinstance(self.data_types, dict) or not self.data_types:
            issues.append(
                ValidationIssue(
                    "DATA_TYPES_MISSING",
                    "The data type registry must contain a non-empty 'data_types' mapping.",
                )
            )
            return ValidationReport(issues)

        if not isinstance(self.providers, dict) or not self.providers:
            issues.append(
                ValidationIssue(
                    "PROVIDERS_MISSING",
                    "The catalog must contain a non-empty 'providers' mapping.",
                )
            )
            return ValidationReport(issues)

        for provider_id, provider in self.providers.items():
            if not isinstance(provider, dict):
                issues.append(
                    ValidationIssue(
                        "PROVIDER_NOT_MAPPING",
                        f"Provider '{provider_id}' must be a mapping.",
                    )
                )
                continue

            seen_ports: Dict[str, str] = {}
            ports = provider.get("ports", {})
            if not isinstance(ports, dict):
                issues.append(
                    ValidationIssue(
                        "PORTS_NOT_MAPPING",
                        f"Provider '{provider_id}' has a non-mapping 'ports' field.",
                    )
                )
                continue

            for section in self.PORT_SECTIONS:
                entries = ports.get(section, [])
                if entries is None:
                    entries = []
                if not isinstance(entries, list):
                    issues.append(
                        ValidationIssue(
                            "PORT_SECTION_NOT_LIST",
                            f"Provider '{provider_id}' section '{section}' must be a list.",
                        )
                    )
                    continue

                for index, port in enumerate(entries):
                    if not isinstance(port, dict):
                        issues.append(
                            ValidationIssue(
                                "PORT_NOT_MAPPING",
                                f"Provider '{provider_id}' {section}[{index}] must be a mapping.",
                            )
                        )
                        continue
                    name = port.get("name")
                    type_ref = port.get("type")
                    if not isinstance(name, str) or not name:
                        issues.append(
                            ValidationIssue(
                                "PORT_NAME_MISSING",
                                f"Provider '{provider_id}' {section}[{index}] has no valid name.",
                            )
                        )
                    elif name in seen_ports:
                        issues.append(
                            ValidationIssue(
                                "DUPLICATE_PORT_NAME",
                                f"Provider '{provider_id}' reuses port name '{name}' "
                                f"in '{section}' and '{seen_ports[name]}'.",
                            )
                        )
                    else:
                        seen_ports[name] = section

                    if not isinstance(type_ref, str) or not type_ref:
                        issues.append(
                            ValidationIssue(
                                "PORT_TYPE_MISSING",
                                f"Provider '{provider_id}' port '{name}' has no valid type.",
                            )
                        )
                    elif type_ref not in self.data_types:
                        issues.append(
                            ValidationIssue(
                                "UNKNOWN_DATA_TYPE",
                                f"Provider '{provider_id}' port '{name}' references "
                                f"unknown data type '{type_ref}'.",
                            )
                        )
                    elif section in ("state_inputs", "state_outputs"):
                        kind = self.data_types[type_ref].get("kind")
                        if kind != "provider_state":
                            issues.append(
                                ValidationIssue(
                                    "STATE_PORT_TYPE_KIND",
                                    f"Provider '{provider_id}' state port '{name}' uses "
                                    f"'{type_ref}', whose kind is '{kind}', not 'provider_state'.",
                                )
                            )

            input_ports = self._provider_input_ports(provider_id)
            for group_index, group in enumerate(provider.get("compatibility_groups", []) or []):
                if not isinstance(group, dict):
                    issues.append(
                        ValidationIssue(
                            "COMPATIBILITY_GROUP_NOT_MAPPING",
                            f"Provider '{provider_id}' compatibility_groups[{group_index}] "
                            "must be a mapping.",
                        )
                    )
                    continue
                for port_name in group.get("ports", []) or []:
                    if port_name not in input_ports:
                        issues.append(
                            ValidationIssue(
                                "UNKNOWN_COMPATIBILITY_PORT",
                                f"Provider '{provider_id}' compatibility group references "
                                f"unknown input port '{port_name}'.",
                            )
                        )
                for key in group.get("require_same_runtime_keys", []) or []:
                    if not isinstance(key, str) or not key:
                        issues.append(
                            ValidationIssue(
                                "INVALID_RUNTIME_COMPATIBILITY_KEY",
                                f"Provider '{provider_id}' has an invalid runtime compatibility key.",
                            )
                        )

        return ValidationReport(issues)

    def validate_chain(
        self,
        chain_id: str,
        chain_override: Optional[Mapping[str, Any]] = None,
    ) -> ValidationReport:
        issues: List[ValidationIssue] = []

        chain = copy.deepcopy(
            dict(chain_override) if chain_override is not None else self.chains.get(chain_id)
        ) if (chain_override is not None or chain_id in self.chains) else None

        if chain is None:
            return ValidationReport(
                [
                    ValidationIssue(
                        "UNKNOWN_CHAIN",
                        f"Chain '{chain_id}' is not present in the catalog.",
                        chain_id=chain_id,
                    )
                ]
            )
        if not isinstance(chain, dict):
            return ValidationReport(
                [
                    ValidationIssue(
                        "CHAIN_NOT_MAPPING",
                        f"Chain '{chain_id}' must be a mapping.",
                        chain_id=chain_id,
                    )
                ]
            )

        external_inputs = chain.get("external_inputs", {})
        if external_inputs is None:
            external_inputs = {}
        if not isinstance(external_inputs, dict):
            issues.append(
                ValidationIssue(
                    "EXTERNAL_INPUTS_NOT_MAPPING",
                    "'external_inputs' must be a mapping.",
                    chain_id=chain_id,
                )
            )
            external_inputs = {}

        external_types: Dict[str, str] = {}
        for name, spec in external_inputs.items():
            if isinstance(spec, str):
                type_ref = spec
            elif isinstance(spec, dict):
                type_ref = spec.get("type")
            else:
                type_ref = None

            if not isinstance(name, str) or not name:
                issues.append(
                    ValidationIssue(
                        "EXTERNAL_INPUT_NAME_INVALID",
                        "External input names must be non-empty strings.",
                        chain_id=chain_id,
                    )
                )
                continue

            if not isinstance(type_ref, str) or not type_ref:
                issues.append(
                    ValidationIssue(
                        "EXTERNAL_INPUT_TYPE_MISSING",
                        f"External input '{name}' has no valid type.",
                        chain_id=chain_id,
                    )
                )
            elif type_ref not in self.data_types:
                issues.append(
                    ValidationIssue(
                        "UNKNOWN_DATA_TYPE",
                        f"External input '{name}' references unknown type '{type_ref}'.",
                        chain_id=chain_id,
                    )
                )
            else:
                external_types[name] = type_ref

        steps = chain.get("steps", [])
        if not isinstance(steps, list):
            return ValidationReport(
                issues
                + [
                    ValidationIssue(
                        "STEPS_NOT_LIST",
                        "'steps' must be a list.",
                        chain_id=chain_id,
                    )
                ]
            )

        completed_steps: Dict[str, Dict[str, str]] = {}
        all_step_ids: set[str] = set()

        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                issues.append(
                    ValidationIssue(
                        "STEP_NOT_MAPPING",
                        f"steps[{step_index}] must be a mapping.",
                        chain_id=chain_id,
                    )
                )
                continue

            step_id = step.get("id")
            provider_id = step.get("provider")

            if not isinstance(step_id, str) or not step_id:
                issues.append(
                    ValidationIssue(
                        "STEP_ID_MISSING",
                        f"steps[{step_index}] has no valid id.",
                        chain_id=chain_id,
                    )
                )
                continue

            if step_id in all_step_ids:
                issues.append(
                    ValidationIssue(
                        "DUPLICATE_STEP_ID",
                        f"Step id '{step_id}' is repeated.",
                        chain_id=chain_id,
                        step_id=step_id,
                    )
                )
                continue
            all_step_ids.add(step_id)

            if not isinstance(provider_id, str) or provider_id not in self.providers:
                issues.append(
                    ValidationIssue(
                        "UNKNOWN_PROVIDER",
                        f"Step '{step_id}' references unknown provider '{provider_id}'.",
                        chain_id=chain_id,
                        step_id=step_id,
                    )
                )
                continue

            input_ports = self._provider_input_ports(provider_id)
            output_ports = self._provider_output_ports(provider_id)
            bindings = step.get("bind", {})
            if bindings is None:
                bindings = {}
            if not isinstance(bindings, dict):
                issues.append(
                    ValidationIssue(
                        "BINDINGS_NOT_MAPPING",
                        "Step 'bind' must be a mapping from input-port names to sources.",
                        chain_id=chain_id,
                        step_id=step_id,
                    )
                )
                bindings = {}

            for bound_port in bindings:
                if bound_port not in input_ports:
                    issues.append(
                        ValidationIssue(
                            "UNKNOWN_INPUT_PORT",
                            f"Provider '{provider_id}' has no input port '{bound_port}'.",
                            chain_id=chain_id,
                            step_id=step_id,
                        )
                    )

            for port_name, port_spec in input_ports.items():
                required = bool(port_spec.get("required", True))
                if required and port_name not in bindings:
                    issues.append(
                        ValidationIssue(
                            "MISSING_REQUIRED_INPUT",
                            f"Required input '{port_name}' of provider '{provider_id}' is not bound.",
                            chain_id=chain_id,
                            step_id=step_id,
                        )
                    )

            for port_name, source_ref in bindings.items():
                if port_name not in input_ports:
                    continue
                expected_types = self._accepted_types(input_ports[port_name])
                actual_type, source_issue = self._resolve_source_type(
                    source_ref=source_ref,
                    external_types=external_types,
                    completed_steps=completed_steps,
                    all_declared_step_ids={
                        item.get("id")
                        for item in steps
                        if isinstance(item, dict) and isinstance(item.get("id"), str)
                    },
                )
                if source_issue:
                    issues.append(
                        ValidationIssue(
                            source_issue[0],
                            source_issue[1],
                            chain_id=chain_id,
                            step_id=step_id,
                        )
                    )
                    continue
                if actual_type not in expected_types:
                    issues.append(
                        ValidationIssue(
                            "TYPE_MISMATCH",
                            f"Input '{port_name}' expects {sorted(expected_types)}, "
                            f"but '{source_ref}' produces '{actual_type}'.",
                            chain_id=chain_id,
                            step_id=step_id,
                        )
                    )

            completed_steps[step_id] = {
                port_name: port_spec["type"]
                for port_name, port_spec in output_ports.items()
                if isinstance(port_spec.get("type"), str)
            }

        outputs = chain.get("outputs", {})
        if outputs is None:
            outputs = {}
        if not isinstance(outputs, dict):
            issues.append(
                ValidationIssue(
                    "CHAIN_OUTPUTS_NOT_MAPPING",
                    "'outputs' must be a mapping.",
                    chain_id=chain_id,
                )
            )
        else:
            for output_name, source_ref in outputs.items():
                actual_type, source_issue = self._resolve_source_type(
                    source_ref=source_ref,
                    external_types=external_types,
                    completed_steps=completed_steps,
                    all_declared_step_ids=all_step_ids,
                )
                if source_issue:
                    issues.append(
                        ValidationIssue(
                            source_issue[0],
                            f"Chain output '{output_name}': {source_issue[1]}",
                            chain_id=chain_id,
                        )
                    )
                elif actual_type is None:
                    issues.append(
                        ValidationIssue(
                            "CHAIN_OUTPUT_TYPE_UNRESOLVED",
                            f"Could not resolve chain output '{output_name}'.",
                            chain_id=chain_id,
                        )
                    )

        return ValidationReport(issues)

    def _provider_input_ports(self, provider_id: str) -> Dict[str, Dict[str, Any]]:
        return self._collect_ports(provider_id, self.INPUT_PORT_SECTIONS)

    def _provider_output_ports(self, provider_id: str) -> Dict[str, Dict[str, Any]]:
        return self._collect_ports(provider_id, self.OUTPUT_PORT_SECTIONS)

    def _collect_ports(
        self,
        provider_id: str,
        sections: Sequence[str],
    ) -> Dict[str, Dict[str, Any]]:
        provider = self.providers.get(provider_id, {})
        ports = provider.get("ports", {})
        collected: Dict[str, Dict[str, Any]] = {}
        if not isinstance(ports, dict):
            return collected
        for section in sections:
            entries = ports.get(section, []) or []
            if not isinstance(entries, list):
                continue
            for port in entries:
                if isinstance(port, dict) and isinstance(port.get("name"), str):
                    collected[port["name"]] = port
        return collected

    @staticmethod
    def _accepted_types(port_spec: Mapping[str, Any]) -> set[str]:
        accepted = port_spec.get("accepts")
        if isinstance(accepted, list) and accepted:
            return {item for item in accepted if isinstance(item, str)}
        type_ref = port_spec.get("type")
        return {type_ref} if isinstance(type_ref, str) else set()

    @staticmethod
    def _resolve_source_type(
        source_ref: Any,
        external_types: Mapping[str, str],
        completed_steps: Mapping[str, Mapping[str, str]],
        all_declared_step_ids: Iterable[str],
    ) -> Tuple[Optional[str], Optional[Tuple[str, str]]]:
        if not isinstance(source_ref, str) or "." not in source_ref:
            return None, (
                "INVALID_SOURCE_REFERENCE",
                f"Source reference '{source_ref}' must use 'external.name' or 'step.port'.",
            )

        owner, port = source_ref.split(".", 1)
        if not owner or not port:
            return None, (
                "INVALID_SOURCE_REFERENCE",
                f"Source reference '{source_ref}' is malformed.",
            )

        if owner == "external":
            if port not in external_types:
                return None, (
                    "UNKNOWN_EXTERNAL_INPUT",
                    f"Source '{source_ref}' references an unknown external input.",
                )
            return external_types[port], None

        if owner not in completed_steps:
            if owner in set(all_declared_step_ids):
                return None, (
                    "FORWARD_STEP_REFERENCE",
                    f"Source '{source_ref}' references step '{owner}' before it has executed.",
                )
            return None, (
                "UNKNOWN_SOURCE_STEP",
                f"Source '{source_ref}' references unknown step '{owner}'.",
            )

        outputs = completed_steps[owner]
        if port not in outputs:
            return None, (
                "UNKNOWN_OUTPUT_PORT",
                f"Source '{source_ref}' references an unknown output port.",
            )
        return outputs[port], None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate FABLE data types, provider contracts, and provider chains."
    )
    parser.add_argument(
        "--data-types",
        default="providers/registry/data_types.yaml",
        help="Path to data_types.yaml",
    )
    parser.add_argument(
        "--catalog",
        default="providers/registry/catalog.yaml",
        help="Path to catalog.yaml",
    )
    parser.add_argument(
        "--chain",
        action="append",
        default=[],
        help="Validate only this named chain; may be supplied more than once.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        validator = ChainValidator.from_files(args.data_types, args.catalog)
    except ChainValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    catalog_report = validator.validate_catalog()
    if args.chain:
        report = catalog_report
        for chain_id in args.chain:
            report.extend(validator.validate_chain(chain_id))
    else:
        report = validator.validate_all()

    if report.ok:
        selected = ", ".join(args.chain) if args.chain else "all catalog chains"
        print(f"Validation passed for {selected}.")
        return 0

    print("Validation failed:", file=sys.stderr)
    for issue in report.issues:
        print(f"  - {issue.format()}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
