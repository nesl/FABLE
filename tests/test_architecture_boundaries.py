"""Static regression tests for FABLE's architectural dependency boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

from fable.common.schemas import ExecutionPlan, SemanticGraph
from fable.contracts.execution import ExecutionPlan as CanonicalExecutionPlan
from fable.contracts.semantic import SemanticGraph as CanonicalSemanticGraph
from fable.planning.predicate_registry import default_predicate_registry
from fable.planning.provider_registry import default_provider_profiles

ROOT = Path(__file__).resolve().parents[1]


def _imports(path: str) -> tuple[str, ...]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return tuple(imported)


def test_distributed_node_agent_does_not_import_concrete_providers() -> None:
    assert not any(name == "providers" or name.startswith("providers.") for name in _imports("fable/distributed/node_agent.py"))


def test_core_provider_registry_does_not_import_concrete_providers() -> None:
    assert not any(name == "providers" or name.startswith("providers.") for name in _imports("fable/planning/provider_registry.py"))


def test_scheduler_lifecycle_does_not_import_physical_alternative_models() -> None:
    text = (ROOT / "fable/scheduling/lifecycle.py").read_text(encoding="utf-8")
    assert "PhysicalAlternative" not in text
    assert "StepPlacement" not in text


def test_contract_facade_preserves_canonical_class_identity() -> None:
    assert SemanticGraph is CanonicalSemanticGraph
    assert ExecutionPlan is CanonicalExecutionPlan


def test_authored_predicates_and_fallback_profiles_are_catalog_data() -> None:
    assert len(default_predicate_registry().predicate_ids) == 20
    assert len(default_provider_profiles()) == 35
    assert (ROOT / "fable/catalog/default_predicates.yaml").is_file()
    assert (ROOT / "fable/catalog/default_provider_profiles.yaml").is_file()


def test_controller_is_testbed_agnostic() -> None:
    text = (ROOT / "fable/orchestration/controller.py").read_text(encoding="utf-8").lower()
    assert "netwaggle" not in text
    assert "configured_one_way_ms" not in text
