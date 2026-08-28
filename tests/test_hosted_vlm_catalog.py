from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.provider_execution import HostedProviderRunGate
from evaluation.profiled_vlm import ProfiledHostedVlmReplay
from fable.distributed.config import load_deployment_graph
from fable.planning.provider_registry import ProviderRegistry


ROOT = Path(__file__).resolve().parents[1]


def _registry() -> ProviderRegistry:
    return ProviderRegistry.from_files(
        catalog_path=ROOT / "providers/registry/catalog.yaml",
        data_types_path=ROOT / "providers/registry/data_types.yaml",
    )


def test_hosted_vlm_is_typed_cloud_capability_provider() -> None:
    registry = _registry()
    provider = registry.provider("hosted_vlm_identity_comparator")
    contract = provider.evaluation_contract
    assert provider.semantic_capabilities.predicate_ids == ("SAME_ENTITY",)
    assert provider.eligible_node_classes == ("server",)
    assert provider.required_node_capabilities == ("hosted_vlm",)
    assert contract.supported_modes == ("LIVE", "PROFILED")
    assert contract.hosted_external
    assert contract.maximum_invocations_per_run == 4
    assert contract.required_secret_names == ("OPENAI_API_KEY",)
    chain = registry.chain("same_entity_hosted_vlm_fallback")
    assert chain.output_types["result"] == "canonical_entity_map.v1"
    assert chain.continuation_output_types == ("canonical_entity_map.v1",)


def test_only_cloud_node_satisfies_hosted_vlm_capability() -> None:
    deployment = load_deployment_graph(
        ROOT / "iobt-minimal-ce-replay/config/fable_deployment.yaml"
    )
    nodes = deployment.candidate_nodes(required_capabilities=("hosted_vlm",))
    assert [node.node_id for node in nodes] == ["cloud1"]


def test_profiled_mode_needs_no_secret_but_keeps_four_call_limit() -> None:
    provider = _registry().provider("hosted_vlm_identity_comparator")
    gate = HostedProviderRunGate(
        run_id="run-profiled",
        mode="PROFILED",
        environment={},
    )
    permits = [gate.acquire(provider) for _ in range(4)]
    assert [item.invocation_number for item in permits] == [1, 2, 3, 4]
    with pytest.raises(RuntimeError, match="budget exhausted"):
        gate.acquire(provider)


def test_live_mode_checks_secret_name_without_exposing_value() -> None:
    provider = _registry().provider("hosted_vlm_identity_comparator")
    missing = HostedProviderRunGate(
        run_id="run-live",
        mode="LIVE",
        environment={},
    )
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY") as exc:
        missing.acquire(provider)
    assert "secret-value" not in str(exc.value)

    available = HostedProviderRunGate(
        run_id="run-live",
        mode="LIVE",
        environment={"OPENAI_API_KEY": "secret-value"},
    )
    permit = available.acquire(provider)
    assert permit.invocation_number == 1
    assert "secret-value" not in repr(permit)


def test_ambiguity_gate_uses_catalog_threshold() -> None:
    provider = _registry().provider("hosted_vlm_identity_comparator")
    gate = HostedProviderRunGate(run_id="run", mode="PROFILED")
    assert gate.fallback_eligible(provider, ambiguity_score=0.8)
    assert not gate.fallback_eligible(provider, ambiguity_score=0.81)


def test_profiled_vlm_replay_is_deterministic_and_single_use() -> None:
    provider = _registry().provider("hosted_vlm_identity_comparator")
    gate = HostedProviderRunGate(run_id="run", mode="PROFILED")
    replay = ProfiledHostedVlmReplay.from_yaml(
        ROOT
        / "evaluation/manifests/providers/hosted_vlm_profiled.example.yaml",
        provider=provider,
        gate=gate,
    )
    permit, decision = replay.invoke("example-person-pair")
    assert permit.mode == "PROFILED"
    assert decision.same_identity
    assert decision.confidence == 0.87
    with pytest.raises(RuntimeError, match="already consumed"):
        replay.invoke("example-person-pair")
    with pytest.raises(RuntimeError, match="is missing"):
        replay.invoke("not-in-profile")
