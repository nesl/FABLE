from __future__ import annotations

import pytest

from evaluation.e4_campaign import (
    event_request_spec,
    planning_policy_id_for_baseline,
    load_condition_trace,
    runtime_disturbance_for_transition,
    runtime_disturbance_for_resource_change,
)
from evaluation.schemas import BaselineId
from evaluation.planning_cases import VARIANT_TEMPLATES
from fable.semantic.request_compiler import EventRequestCompiler
from fable.distributed.models import ResourceChange, ResourceKind


def test_legacy_b1_identifier_maps_to_redesigned_policy_id() -> None:
    assert planning_policy_id_for_baseline("B1_STATIC_WHOLE_EVENT") == (
        BaselineId.B1_HANDWRITTEN_STATIC.value
    )
    assert planning_policy_id_for_baseline(BaselineId.B1_HANDWRITTEN_STATIC) == (
        BaselineId.B1_HANDWRITTEN_STATIC.value
    )


CONDITIONS = (
    "evaluation/manifests/adaptation/physical_e4_multitrace_conditions/"
    "20241008-route-convoy-1-r012"
)


def test_manifest_alias_and_variant_translate_to_redesigned_request() -> None:
    spec = event_request_spec(
        ce_variant="Route convoy",
        baseline="B1_STATIC_WHOLE_EVENT",
    )
    assert spec.baseline_id == BaselineId.B1_HANDWRITTEN_STATIC
    assert spec.family_id == "convoy"
    assert spec.parameters["evaluation_profile"] == "sequential_passes"


@pytest.mark.parametrize(
    ("suffix", "field", "expected"),
    (
        ("compute_contention.json", "node_updates", 1),
        ("network_degradation.json", "link_updates", 1),
        ("network_disconnect.json", "link_updates", 1),
    ),
)
def test_physical_condition_translates_to_typed_disturbance(
    suffix: str, field: str, expected: int
) -> None:
    trace = load_condition_trace(f"{CONDITIONS}.{suffix}")
    request = runtime_disturbance_for_transition(
        trace=trace,
        transition_index=0,
        submitter_id="e4-test",
        planner_node_id="dvpg_gq_orin_1",
    )
    assert len(getattr(request, field)) == expected


def test_compatibility_compute_change_updates_target_gpu_capacity() -> None:
    apply = ResourceChange(
        run_id="e4-run",
        condition="E1",
        action="APPLY",
        condition_epoch=1,
        target_id="dvpg_gq_orin_7",
        resource_kind=ResourceKind.COMPUTE,
    )
    disturbed = runtime_disturbance_for_resource_change(
        change=apply,
        submitter_id="test",
        nominal_gpu_memory_mb=8192,
    )
    assert disturbed.node_updates[0].node_id == "dvpg_gq_orin_7"
    assert disturbed.node_updates[0].gpu_memory_available_mb == 0

    restore = apply.model_copy(update={"action": "RESTORE", "condition_epoch": 2})
    nominal = runtime_disturbance_for_resource_change(
        change=restore,
        submitter_id="test",
        nominal_gpu_memory_mb=8192,
    )
    assert nominal.node_updates[0].gpu_memory_available_mb == 8192


def test_unknown_baseline_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported redesigned E4 baseline"):
        event_request_spec(ce_variant="Route convoy", baseline="B0")


@pytest.mark.parametrize("ce_variant", tuple(VARIANT_TEMPLATES))
def test_every_e4_variant_compiles_through_redesigned_registry(
    ce_variant: str,
) -> None:
    spec = event_request_spec(ce_variant=ce_variant, baseline="FABLE")
    compilation = EventRequestCompiler().compile(
        {"family_id": spec.family_id, "parameters": spec.parameters}
    )
    assert compilation.family_id == spec.family_id
    assert compilation.graph.nodes
