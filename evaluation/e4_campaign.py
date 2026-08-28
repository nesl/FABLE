"""Strict adapters from the prepared physical E4 manifest to redesigned APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evaluation.condition_trace import ConditionAction, ConditionTrace
from evaluation.planning_cases import VARIANT_TEMPLATES
from evaluation.schemas import BaselineId
from fable.common.schemas import RuntimeLinkUpdate, RuntimeNodeUpdate
from fable.distributed.models import ResourceChange, ResourceKind, RuntimeDisturbanceRequest


BASELINE_ALIASES = {
    "B1_STATIC_WHOLE_EVENT": BaselineId.B1_HANDWRITTEN_STATIC,
    "B1_HANDWRITTEN_STATIC": BaselineId.B1_HANDWRITTEN_STATIC,
    "B3_TASK_RESOURCE_ADAPTIVE": BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
    "FABLE": BaselineId.FABLE,
}


def planning_policy_id_for_baseline(baseline: str | BaselineId) -> str:
    """Resolve evaluation-facing aliases at the service trust boundary."""

    value = baseline.value if isinstance(baseline, BaselineId) else str(baseline)
    try:
        return BASELINE_ALIASES[value].value
    except KeyError:
        # FABLE is implemented by the controller's native/default policy and
        # may not need a baseline adapter. Preserve any explicitly supported
        # identifier so the controller remains the final policy authority.
        return value


@dataclass(frozen=True)
class E4EventRequestSpec:
    family_id: str
    parameters: dict[str, object]
    baseline_id: BaselineId


def event_request_spec(*, ce_variant: str, baseline: str) -> E4EventRequestSpec:
    try:
        template = VARIANT_TEMPLATES[ce_variant]
    except KeyError as exc:
        raise ValueError(f"unsupported E4 CE variant: {ce_variant}") from exc
    try:
        baseline_id = BASELINE_ALIASES[baseline]
    except KeyError as exc:
        raise ValueError(f"unsupported redesigned E4 baseline: {baseline}") from exc
    return E4EventRequestSpec(
        family_id=template.family_id,
        parameters=dict(template.request_parameters or {}),
        baseline_id=baseline_id,
    )


def load_condition_trace(path: str | Path) -> ConditionTrace:
    return ConditionTrace.model_validate_json(Path(path).read_text(encoding="utf-8"))


def runtime_disturbance_for_transition(
    *,
    trace: ConditionTrace,
    transition_index: int,
    submitter_id: str,
    planner_node_id: str,
    site_node_id: str = "x86server",
) -> RuntimeDisturbanceRequest:
    """Translate one allowlisted physical transition into planner-visible state.

    Host mutation remains the physical runner's responsibility. This function
    creates the matching controller update and rejects every unknown action.
    """

    transition = trace.transitions[transition_index]
    action = transition.action
    reason = f"E4 {trace.trace_id} transition {transition.transition_id}"
    if action == ConditionAction.APPLY_COMPUTE_CONTENTION:
        nodes = (RuntimeNodeUpdate(node_id=planner_node_id, gpu_memory_available_mb=0),)
        links = ()
    elif action == ConditionAction.CLEAR_COMPUTE_CONTENTION:
        # The physical E4 deployment profiles provision Jetson-class nodes with
        # 8 GiB. The campaign validator checks that this target exists before run.
        nodes = (RuntimeNodeUpdate(node_id=planner_node_id, gpu_memory_available_mb=8192),)
        links = ()
    elif action == ConditionAction.APPLY_NETWORK_PROFILE:
        nodes = ()
        links = (
            RuntimeLinkUpdate(
                source_node_id=planner_node_id,
                target_node_id=site_node_id,
                latency_ms=75,
                bandwidth_mbps=5,
                available=True,
            ),
        )
    elif action == ConditionAction.RESTORE_NETWORK_PROFILE:
        nodes = ()
        links = (
            RuntimeLinkUpdate(
                source_node_id=planner_node_id,
                target_node_id=site_node_id,
                latency_ms=2,
                bandwidth_mbps=1000,
                available=True,
            ),
        )
    elif action == ConditionAction.FAIL_LINK:
        nodes = ()
        links = (
            RuntimeLinkUpdate(
                source_node_id=planner_node_id,
                target_node_id=site_node_id,
                available=False,
            ),
        )
    elif action == ConditionAction.RESTORE_LINK:
        nodes = ()
        links = (
            RuntimeLinkUpdate(
                source_node_id=planner_node_id,
                target_node_id=site_node_id,
                latency_ms=2,
                bandwidth_mbps=1000,
                available=True,
            ),
        )
    else:
        raise ValueError(f"E4 physical adapter does not allow action {action.value}")
    return RuntimeDisturbanceRequest(
        submitter_id=submitter_id,
        disturbance_id=transition.transition_id,
        node_updates=nodes,
        link_updates=links,
        reason=reason,
    )


def runtime_disturbance_for_resource_change(
    *,
    change: ResourceChange,
    submitter_id: str,
    nominal_gpu_memory_mb: int,
) -> RuntimeDisturbanceRequest:
    """Translate a compatibility compute epoch into authoritative capacity."""

    if change.resource_kind not in {ResourceKind.COMPUTE, ResourceKind.GPU}:
        raise ValueError(
            f"resource kind {change.resource_kind.value} is not a compute disturbance"
        )
    if not change.target_id:
        raise ValueError("compute resource change requires target_id")
    action = change.action.upper()
    if action == "APPLY":
        available_gpu = 0
    elif action in {"RESTORE", "CLEAR"}:
        available_gpu = nominal_gpu_memory_mb
    else:
        raise ValueError(f"unsupported compute resource action: {change.action}")
    return RuntimeDisturbanceRequest(
        submitter_id=submitter_id,
        disturbance_id=(
            f"compat:{change.run_id}:{change.condition_epoch}:"
            f"{change.resource_kind.value}:{change.target_id}"
        ),
        node_updates=(
            RuntimeNodeUpdate(
                node_id=change.target_id,
                gpu_memory_available_mb=available_gpu,
            ),
        ),
        reason=(
            f"evaluation resource epoch {change.condition_epoch}: "
            f"{change.resource_kind.value}:{action}:{change.condition}"
        ),
        submitted_at=change.observed_at,
    )


__all__ = [
    "BASELINE_ALIASES",
    "planning_policy_id_for_baseline",
    "E4EventRequestSpec",
    "event_request_spec",
    "load_condition_trace",
    "runtime_disturbance_for_transition",
    "runtime_disturbance_for_resource_change",
]
