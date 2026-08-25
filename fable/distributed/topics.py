"""Canonical MQTT topics for FABLE Phase 6."""

from __future__ import annotations


PREFIX = "fable/v1"


def activate_topic(node_id: str) -> str:
    return f"{PREFIX}/command/{node_id}/activate"


def cancel_topic(node_id: str) -> str:
    return f"{PREFIX}/command/{node_id}/cancel"


def state_transfer_topic(node_id: str) -> str:
    return f"{PREFIX}/command/{node_id}/state-transfer"


def ack_topic(entity_id: str) -> str:
    return f"{PREFIX}/ack/{entity_id}"


def result_topic(request_id: str, predicate_id: str) -> str:
    return f"{PREFIX}/result/{request_id}/{predicate_id}"


def result_filter() -> str:
    return f"{PREFIX}/result/+/+"


def heartbeat_topic(node_id: str) -> str:
    return f"{PREFIX}/status/{node_id}/heartbeat"


def heartbeat_filter() -> str:
    return f"{PREFIX}/status/+/heartbeat"


def activity_topic(node_id: str) -> str:
    return f"{PREFIX}/status/{node_id}/activity"


def telemetry_topic(node_id: str) -> str:
    return f"{PREFIX}/telemetry/{node_id}/resources"


def provider_status_topic(node_id: str) -> str:
    return f"{PREFIX}/status/{node_id}/provider"


def provider_status_filter() -> str:
    return f"{PREFIX}/status/+/provider"


def artifact_topic(node_id: str) -> str:
    return f"{PREFIX}/artifact/{node_id}/announce"


def artifact_filter() -> str:
    return f"{PREFIX}/artifact/+/announce"


def dispatch_request_topic(orchestrator_id: str = "orchestrator") -> str:
    return f"{PREFIX}/control/{orchestrator_id}/dispatch"


def dispatch_response_topic(submitter_id: str) -> str:
    return f"{PREFIX}/control/{submitter_id}/dispatch-result"


def fault_topic(target_id: str) -> str:
    return f"{PREFIX}/control/{target_id}/fault"


def event_request_topic(orchestrator_id: str = "orchestrator") -> str:
    return f"{PREFIX}/control/{orchestrator_id}/event-request"


def event_response_topic(submitter_id: str) -> str:
    return f"{PREFIX}/control/{submitter_id}/event-request-result"


def terminal_event_topic(request_id: str) -> str:
    return f"{PREFIX}/event/{request_id}/completed"


def terminal_event_filter() -> str:
    return f"{PREFIX}/event/+/completed"


def disturbance_request_topic(orchestrator_id: str = "orchestrator") -> str:
    return f"{PREFIX}/control/{orchestrator_id}/runtime-disturbance"


def disturbance_ack_topic(submitter_id: str) -> str:
    return f"{PREFIX}/control/{submitter_id}/runtime-disturbance-ack"


def resource_change_topic() -> str:
    """Stable evaluator-to-controller resource transition topic."""

    return f"{PREFIX}/evaluation/resource_change"


def resource_change_ack_topic(run_id: str, request_message_id: str) -> str:
    return f"{PREFIX}/evaluation/resource_change_ack/{run_id}/{request_message_id}"


def resource_change_ack_filter(run_id: str) -> str:
    return f"{PREFIX}/evaluation/resource_change_ack/{run_id}/+"


# Legacy evaluation-wire boundary.  These names remain isolated here so the
# replay driver can be migrated without leaking its protocol into core models.
def live_request_topic(orchestrator_id: str = "orchestrator") -> str:
    return f"{PREFIX}/evaluation/{orchestrator_id}/live-request"


def live_request_response_topic(submitter_id: str) -> str:
    return f"{PREFIX}/evaluation/{submitter_id}/live-request-response"


def live_request_progress_topic(request_id: str) -> str:
    return f"{PREFIX}/evaluation/{request_id}/live-request-progress"


def live_request_cancel_topic(orchestrator_id: str = "orchestrator") -> str:
    return f"{PREFIX}/evaluation/{orchestrator_id}/live-request-cancel"


def live_request_cancel_response_topic(submitter_id: str) -> str:
    return f"{PREFIX}/evaluation/{submitter_id}/live-request-cancel-response"
