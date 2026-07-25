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
