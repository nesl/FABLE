"""Typed E4 live-run client for the redesigned FABLE controller.

The worker talks only to public FABLE contracts: event request/response,
runtime-disturbance request/ack, and terminal complex-event output.  It does not
reach into legacy live-evaluation records or controller persistence internals.
"""

from __future__ import annotations

import threading

from fable.common.schemas import TerminalComplexEvent
from fable.distributed.codec import decode_model, encode_model
from fable.distributed.models import (
    EventRequestResponse,
    EventRequestSubmission,
    RuntimeDisturbanceAck,
    RuntimeDisturbanceRequest,
)
from fable.distributed.topics import (
    disturbance_ack_topic,
    disturbance_request_topic,
    event_request_topic,
    event_response_topic,
    terminal_event_filter,
)
from fable.distributed.transport import Transport

from evaluation.runner import terminal_complex_event_record
from evaluation.schemas import BaselineId, ComplexEventResult


class E4TimeoutError(TimeoutError):
    pass


class E4Worker:
    """Synchronous facade over the typed MQTT control/result streams used by E4."""

    def __init__(
        self,
        *,
        transport: Transport,
        submitter_id: str,
        orchestrator_id: str = "orchestrator",
    ) -> None:
        self.transport = transport
        self.submitter_id = submitter_id
        self.orchestrator_id = orchestrator_id
        self._condition = threading.Condition()
        self._event_responses: dict[str, EventRequestResponse] = {}
        self._disturbance_acks: dict[str, RuntimeDisturbanceAck] = {}
        self._terminal_events: dict[str, list[TerminalComplexEvent]] = {}
        self._bound = False

    def bind(self) -> None:
        if self._bound:
            return
        self.transport.subscribe(
            event_response_topic(self.submitter_id),
            self._on_event_response,
            qos=1,
        )
        self.transport.subscribe(
            disturbance_ack_topic(self.submitter_id),
            self._on_disturbance_ack,
            qos=1,
        )
        self.transport.subscribe(terminal_event_filter(), self._on_terminal_event, qos=1)
        self._bound = True

    def submit_event(
        self,
        submission: EventRequestSubmission,
        *,
        timeout: float = 10.0,
    ) -> EventRequestResponse:
        if submission.submitter_id != self.submitter_id:
            raise ValueError("submission submitter_id must match E4Worker submitter_id")
        self.bind()
        self.transport.publish(
            event_request_topic(self.orchestrator_id),
            encode_model(submission),
            qos=1,
            retain=False,
        )
        key = str(submission.message_id)
        return self._wait_for(self._event_responses, key, timeout, "event request response")

    def inject_disturbance(
        self,
        request: RuntimeDisturbanceRequest,
        *,
        timeout: float = 10.0,
    ) -> RuntimeDisturbanceAck:
        if request.submitter_id != self.submitter_id:
            raise ValueError("disturbance submitter_id must match E4Worker submitter_id")
        self.bind()
        self.transport.publish(
            disturbance_request_topic(self.orchestrator_id),
            encode_model(request),
            qos=1,
            retain=False,
        )
        key = str(request.message_id)
        return self._wait_for(self._disturbance_acks, key, timeout, "disturbance acknowledgment")

    def wait_terminal(
        self,
        request_id: str,
        *,
        timeout: float = 60.0,
    ) -> TerminalComplexEvent:
        self.bind()
        with self._condition:
            if not self._condition.wait_for(
                lambda: bool(self._terminal_events.get(request_id)),
                timeout=timeout,
            ):
                raise E4TimeoutError(
                    f"timed out after {timeout:.1f}s waiting for terminal CE request={request_id}"
                )
            return self._terminal_events[request_id][0]

    @staticmethod
    def to_evaluation_result(
        event: TerminalComplexEvent,
        *,
        run_id: str,
        trace_id: str,
        baseline_id: BaselineId,
    ) -> ComplexEventResult:
        return terminal_complex_event_record(
            event,
            run_id=run_id,
            trace_id=trace_id,
            baseline_id=baseline_id,
        )

    def _wait_for(self, mapping, key: str, timeout: float, label: str):
        with self._condition:
            if not self._condition.wait_for(lambda: key in mapping, timeout=timeout):
                raise E4TimeoutError(f"timed out after {timeout:.1f}s waiting for {label}")
            return mapping[key]

    def _on_event_response(self, _topic: str, payload: bytes) -> None:
        response = decode_model(payload, EventRequestResponse)
        with self._condition:
            self._event_responses[str(response.request_message_id)] = response
            self._condition.notify_all()

    def _on_disturbance_ack(self, _topic: str, payload: bytes) -> None:
        ack = decode_model(payload, RuntimeDisturbanceAck)
        with self._condition:
            self._disturbance_acks[str(ack.request_message_id)] = ack
            self._condition.notify_all()

    def _on_terminal_event(self, _topic: str, payload: bytes) -> None:
        event = decode_model(payload, TerminalComplexEvent)
        with self._condition:
            self._terminal_events.setdefault(event.request_id, []).append(event)
            self._condition.notify_all()


def planning_policy_for_baseline(baseline_id: BaselineId) -> str:
    if baseline_id in {
        BaselineId.B1_HANDWRITTEN_STATIC,
        BaselineId.B3_TASK_RESOURCE_ADAPTIVE,
        BaselineId.B4_GREEDY_FRONTIER,
        BaselineId.FABLE,
    }:
        return baseline_id.value
    raise ValueError(f"E4 live controller does not expose baseline {baseline_id.value}")


__all__ = ["E4TimeoutError", "E4Worker", "planning_policy_for_baseline"]
