from __future__ import annotations

from types import SimpleNamespace
import json
import queue
import threading
from itertools import count

from fable.distributed.transport import PahoMQTTTransport


class _Client:
    def __init__(self) -> None:
        self.subscriptions: list[tuple[str, int]] = []

    def subscribe(self, topic: str, *, qos: int) -> None:
        self.subscriptions.append((topic, qos))


def test_on_connect_accepts_paho_v2_reason_code() -> None:
    transport = object.__new__(PahoMQTTTransport)
    transport.client_id = "test-client"
    transport.host = "mqtt"
    transport.port = 1883
    transport._connected = __import__("threading").Event()
    transport._subscriptions = {"fable/#": (1, [])}
    client = _Client()

    transport._on_connect(client, None, {}, SimpleNamespace(value=0))

    assert transport.connected
    assert client.subscriptions == [("fable/#", 1)]


def test_connected_duplicate_subscription_refreshes_retained_state() -> None:
    transport = object.__new__(PahoMQTTTransport)
    transport._connected = __import__("threading").Event()
    transport._connected.set()
    transport._client = _Client()
    first = lambda _topic, _payload: None
    second = lambda _topic, _payload: None
    transport._subscriptions = {"readiness/node/provider": (0, [first])}

    transport.subscribe("readiness/node/provider", second, qos=1)

    assert transport._subscriptions["readiness/node/provider"][1] == [first, second]
    assert transport._client.subscriptions == [("readiness/node/provider", 0)]


def test_on_message_queues_handlers_off_the_network_thread() -> None:
    transport = object.__new__(PahoMQTTTransport)
    transport.client_id = "test-client"
    transport._callback_queue = queue.PriorityQueue()
    transport._callback_sequence = count()
    seen = []
    transport._subscriptions = {
        "vehicle/#": (0, [lambda topic, payload: seen.append((topic, payload))])
    }

    transport._on_message(
        None,
        None,
        SimpleNamespace(topic="vehicle/tracks", payload=b"payload"),
    )

    _priority, _sequence, callback, topic, payload = (
        transport._callback_queue.get_nowait()
    )
    assert seen == []
    callback(topic, payload)
    assert seen == [("vehicle/tracks", b"payload")]


def test_resource_change_preempts_bulk_observation_callbacks() -> None:
    transport = object.__new__(PahoMQTTTransport)
    transport.client_id = "test-client"
    transport._callback_queue = queue.PriorityQueue()
    transport._callback_sequence = count()
    callback = lambda _topic, _payload: None
    transport._subscriptions = {
        "#": (1, [callback]),
    }

    transport._on_message(
        None,
        None,
        SimpleNamespace(
            topic="/orin14/fable/vehicle/predicates", payload=b"observation"
        ),
    )
    transport._on_message(
        None,
        None,
        SimpleNamespace(
            topic="fable/v1/evaluation/resource_change", payload=b"control"
        ),
    )

    priority, _sequence, _callback, topic, payload = (
        transport._callback_queue.get_nowait()
    )
    assert priority == -2
    assert topic == "fable/v1/evaluation/resource_change"
    assert payload == b"control"


def test_typed_result_preempts_raw_predicates_and_bulk_traffic() -> None:
    transport = object.__new__(PahoMQTTTransport)
    transport.client_id = "test-client"
    transport._callback_queue = queue.PriorityQueue()
    transport._callback_sequence = count()
    callback = lambda _topic, _payload: None
    transport._subscriptions = {"#": (1, [callback])}

    for index in range(10_000):
        transport._on_message(
            None,
            None,
            SimpleNamespace(
                topic="/orin14/fable/vehicle/tracks",
                payload=str(index).encode(),
            ),
        )
    transport._on_message(
        None,
        None,
        SimpleNamespace(
            topic="fable/v1/result/request-1/PASSES",
            payload=b"authoritative-result",
        ),
    )
    transport._on_message(
        None,
        None,
        SimpleNamespace(
            topic="/orin14/fable/vehicle/predicates",
            payload=b"sparse-semantic-evidence",
        ),
    )

    priority, _sequence, _callback, topic, payload = (
        transport._callback_queue.get_nowait()
    )
    assert priority == 0
    assert topic == "fable/v1/result/request-1/PASSES"
    assert payload == b"authoritative-result"


def test_identical_reliable_result_is_only_queued_once_while_pending() -> None:
    transport = object.__new__(PahoMQTTTransport)
    transport.client_id = "test-client"
    transport._callback_queue = queue.PriorityQueue()
    transport._callback_sequence = count()
    transport._pending_callback_tokens = set()
    transport._callback_tokens = {}
    transport._callback_token_lock = threading.Lock()
    callback = lambda _topic, _payload: None
    transport._subscriptions = {"#": (1, [callback])}
    message = SimpleNamespace(
        topic="fable/v1/result/request-1/PASSES",
        payload=json.dumps({"message_id": "same-reliable-message"}).encode(),
    )

    transport._on_message(None, None, message)
    transport._on_message(None, None, message)

    assert transport._callback_queue.qsize() == 1


def test_distinct_reliable_results_are_not_coalesced() -> None:
    transport = object.__new__(PahoMQTTTransport)
    transport.client_id = "test-client"
    transport._callback_queue = queue.PriorityQueue()
    transport._callback_sequence = count()
    transport._pending_callback_tokens = set()
    transport._callback_tokens = {}
    transport._callback_token_lock = threading.Lock()
    callback = lambda _topic, _payload: None
    transport._subscriptions = {"#": (1, [callback])}

    for message_id in ("result-a", "result-b"):
        transport._on_message(
            None,
            None,
            SimpleNamespace(
                topic="fable/v1/result/request-1/PASSES",
                payload=json.dumps({"message_id": message_id}).encode(),
            ),
        )

    assert transport._callback_queue.qsize() == 2


def test_sparse_transition_preempts_dense_predicates_without_preempting_control() -> None:
    transport = object.__new__(PahoMQTTTransport)
    transport.client_id = "test-client"
    transport._callback_queue = queue.PriorityQueue()
    transport._callback_sequence = count()
    callback = lambda _topic, _payload: None
    transport._subscriptions = {"#": (1, [callback])}

    for predicate_id in ("DISTANCE_LT", "STOPPED", "PASSES"):
        transport._on_message(
            None,
            None,
            SimpleNamespace(
                topic="/orin14/fable/vehicle/predicates",
                payload=json.dumps({"predicate_id": predicate_id}).encode(),
            ),
        )

    priority, _sequence, _callback, _topic, payload = (
        transport._callback_queue.get_nowait()
    )
    assert priority == 1
    assert json.loads(payload)["predicate_id"] == "PASSES"


def test_control_callback_runs_while_evidence_callback_is_blocked() -> None:
    transport = object.__new__(PahoMQTTTransport)
    transport.client_id = "test-client"
    transport._callback_queue = queue.PriorityQueue()
    transport._control_callback_queue = queue.PriorityQueue()
    transport._callback_sequence = count()
    evidence_started = threading.Event()
    release_evidence = threading.Event()
    control_completed = threading.Event()

    def callback(topic: str, _payload: bytes) -> None:
        if topic == "/orin14/fable/vehicle/predicates":
            evidence_started.set()
            release_evidence.wait(timeout=2.0)
        elif "/command/" in topic:
            control_completed.set()

    transport._subscriptions = {"#": (1, [callback])}
    evidence_worker = threading.Thread(
        target=transport._run_callbacks,
        args=(transport._callback_queue,),
        daemon=True,
    )
    control_worker = threading.Thread(
        target=transport._run_callbacks,
        args=(transport._control_callback_queue,),
        daemon=True,
    )
    evidence_worker.start()
    control_worker.start()
    transport._on_message(
        None,
        None,
        SimpleNamespace(
            topic="/orin14/fable/vehicle/predicates",
            payload=json.dumps({"predicate_id": "DISTANCE_LT"}).encode(),
        ),
    )
    assert evidence_started.wait(timeout=1.0)
    transport._on_message(
        None,
        None,
        SimpleNamespace(topic="fable/v1/command/orin14/activate", payload=b"{}"),
    )

    assert control_completed.wait(timeout=1.0)
    assert not release_evidence.is_set()
    release_evidence.set()
    transport._callback_queue.put((100, 10_000, None, "", b""))
    transport._control_callback_queue.put((100, 10_001, None, "", b""))
    evidence_worker.join(timeout=1.0)
    control_worker.join(timeout=1.0)
