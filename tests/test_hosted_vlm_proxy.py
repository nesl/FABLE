from __future__ import annotations

from datetime import timedelta
import pytest
import threading
from types import SimpleNamespace

from fable.common.enums import ResultKind
from fable.common.examples import BASE_TIME
from fable.common.schemas import PredicateRole, SemanticPredicate
from fable.distributed.config import ProviderRuntimeResolver
from fable.distributed.models import ReplayOutputAdapter, RuntimeMode
from fable.distributed.node_agent import NodeAgent
from fable.integrations.replay.output_adapters import build_replay_output_adapter_registry
from fable.planning.testing import fake_follow_demand
from providers.vehicle.models import EntityAssociation, EntityAssociationSet
from providers.vehicle.vlm_proxy import (
    HostedVlmProxy,
    ThreadingUnixHTTPServer,
    handler_for,
)
from providers.vehicle.vlm_reid import (
    OpenAIVisionIdentityComparator,
    RemoteVisionIdentityComparator,
    VlmIdentityDecision,
)


def test_remote_comparator_accepts_absolute_unix_socket_endpoint():
    comparator = RemoteVisionIdentityComparator(endpoint="unix:///run/fable/proxy.sock")
    assert comparator.endpoint == "/v1/compare"
    assert comparator._unix_socket == "/run/fable/proxy.sock"


def test_remote_comparator_rejects_relative_unix_socket_endpoint():
    with pytest.raises(ValueError, match="absolute"):
        RemoteVisionIdentityComparator(endpoint="unix://relative/proxy.sock")


def test_remote_comparator_reaches_proxy_over_unix_socket(tmp_path) -> None:
    socket_path = tmp_path / "proxy.sock"
    proxy = HostedVlmProxy(FakeComparator())
    server = ThreadingUnixHTTPServer(str(socket_path), handler_for(proxy))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        comparator = RemoteVisionIdentityComparator(
            endpoint=f"unix://{socket_path}", timeout_seconds=1
        )
        comparator.set_run_id("unix-test")
        decision = comparator.compare(
            entity_kind="vehicle",
            left_image_url="data:image/jpeg;base64,bGVmdA==",
            right_image_url="data:image/jpeg;base64,cmlnaHQ=",
        )
        assert decision.same_identity
        assert decision.confidence == 0.9
    finally:
        server.shutdown()
        server.server_close()


def test_direct_comparator_has_default_responses_transport() -> None:
    comparator = OpenAIVisionIdentityComparator(api_key="test-secret")
    assert callable(comparator._transport)


class FakeComparator:
    def compare(self, **_kwargs):
        return VlmIdentityDecision(
            same_identity=True,
            confidence=0.9,
            reason="profile fixture",
        )


def _request(run="run", invocation="run:1"):
    return {
        "schema_version": "fable.hosted_vlm_request.v1",
        "run_id": run,
        "invocation_id": invocation,
        "entity_kind": "vehicle",
        "left_image_url": "data:image/jpeg;base64,bGVmdA==",
        "right_image_url": "data:image/jpeg;base64,cmlnaHQ=",
    }


def test_proxy_rejects_unknown_fields_duplicates_and_eleventh_call() -> None:
    proxy = HostedVlmProxy(FakeComparator(), maximum_calls_per_run=10)
    first = proxy.compare(_request())
    assert first["same_identity"]
    with pytest.raises(ValueError, match="duplicate"):
        proxy.compare(_request())
    for number in range(2, 11):
        proxy.compare(_request(invocation=f"run:{number}"))
    with pytest.raises(ValueError, match="budget exhausted"):
        proxy.compare(_request(invocation="run:11"))
    unknown = _request(run="other", invocation="other:1")
    unknown["command"] = "anything"
    with pytest.raises(ValueError, match="unknown"):
        proxy.compare(unknown)


def test_proxy_accepts_only_inline_images() -> None:
    proxy = HostedVlmProxy(FakeComparator())
    document = _request()
    document["left_image_url"] = "file:///tmp/secret"
    with pytest.raises(ValueError, match="inline image"):
        proxy.compare(document)


def test_proxy_health_reflects_external_dependency_readiness() -> None:
    ready = HostedVlmProxy(
        FakeComparator(), readiness_check=lambda: (True, "resolved")
    )
    unavailable = HostedVlmProxy(
        FakeComparator(), readiness_check=lambda: (False, "dns unavailable")
    )

    assert ready.readiness() == (True, "resolved")
    assert unavailable.readiness() == (False, "dns unavailable")


def test_proxy_persists_exact_request_images_and_decision(tmp_path) -> None:
    proxy = HostedVlmProxy(FakeComparator(), debug_directory=tmp_path)
    proxy.compare(_request(run="replay/unsafe", invocation="replay/unsafe:1"))

    target = tmp_path / "replay_unsafe"
    assert (target / "replay_unsafe_1-left.jpg").read_bytes() == b"left"
    assert (target / "replay_unsafe_1-right.jpg").read_bytes() == b"right"
    decision = (target / "replay_unsafe_1-decision.json").read_text()
    assert '"same_identity": true' in decision
    assert "test-secret" not in decision


def test_remote_client_is_secret_free_and_scopes_invocations_by_run() -> None:
    requests = []

    def transport(document):
        requests.append(document)
        return {
            "schema_version": "fable.hosted_vlm_response.v1",
            "same_identity": False,
            "confidence": 0.85,
            "reason": "different",
        }

    client = RemoteVisionIdentityComparator(
        endpoint="http://fable-vlm-cloud1:8080",
        transport=transport,
    )
    client.set_run_id("replay-7")
    decision = client.compare(
        entity_kind="person",
        left_image_url="data:image/jpeg;base64,bGVmdA==",
        right_image_url="data:image/jpeg;base64,cmlnaHQ=",
    )
    assert not decision.same_identity
    assert requests[0]["invocation_id"] == "replay-7:1"
    assert "api_key" not in requests[0]


def test_cloud_runtime_maps_proxy_associations_to_same_entity_result() -> None:
    resolver = ProviderRuntimeResolver.from_yaml(
        "iobt-minimal-ce-replay/config/fable_provider_runtimes.yaml"
    )
    runtime = resolver.resolve(
        node_id="cloud1",
        provider_id="hosted_vlm_identity_comparator",
    )
    assert runtime.mode == RuntimeMode.ADOPT_EXISTING
    assert runtime.output_adapter == ReplayOutputAdapter.IDENTITY_ASSOCIATION
    assert runtime.readiness.container_health_required

    base = fake_follow_demand()
    demand = base.model_copy(
        update={
            "semantic_predicate": SemanticPredicate(
                predicate_id="SAME_ENTITY",
                roles=(
                    PredicateRole(
                        role_name="left",
                        variable="left",
                        entity_type="vehicle",
                    ),
                    PredicateRole(
                        role_name="right",
                        variable="right",
                        entity_type="vehicle",
                    ),
                ),
                result_kind=ResultKind.STATE_OBSERVATION,
            ),
            "bound_roles": {"left": "camera-a:track-1"},
            "unbound_roles": ("right",),
        }
    )
    associations = EntityAssociationSet(
        left_source_id="camera-a",
        right_source_id="camera-b",
        event_time_interval=demand.event_time_interval,
        entity_kind="vehicle",
        feature_space_key=(
            "vehicle",
            "model",
            "v1",
            "prep",
            512,
            "l2",
            "cosine",
        ),
        associations=(
            EntityAssociation(
                left_local_entity_id="camera-a:track-1",
                right_local_entity_id="camera-b:track-9",
                canonical_entity_id="vehicle-canonical-7",
                distance=0.1,
                confidence=0.9,
                association_basis="vlm_fallback",
            ),
        ),
    )
    agent = object.__new__(NodeAgent)
    agent.node_id = "cloud1"
    adapted = agent._adapt_provider_output(
        SimpleNamespace(demand=demand, runtime=runtime),
        ReplayOutputAdapter.IDENTITY_ASSOCIATION,
        associations.model_dump(mode="json"),
    )
    assert adapted is not None
    assert adapted[2] == {"right": "vehicle-canonical-7"}
    assert adapted[3] == 0.9
    reid_only = associations.model_copy(
        update={
            "associations": (
                associations.associations[0].model_copy(
                    update={"association_basis": "reid"}
                ),
            )
        }
    )
    reid_adapted = agent._adapt_provider_output(
        SimpleNamespace(demand=demand, runtime=runtime),
        ReplayOutputAdapter.IDENTITY_ASSOCIATION,
        reid_only.model_dump(mode="json"),
    )
    assert reid_adapted is not None
    assert reid_adapted[2] == {"right": "vehicle-canonical-7"}


def test_exact_identity_demand_accepts_bounded_retrospective_association() -> None:
    resolver = ProviderRuntimeResolver.from_yaml(
        "iobt-minimal-ce-replay/config/fable_provider_runtimes.yaml"
    )
    runtime = resolver.resolve(
        node_id="x86server",
        provider_id="cross_sensor_identity_association",
    )
    base = fake_follow_demand()
    demand = base.model_copy(
        update={
            "semantic_predicate": SemanticPredicate(
                predicate_id="SAME_ENTITY",
                roles=(
                    PredicateRole(role_name="left", variable="left", entity_type="vehicle"),
                    PredicateRole(role_name="right", variable="right", entity_type="vehicle"),
                ),
                parameters={"minimum_confidence": 0.4},
                result_kind=ResultKind.STATE_OBSERVATION,
            ),
            "bound_roles": {
                "left": "camera-a:track-1",
                "right": "camera-a:track-9",
            },
            "unbound_roles": (),
        }
    )
    retrospective_interval = demand.event_time_interval.model_copy(
        update={
            "start": demand.event_time_interval.start - timedelta(seconds=90),
            "end": demand.event_time_interval.start - timedelta(seconds=60),
        }
    )
    associations = EntityAssociationSet(
        left_source_id="camera-a",
        right_source_id="camera-a",
        event_time_interval=retrospective_interval,
        entity_kind="vehicle",
        feature_space_key=("model", "v1", "prep", 512, "l2", "cosine"),
        associations=(
            EntityAssociation(
                left_local_entity_id="camera-a:track-1",
                right_local_entity_id="camera-a:track-9",
                canonical_entity_id="vehicle-canonical-7",
                distance=0.17,
                confidence=0.428,
            ),
        ),
    )
    agent = object.__new__(NodeAgent)
    agent.node_id = "x86server"
    agent.output_adapters = build_replay_output_adapter_registry()
    adapted = agent._adapt_provider_output(
        SimpleNamespace(demand=demand, runtime=runtime),
        ReplayOutputAdapter.IDENTITY_ASSOCIATION,
        associations.model_dump(mode="json"),
    )
    assert adapted is not None
    assert adapted[3] == pytest.approx(0.428)
