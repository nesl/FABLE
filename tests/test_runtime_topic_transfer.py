from fable.distributed.config import ProviderRuntimeResolver
from fable.distributed.models import ProviderRuntimeSpec, RuntimeMode


def _runtime(
    provider_id: str,
    node_id: str,
    *,
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
) -> ProviderRuntimeSpec:
    return ProviderRuntimeSpec(
        provider_id=provider_id,
        provider_contract_version=1,
        node_id=node_id,
        mode=RuntimeMode.ADOPT_EXISTING,
        container_name=f"{provider_id}-{node_id}",
        artifact_topic_inputs=inputs or {},
        artifact_topic_outputs=outputs or {},
    )


def test_exact_same_node_typed_topic_transfer_is_executable() -> None:
    topic = "/sensor_a/analytics/yolo/bbox"
    resolver = ProviderRuntimeResolver(
        {
            ("sensor_a", "detect"): _runtime(
                "detect", "sensor_a", outputs={"detection_set.v1": topic}
            ),
            ("sensor_a", "track"): _runtime(
                "track", "sensor_a", inputs={"detection_set.v1": topic}
            ),
        }
    )
    assert resolver.supports_artifact_topic_transfer(
        source_node_id="sensor_a",
        source_provider_id="detect",
        target_node_id="sensor_a",
        target_provider_id="track",
        data_type="detection_set.v1",
    )


def test_topic_transfer_rejects_mismatch_and_cross_node() -> None:
    resolver = ProviderRuntimeResolver(
        {
            ("sensor_a", "detect"): _runtime(
                "detect",
                "sensor_a",
                outputs={"detection_set.v1": "/sensor_a/detections"},
            ),
            ("sensor_a", "track"): _runtime(
                "track",
                "sensor_a",
                inputs={"detection_set.v1": "/wrong/topic"},
            ),
            ("sensor_b", "track"): _runtime(
                "track",
                "sensor_b",
                inputs={"detection_set.v1": "/sensor_a/detections"},
            ),
        }
    )
    common = dict(
        source_node_id="sensor_a",
        source_provider_id="detect",
        target_provider_id="track",
        data_type="detection_set.v1",
    )
    assert not resolver.supports_artifact_topic_transfer(
        target_node_id="sensor_a", **common
    )
    assert not resolver.supports_artifact_topic_transfer(
        target_node_id="sensor_b", **common
    )


def test_cross_node_transfer_accepts_explicit_mqtt_subscription_filter() -> None:
    source = _runtime(
        "crops",
        "sensor_a",
        outputs={"image_crop_set.v1": "/sensor_a/fable/identity/bounded-crops"},
    ).model_copy(update={"artifact_broker_scope_id": "evaluation-mqtt"})
    target = _runtime(
        "reid",
        "site",
        inputs={"image_crop_set.v1": "/+/fable/identity/bounded-crops"},
    ).model_copy(update={"artifact_broker_scope_id": "evaluation-mqtt"})
    resolver = ProviderRuntimeResolver(
        {("sensor_a", "crops"): source, ("site", "reid"): target}
    )
    assert resolver.supports_artifact_topic_transfer(
        source_node_id="sensor_a",
        source_provider_id="crops",
        target_node_id="site",
        target_provider_id="reid",
        data_type="image_crop_set.v1",
    )
