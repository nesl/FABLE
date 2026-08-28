from __future__ import annotations

from evaluation.gpu_partition import GpuPartition, pin_compose_service


def test_pin_compose_service_exposes_exactly_one_uuid() -> None:
    service = {"gpus": "all", "environment": {"YOLO_DEVICE": "auto"}}
    pin_compose_service(service, gpu_uuid="GPU-site", tier="site_local")
    assert service["gpus"] == [
        {"driver": "nvidia", "device_ids": ["GPU-site"]}
    ]
    assert service["environment"] == {
        "YOLO_DEVICE": "0",
        "NVIDIA_VISIBLE_DEVICES": "GPU-site",
        "FABLE_ASSIGNED_GPU_UUID": "GPU-site",
        "FABLE_COMPUTE_TIER": "site_local",
    }


def test_partition_requires_distinct_uuids() -> None:
    partition = GpuPartition("GPU-device", "GPU-site")
    assert partition.as_dict() == {
        "device_gpu_uuid": "GPU-device",
        "site_gpu_uuid": "GPU-site",
    }
