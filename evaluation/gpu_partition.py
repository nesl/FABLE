"""Explicit two-GPU partition used by desktop evaluation bundles."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml


@dataclass(frozen=True)
class GpuPartition:
    device_gpu_uuid: str
    site_gpu_uuid: str

    def as_dict(self) -> dict[str, str]:
        return {
            "device_gpu_uuid": self.device_gpu_uuid,
            "site_gpu_uuid": self.site_gpu_uuid,
        }


def resolve_gpu_partition() -> GpuPartition:
    """Resolve GPU 0/device and GPU 1/site by UUID and reject ambiguity."""

    rows = _nvidia_gpus()
    by_index = {row["index"]: row["uuid"] for row in rows}
    device = os.environ.get("FABLE_DEVICE_GPU_UUID") or by_index.get(0)
    site = os.environ.get("FABLE_SITE_GPU_UUID") or by_index.get(1)
    if not device or not site:
        raise RuntimeError(
            "compute evaluation requires two GPUs (index 0=device, index 1=site) "
            "or explicit FABLE_DEVICE_GPU_UUID/FABLE_SITE_GPU_UUID"
        )
    known = {row["uuid"] for row in rows}
    if device not in known or site not in known:
        raise RuntimeError("configured GPU UUID is not present on this host")
    if device == site:
        raise RuntimeError("device and site tiers must use distinct physical GPUs")
    return GpuPartition(device_gpu_uuid=device, site_gpu_uuid=site)


def pin_compose_service(
    service: dict[str, Any], *, gpu_uuid: str, tier: str
) -> None:
    """Expose exactly one physical GPU to a Compose service."""

    # `gpus: all` plus NVIDIA_VISIBLE_DEVICES is not isolating on every Docker
    # Engine/Compose combination. Emit an exact DeviceRequest so the daemon,
    # not application convention, enforces the boundary.
    service["gpus"] = [{"driver": "nvidia", "device_ids": [gpu_uuid]}]
    environment = service.setdefault("environment", {})
    if isinstance(environment, list):
        converted = {}
        for item in environment:
            key, _, value = str(item).partition("=")
            converted[key] = value
        environment = converted
        service["environment"] = environment
    environment["NVIDIA_VISIBLE_DEVICES"] = gpu_uuid
    environment["FABLE_ASSIGNED_GPU_UUID"] = gpu_uuid
    environment["FABLE_COMPUTE_TIER"] = tier
    if "YOLO_DEVICE" in environment:
        # CUDA reindexes the sole visible UUID as device zero in the container.
        environment["YOLO_DEVICE"] = "0"


def inspect_partitioned_service(service: dict[str, Any]) -> tuple[str, str] | None:
    environment = service.get("environment") or {}
    if isinstance(environment, list):
        environment = {
            str(item).partition("=")[0]: str(item).partition("=")[2]
            for item in environment
        }
    uuid = str(environment.get("FABLE_ASSIGNED_GPU_UUID") or "")
    tier = str(environment.get("FABLE_COMPUTE_TIER") or "")
    return (uuid, tier) if uuid and tier else None


def validate_evaluation_bundle(bundle: Path, partition: GpuPartition) -> dict[str, Any]:
    """Fail closed if a generated bundle crosses or omits GPU boundaries."""

    assigned: dict[str, dict[str, str]] = {}
    compose_services: dict[str, dict[str, Any]] = {}
    for filename in ("compose.replay.yaml", "compose.fable.providers.yaml"):
        document = yaml.safe_load((bundle / filename).read_text(encoding="utf-8")) or {}
        for name, service in document.get("services", {}).items():
            if not isinstance(service, dict):
                continue
            container_name = str(service.get("container_name") or name)
            compose_services[container_name] = service
            row = inspect_partitioned_service(service)
            requests_gpu = bool(service.get("gpus"))
            if requests_gpu and row is None:
                raise RuntimeError(f"GPU service {name} has no explicit tier/UUID assignment")
            if row is None:
                continue
            uuid, tier = row
            expected = (
                partition.device_gpu_uuid if tier == "device"
                else partition.site_gpu_uuid if tier == "site_local"
                else None
            )
            if expected is None or uuid != expected:
                raise RuntimeError(f"GPU service {name} has invalid {tier}/{uuid} assignment")
            assigned[name] = {"tier": tier, "gpu_uuid": uuid}

    runtime_document = yaml.safe_load(
        (bundle / "fable_provider_runtimes.yaml").read_text(encoding="utf-8")
    ) or {}
    runtime_count = 0
    for node_id, node in runtime_document.get("nodes", {}).items():
        expected = (
            partition.site_gpu_uuid if node_id == "x86server"
            else partition.device_gpu_uuid
            if node_id.startswith("dvpg_gq_orin_") or node_id.startswith("mobile_")
            else None
        )
        for provider_id, runtime in node.get("providers", {}).items():
            if runtime.get("mode") == "REFERENCE" or expected is None:
                continue
            actual = tuple(runtime.get("gpu_device_ids") or ())
            if actual != (expected,):
                raise RuntimeError(
                    f"runtime {node_id}/{provider_id} GPU assignment {actual} != {(expected,)}"
                )
            if runtime.get("mode") == "ADOPT_EXISTING":
                container_name = str(runtime.get("container_name") or "")
                service = compose_services.get(container_name)
                if service is None:
                    raise RuntimeError(
                        f"runtime {node_id}/{provider_id} adopts unknown container "
                        f"{container_name!r}"
                    )
                assignment = inspect_partitioned_service(service)
                if assignment != (expected, "site_local" if node_id == "x86server" else "device"):
                    raise RuntimeError(
                        f"runtime {node_id}/{provider_id} requires GPU {expected}, but "
                        f"container {container_name} has assignment {assignment or 'unset'}"
                    )
            runtime_count += 1

    deployment = yaml.safe_load(
        (bundle / "fable_deployment.yaml").read_text(encoding="utf-8")
    ) or {}
    for node_id, node in deployment.get("nodes", {}).items():
        expected_pool = (
            "site_gpu1" if node_id == "x86server" else "device_gpu0"
            if node_id.startswith("dvpg_gq_orin_") or node_id.startswith("mobile_")
            else "desktop_cpu" if node_id == "cloud1" else None
        )
        if expected_pool and node.get("resource_pool_id") != expected_pool:
            raise RuntimeError(f"node {node_id} is not assigned to {expected_pool}")
    return {
        "schema_version": "fable.gpu_partition_validation.v1",
        **partition.as_dict(),
        "compose_service_count": len(assigned),
        "runtime_count": runtime_count,
        "validated": True,
    }


def _nvidia_gpus() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        shell=False, check=False, capture_output=True, text=True, timeout=5,
    )
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi could not enumerate the evaluation GPUs")
    rows = []
    for line in completed.stdout.splitlines():
        index, uuid = (item.strip() for item in line.split(",", 1))
        rows.append({"index": int(index), "uuid": uuid})
    return rows
