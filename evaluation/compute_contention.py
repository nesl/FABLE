"""Closed, measurable controller for the site-local GPU contention experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


CONTAINER_NAME = "fable-gpu-contention"
IMAGE = "iobt-minimal/yolo-detector:latest"
STATE_PATH = Path("/tmp/fable-gpu-contention.json")
CALIBRATION_PATH = Path("runs/compute_contention/gpu1_calibration.json")


def calibrate_e1(*, settle_seconds: float = 8.0) -> dict[str, Any]:
    """Calibrate before a campaign; never consume trace time doing this."""
    requested_at = time.time()
    _remove_stale()
    gpu = _gpu_identity(tier="site")
    device_gpu = _gpu_identity(tier="device")
    nominal_samples = _gpu_samples(gpu["uuid"])
    device_nominal_samples = _gpu_samples(device_gpu["uuid"])
    nominal_benchmark = _benchmark(gpu["uuid"])
    started = _start_contention(gpu["uuid"])
    container_started_at = started["container_started_at_unix"]
    ready_at = started["ready_at_unix"]
    ready_record = started["ready_record"]
    deadline = time.monotonic() + settle_seconds
    samples = []
    while time.monotonic() < deadline:
        samples.extend(_gpu_samples(gpu["uuid"]))
        time.sleep(0.5)
    running = _inspect_running()
    if not running:
        raise RuntimeError("compute contention container exited during calibration")
    utilization = [row["gpu_utilization_percent"] for row in samples]
    memory = [row["gpu_memory_fraction"] for row in samples]
    device_disturbed_samples = _gpu_samples(device_gpu["uuid"])
    disturbed_benchmark = _benchmark(gpu["uuid"])
    slowdown = (
        disturbed_benchmark["p95_ms"] / nominal_benchmark["p95_ms"]
        if nominal_benchmark["p95_ms"] > 0 else None
    )
    mean_utilization = sum(utilization) / len(utilization) if utilization else None
    device_nominal_utilization = _mean_utilization(device_nominal_samples)
    device_disturbed_utilization = _mean_utilization(device_disturbed_samples)
    device_utilization_delta = (
        device_disturbed_utilization - device_nominal_utilization
        if device_nominal_utilization is not None
        and device_disturbed_utilization is not None else None
    )
    calibrated = bool(
        mean_utilization is not None
        and 70 <= mean_utilization <= 90
        and slowdown is not None
        and 1.8 <= slowdown <= 2.5
        and device_utilization_delta is not None
        and device_utilization_delta <= 15.0
    )
    result = {
        "profile_id": "E1",
        "container_name": CONTAINER_NAME,
        "image": IMAGE,
        "sample_count": len(samples),
        "gpu_uuid": gpu["uuid"],
        "gpu_index": gpu["index"],
        "gpu_tier": "site_local",
        "device_gpu_uuid": device_gpu["uuid"],
        "requested_at_unix": requested_at,
        "container_started_at_unix": container_started_at,
        "ready_at_unix": ready_at,
        "ready_record": ready_record,
        "mean_gpu_utilization_percent": mean_utilization,
        "p95_gpu_utilization_percent": _percentile(utilization, 0.95),
        "mean_gpu_memory_fraction": sum(memory) / len(memory) if memory else None,
        "peak_gpu_memory_fraction": max(memory) if memory else None,
        "nominal_gpu_samples": nominal_samples,
        "device_gpu_nominal_samples": device_nominal_samples,
        "device_gpu_disturbed_samples": device_disturbed_samples,
        "device_gpu_utilization_delta_percent": device_utilization_delta,
        "nominal_provider_benchmark": nominal_benchmark,
        "disturbed_provider_benchmark": disturbed_benchmark,
        "provider_p95_slowdown": slowdown,
        "target_gpu_utilization_percent": [70, 90],
        "target_provider_p95_slowdown": [1.8, 2.5],
        "maximum_device_gpu_utilization_delta_percent": 15.0,
        "running": running,
        "calibrated": calibrated,
    }
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    STATE_PATH.write_text(json.dumps(result) + "\n", encoding="utf-8")
    recovery = restore_n0()
    result["calibration_recovery"] = recovery
    CALIBRATION_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def apply_e1(*, settle_seconds: float = 5.0) -> dict[str, Any]:
    """Apply a previously calibrated workload at the trace transition."""

    if not CALIBRATION_PATH.is_file():
        raise RuntimeError(
            f"missing pre-run GPU calibration: {CALIBRATION_PATH}; run CALIBRATE first"
        )
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    if calibration.get("calibrated") is not True:
        raise RuntimeError("GPU contention calibration is not valid")
    gpu = _gpu_identity(tier="site")
    device_gpu = _gpu_identity(tier="device")
    if calibration.get("gpu_uuid") != gpu["uuid"]:
        raise RuntimeError("site GPU UUID changed since calibration")
    if calibration.get("device_gpu_uuid") != device_gpu["uuid"]:
        raise RuntimeError("device GPU UUID changed since calibration")
    _remove_stale()
    requested_at = time.time()
    started = _start_contention(gpu["uuid"])
    site_samples: list[dict[str, float]] = []
    device_samples: list[dict[str, float]] = []
    deadline = time.monotonic() + settle_seconds
    while time.monotonic() < deadline:
        site_samples.extend(_gpu_samples(gpu["uuid"]))
        device_samples.extend(_gpu_samples(device_gpu["uuid"]))
        time.sleep(0.5)
    site_mean = _mean_utilization(site_samples)
    device_mean = _mean_utilization(device_samples)
    nominal_device = _mean_utilization(calibration.get("device_gpu_nominal_samples", []))
    device_delta = (
        device_mean - nominal_device
        if device_mean is not None and nominal_device is not None else None
    )
    gpu_isolation = _contention_gpu_isolated(gpu["uuid"])
    validated = bool(
        _inspect_running()
        and site_mean is not None and 70 <= site_mean <= 90
        and gpu_isolation
    )
    result = {
        **calibration,
        "profile_id": "E1",
        "requested_at_unix": requested_at,
        **started,
        "activation_site_gpu_samples": site_samples,
        "activation_device_gpu_samples": device_samples,
        "mean_gpu_utilization_percent": site_mean,
        "device_gpu_utilization_delta_percent": device_delta,
        "device_gpu_utilization_is_informational": True,
        "contention_gpu_assignment_isolated": gpu_isolation,
        "running": _inspect_running(),
        "calibrated": validated,
        "calibration_path": str(CALIBRATION_PATH),
    }
    STATE_PATH.write_text(json.dumps(result) + "\n", encoding="utf-8")
    return result


def restore_n0() -> dict[str, Any]:
    requested_at = time.time()
    prior = json.loads(STATE_PATH.read_text()) if STATE_PATH.is_file() else {}
    completed = subprocess.run(
        ["docker", "rm", "--force", CONTAINER_NAME],
        shell=False, check=False, capture_output=True, text=True, timeout=20,
    )
    STATE_PATH.unlink(missing_ok=True)
    if completed.returncode != 0 and "No such container" not in completed.stderr:
        raise RuntimeError(completed.stderr.strip())
    stopped_at = time.time()
    uuid = str(prior.get("gpu_uuid") or _gpu_identity(tier="site")["uuid"])
    recovery = []
    deadline = time.monotonic() + 20.0
    nominal = [float(row["gpu_utilization_percent"]) for row in prior.get("nominal_gpu_samples", [])]
    ceiling = (sum(nominal) / len(nominal) + 10.0) if nominal else 20.0
    while time.monotonic() < deadline:
        recovery.extend(_gpu_samples(uuid))
        if recovery and recovery[-1]["gpu_utilization_percent"] <= ceiling:
            break
        time.sleep(0.5)
    recovered_benchmark = _benchmark(uuid)
    nominal_benchmark = prior.get("nominal_provider_benchmark") or {}
    latency_ratio = (
        recovered_benchmark["p95_ms"] / float(nominal_benchmark["p95_ms"])
        if float(nominal_benchmark.get("p95_ms") or 0) > 0 else None
    )
    device_uuid = str(
        prior.get("device_gpu_uuid") or _gpu_identity(tier="device")["uuid"]
    )
    device_recovery = _gpu_samples(device_uuid)
    utilization_recovered = bool(
        recovery and recovery[-1]["gpu_utilization_percent"] <= ceiling
    )
    latency_recovered = bool(latency_ratio is not None and latency_ratio <= 1.25)
    return {
        "profile_id": "N0", "container_name": CONTAINER_NAME, "running": False,
        "stop_requested_at_unix": requested_at, "container_stopped_at_unix": stopped_at,
        "gpu_capacity_restored_at_unix": time.time(), "recovery_samples": recovery,
        "recovered_provider_benchmark": recovered_benchmark,
        "provider_p95_recovery_ratio": latency_ratio,
        "device_gpu_uuid": device_uuid,
        "device_gpu_recovery_samples": device_recovery,
        "utilization_recovered": utilization_recovered,
        "latency_recovered": latency_recovered,
        "recovered": utilization_recovered and latency_recovered,
    }


def _start_contention(gpu_uuid: str) -> dict[str, Any]:
    started = subprocess.run(
        [
            "docker", "run", "--detach", "--name", CONTAINER_NAME,
            "--label", "fable.evaluation.disturbance=compute-contention-e1",
            "--network", "none", "--gpus", f"device={gpu_uuid}",
            "--cpus", "2.0", "--memory", "8g",
            "--env", "YOLO_DEVICE=0",
            "--env", f"FABLE_CONTENTION_GPU_UUID={gpu_uuid}",
            "--env", "FABLE_CONTENTION_BATCH_SIZE=8",
            "--env", "FABLE_CONTENTION_STREAMS=4",
            "--env", "RQ3A_GPU_DUTY=0.80",
            IMAGE, "python3", "-u", "/app/rq3a_contention.py",
        ],
        shell=False, check=False, capture_output=True, text=True, timeout=30,
    )
    if started.returncode != 0:
        raise RuntimeError(started.stderr.strip() or "compute container failed to start")
    container_started_at = time.time()
    deadline = time.monotonic() + 25.0
    ready_record = None
    while time.monotonic() < deadline:
        ready_record = _ready_record()
        if ready_record is not None:
            break
        if not _inspect_running():
            raise RuntimeError("compute contention container exited during startup")
        time.sleep(0.5)
    else:
        raise RuntimeError("compute contention did not emit READY")
    return {
        "container_started_at_unix": container_started_at,
        "ready_at_unix": time.time(),
        "ready_record": ready_record,
    }


def _remove_stale() -> None:
    subprocess.run(
        ["docker", "rm", "--force", CONTAINER_NAME],
        shell=False, check=False, capture_output=True, text=True, timeout=20,
    )


def _inspect_running() -> bool:
    completed = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER_NAME],
        shell=False, check=False, capture_output=True, text=True, timeout=5,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def _contention_gpu_isolated(expected_uuid: str) -> bool:
    """Validate Docker's enforced DeviceRequest, not unrelated GPU-0 activity."""

    completed = subprocess.run(
        [
            "docker", "inspect", "--format",
            "{{json .HostConfig.DeviceRequests}}", CONTAINER_NAME,
        ],
        shell=False, check=False, capture_output=True, text=True, timeout=5,
    )
    if completed.returncode != 0:
        return False
    try:
        requests = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False
    device_ids = {
        str(device_id)
        for request in (requests or [])
        for device_id in (request.get("DeviceIDs") or [])
    }
    return device_ids == {expected_uuid}


def _gpu_identity(*, tier: str = "site") -> dict[str, Any]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        shell=False, check=False, capture_output=True, text=True, timeout=5,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError("CUDA/NVML GPU identity is unavailable")
    rows = []
    for line in completed.stdout.splitlines():
        index, uuid = (item.strip() for item in line.split(",", 1))
        rows.append({"index": int(index), "uuid": uuid})
    env_name = "FABLE_SITE_GPU_UUID" if tier == "site" else "FABLE_DEVICE_GPU_UUID"
    expected_index = 1 if tier == "site" else 0
    requested = os.environ.get(env_name)
    selected = next(
        (row for row in rows if row["uuid"] == requested), None
    ) if requested else next(
        (row for row in rows if row["index"] == expected_index), None
    )
    if selected is None:
        raise RuntimeError(f"{tier} GPU is unavailable ({env_name}, index {expected_index})")
    return selected


def _gpu_samples(uuid: str) -> list[dict[str, float]]:
    completed = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        shell=False, check=False, capture_output=True, text=True, timeout=5,
    )
    if completed.returncode != 0:
        return []
    rows = []
    for line in completed.stdout.splitlines():
        try:
            index, row_uuid, utilization, used, total = (item.strip() for item in line.split(","))
        except (TypeError, ValueError):
            continue
        if row_uuid != uuid:
            continue
        rows.append({
            "gpu_index": int(index),
            "gpu_utilization_percent": float(utilization),
            "gpu_memory_fraction": float(used) / float(total) if float(total) > 0 else 0.0,
        })
    return rows


def _ready_record() -> dict[str, Any] | None:
    completed = subprocess.run(
        ["docker", "logs", CONTAINER_NAME], shell=False, check=False,
        capture_output=True, text=True, timeout=5,
    )
    for line in completed.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "READY":
            return row
    return None


def _benchmark(uuid: str) -> dict[str, float]:
    completed = subprocess.run(
        [
            "docker", "run", "--rm", "--network", "none",
            "--gpus", f"device={uuid}",
            "--cpus", "2.0", "--memory", "8g", "--env", "YOLO_DEVICE=0",
            "--env", "FABLE_CONTENTION_MODE=benchmark",
            "--env", "FABLE_CONTENTION_BATCH_SIZE=8",
            "--env", "FABLE_BENCHMARK_ITERATIONS=80",
            IMAGE, "python3", "-u", "/app/rq3a_contention.py",
        ], shell=False, check=False, capture_output=True, text=True, timeout=90,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "provider benchmark failed")
    for line in reversed(completed.stdout.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "BENCHMARK":
            return {"p50_ms": float(row["p50_ms"]), "p95_ms": float(row["p95_ms"])}
    raise RuntimeError("provider benchmark emitted no result")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _mean_utilization(samples: list[dict[str, float]]) -> float | None:
    values = [row["gpu_utilization_percent"] for row in samples]
    return sum(values) / len(values) if values else None
