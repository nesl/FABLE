"""Run-scoped measured resource instrumentation for live Docker evaluations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import io
import os
from pathlib import Path
from subprocess import run
import threading
import time
from time import perf_counter_ns

from evaluation.runner import JsonlEventStore
from evaluation.schemas import BaselineId, ResourceSample


@dataclass(frozen=True)
class _ContainerCounters:
    container_id: str
    name: str
    pid: int
    cpu_nanoseconds: int
    cpu_throttled_nanoseconds: int
    memory_bytes: int
    network_namespace: str
    rx_bytes: int
    tx_bytes: int


def _container_category(name: str) -> str:
    if name == "fable-gpu-contention" or name.startswith("fable-disturbance-"):
        return "disturbance_workload"
    if name.startswith("netwaggle-node-") or name in {"iobt-minimal-mqtt"}:
        return "infrastructure"
    if "replay" in name:
        return "replay_source"
    return "evaluated_system"


def _read_int(paths: tuple[Path, ...]) -> int:
    for path in paths:
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
    return 0


def _unified_cgroup_path(pid: int) -> Path | None:
    """Return a process' cgroup-v2 directory, when the host uses cgroup v2."""

    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                return Path("/sys/fs/cgroup") / relative.lstrip("/")
    except (OSError, ValueError):
        pass
    return None


def _cpu_stat(path: Path | None) -> tuple[int, int]:
    if path is None:
        return 0, 0
    try:
        values = {}
        for line in (path / "cpu.stat").read_text(encoding="utf-8").splitlines():
            key, value = line.split(None, 1)
            values[key] = int(value)
        return values.get("usage_usec", 0) * 1_000, values.get("throttled_usec", 0) * 1_000
    except (OSError, ValueError):
        return 0, 0


def _network_counters(pid: int) -> tuple[int, int]:
    try:
        lines = Path(f"/proc/{pid}/net/dev").read_text(encoding="utf-8").splitlines()[2:]
    except OSError:
        return 0, 0
    rx = tx = 0
    for line in lines:
        if ":" not in line:
            continue
        interface, values = line.split(":", 1)
        if interface.strip() == "lo":
            continue
        fields = values.split()
        if len(fields) >= 9:
            rx += int(fields[0])
            tx += int(fields[8])
    return rx, tx


def _containers() -> tuple[_ContainerCounters, ...]:
    ids = run(
        ("docker", "ps", "-q"), shell=False, check=False, capture_output=True, text=True
    ).stdout.split()
    if not ids:
        return ()
    completed = run(
        (
            "docker", "inspect", "--format",
            "{{.Id}} {{.Name}} {{.State.Pid}} {{.HostConfig.NetworkMode}}", *ids,
        ),
        shell=False, check=False, capture_output=True, text=True,
    )
    raw_rows = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        container_id, name, pid_text, network_mode = fields
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid <= 0:
            continue
        name = name.lstrip("/")
        unified = _unified_cgroup_path(pid)
        cpu_ns, throttled_ns = _cpu_stat(unified)
        cpu_ns = cpu_ns or _read_int(
            (
                Path(f"/sys/fs/cgroup/cpu,cpuacct/docker/{container_id}/cpuacct.usage"),
                Path(f"/sys/fs/cgroup/cpuacct/docker/{container_id}/cpuacct.usage"),
            )
        )
        memory = _read_int(
            (
                *((unified / "memory.current",) if unified is not None else ()),
                Path(f"/sys/fs/cgroup/memory/docker/{container_id}/memory.usage_in_bytes"),
                Path(f"/sys/fs/cgroup/unified/docker/{container_id}/memory.current"),
            )
        )
        rx, tx = _network_counters(pid)
        raw_rows.append((container_id, name, pid, cpu_ns, throttled_ns, memory, network_mode, rx, tx))
    identifiers = {container_id: container_id for container_id, *_ in raw_rows}
    identifiers.update({name.lstrip("/"): container_id for container_id, name, *_ in raw_rows})
    rows = []
    for container_id, name, pid, cpu_ns, throttled_ns, memory, network_mode, rx, tx in raw_rows:
        if network_mode.startswith("container:"):
            reference = network_mode.split(":", 1)[1]
            namespace = identifiers.get(reference)
            if namespace is None:
                namespace = next(
                    (full for key, full in identifiers.items() if full.startswith(reference)),
                    reference,
                )
        elif network_mode == "host":
            namespace = "host"
        else:
            namespace = container_id
        rows.append(_ContainerCounters(
            container_id, name, pid, cpu_ns, throttled_ns, memory, namespace, rx, tx
        ))
    return tuple(rows)


def _gpu_process_rows() -> tuple[dict[str, int | str], ...]:
    """Return CUDA process VRAM allocations without treating them as GPU work."""

    completed = run(
        (
            "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
        shell=False, check=False, capture_output=True, text=True, timeout=5,
    )
    if completed.returncode != 0:
        return ()
    rows = []
    for fields in csv.reader(io.StringIO(completed.stdout)):
        if len(fields) != 3:
            continue
        try:
            rows.append({
                "pid": int(fields[0].strip()),
                "gpu_uuid": fields[1].strip(),
                "memory_bytes": int(float(fields[2].strip()) * 1024 * 1024),
            })
        except ValueError:
            continue
    return tuple(rows)


def _container_for_pid(pid: int, containers: tuple[_ContainerCounters, ...]) -> _ContainerCounters | None:
    try:
        membership = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
    except OSError:
        return None
    return next((row for row in containers if row.container_id in membership), None)


def _gpu_rows() -> tuple[dict[str, float | int | str], ...]:
    completed = run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ),
        shell=False, check=False, capture_output=True, text=True, timeout=5,
    )
    if completed.returncode != 0:
        return ()
    rows = []
    for fields in csv.reader(io.StringIO(completed.stdout)):
        if len(fields) != 5:
            continue
        try:
            rows.append(
                {
                    "index": int(fields[0].strip()),
                    "uuid": fields[1].strip(),
                    "utilization": float(fields[2].strip()) / 100.0,
                    "memory_bytes": int(float(fields[3].strip()) * 1024 * 1024),
                    "power_watts": float(fields[4].strip()),
                }
            )
        except ValueError:
            continue
    return tuple(rows)


class RunResourceMonitor:
    """Sample Docker cgroups/namespaces and host GPUs into common records."""

    def __init__(
        self,
        *,
        store: JsonlEventStore,
        run_id: str,
        baseline_id: BaselineId,
        trace_id: str,
        request_id: str,
        interval_seconds: float = 1.0,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.baseline_id = baseline_id
        self.trace_id = trace_id
        self.request_id = request_id
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._previous: dict[str, tuple[float, _ContainerCounters]] = {}
        self._previous_network: dict[str, tuple[float, int, int]] = {}
        self.errors: list[str] = []
        self.sample_count = 0
        self.totals = {
            "evaluated_system_cpu_seconds": 0.0,
            "replay_source_cpu_seconds": 0.0,
            "infrastructure_cpu_seconds": 0.0,
            "disturbance_workload_cpu_seconds": 0.0,
            "unique_network_namespace_tx_bytes": 0,
            "unique_network_namespace_rx_bytes": 0,
            "evaluated_path_network_tx_bytes": 0,
            "evaluated_path_network_rx_bytes": 0,
            "replay_only_network_tx_bytes": 0,
            "replay_only_network_rx_bytes": 0,
            "infrastructure_network_tx_bytes": 0,
            "infrastructure_network_rx_bytes": 0,
            "disturbance_workload_network_tx_bytes": 0,
            "disturbance_workload_network_rx_bytes": 0,
            "host_gpu_seconds": 0.0,
            "host_gpu_energy_joules": 0.0,
            "device_tier_host_gpu_seconds": 0.0,
            "device_tier_host_gpu_energy_joules": 0.0,
            "site_local_host_gpu_seconds": 0.0,
            "site_local_host_gpu_energy_joules": 0.0,
            "provider_gpu_memory_byte_seconds": 0.0,
            "disturbance_gpu_memory_byte_seconds": 0.0,
            "peak_provider_gpu_memory_bytes": 0,
            "evaluated_system_cpu_throttled_seconds": 0.0,
        }

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="evaluation-resource-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, self.interval_seconds * 3))

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": "fable.run_resource_instrumentation.v1",
            "sample_count": self.sample_count,
            "interval_seconds": self.interval_seconds,
            "errors": self.errors[-20:],
            "container_scope": "all running containers, categorized in record metadata",
            "gpu_scope": (
                "host GPU totals split by physical tier; process-level GPU metrics "
                "measure VRAM residency, not SM busy time"
            ),
            "counter_semantics": "interval deltas; summable across records",
            "totals": {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in self.totals.items()
            },
        }

    def _append(self, **values: object) -> None:
        self.store.append(
            ResourceSample(
                run_id=self.run_id,
                baseline_id=self.baseline_id,
                trace_id=self.trace_id,
                request_id=self.request_id,
                sensor_id=str(values["node_id"]),
                event_time=datetime.now(UTC),
                monotonic_timestamp_ns=perf_counter_ns(),
                **values,
            )
        )
        self.sample_count += 1

    def _run(self) -> None:
        last_gpu_at = time.monotonic()
        while not self._stop.is_set():
            sampled_at = time.monotonic()
            try:
                current = _containers()
                namespaces: dict[str, list[_ContainerCounters]] = {}
                for row in current:
                    if row.network_namespace:
                        namespaces.setdefault(row.network_namespace, []).append(row)
                    previous = self._previous.get(row.container_id)
                    self._previous[row.container_id] = (sampled_at, row)
                    if previous is None:
                        continue
                    previous_at, old = previous
                    elapsed = max(1e-9, sampled_at - previous_at)
                    cpu_seconds = max(0.0, (row.cpu_nanoseconds - old.cpu_nanoseconds) / 1e9)
                    throttled_seconds = max(
                        0.0,
                        (row.cpu_throttled_nanoseconds - old.cpu_throttled_nanoseconds) / 1e9,
                    )
                    category = _container_category(row.name)
                    self.totals[f"{category}_cpu_seconds"] += cpu_seconds
                    if category == "evaluated_system":
                        self.totals["evaluated_system_cpu_throttled_seconds"] += throttled_seconds
                    self._append(
                        node_id=f"container:{row.name}",
                        cpu_utilization=min(1.0, cpu_seconds / elapsed / max(1, os.cpu_count() or 1)),
                        cpu_time_seconds=cpu_seconds,
                        memory_bytes=max(0, row.memory_bytes),
                        network_rx_bytes=0,
                        network_tx_bytes=0,
                        metadata={
                            "measurement_kind": "docker_cgroup_interval_delta",
                            "attribution": "run_measurement_window",
                            "container_id": row.container_id,
                            "container_name": row.name,
                            "container_category": category,
                            "interval_seconds": elapsed,
                            "cpu_throttled_seconds": throttled_seconds,
                            "cgroup_version": 2 if _unified_cgroup_path(row.pid) else 1,
                        },
                    )
                for namespace, members in namespaces.items():
                    representative = members[0]
                    previous_network = self._previous_network.get(namespace)
                    self._previous_network[namespace] = (
                        sampled_at, representative.rx_bytes, representative.tx_bytes
                    )
                    if previous_network is None:
                        continue
                    previous_at, old_rx, old_tx = previous_network
                    elapsed = max(1e-9, sampled_at - previous_at)
                    rx_delta = max(0, representative.rx_bytes - old_rx)
                    tx_delta = max(0, representative.tx_bytes - old_tx)
                    member_categories = {_container_category(item.name) for item in members}
                    if "evaluated_system" in member_categories:
                        network_scope = "evaluated_path"
                    elif "replay_source" in member_categories:
                        network_scope = "replay_only"
                    elif "disturbance_workload" in member_categories:
                        network_scope = "disturbance_workload"
                    else:
                        network_scope = "infrastructure"
                    self.totals["unique_network_namespace_rx_bytes"] += rx_delta
                    self.totals["unique_network_namespace_tx_bytes"] += tx_delta
                    self.totals[f"{network_scope}_network_rx_bytes"] += rx_delta
                    self.totals[f"{network_scope}_network_tx_bytes"] += tx_delta
                    self._append(
                        node_id=f"netns:{namespace}",
                        cpu_utilization=0.0,
                        cpu_time_seconds=0.0,
                        memory_bytes=0,
                        network_rx_bytes=rx_delta,
                        network_tx_bytes=tx_delta,
                        metadata={
                            "measurement_kind": "network_namespace_interval_delta",
                            "attribution": "shared_emulated_node_total",
                            "network_namespace": namespace,
                            "member_containers": sorted(item.name for item in members),
                            "member_categories": sorted(member_categories),
                            "network_scope": network_scope,
                            "interval_seconds": elapsed,
                        },
                    )
                gpu_elapsed = max(1e-9, sampled_at - last_gpu_at)
                last_gpu_at = sampled_at
                gpu_processes = tuple(
                    (process, _container_for_pid(int(process["pid"]), current))
                    for process in _gpu_process_rows()
                )
                concurrent_provider_memory = sum(
                    int(process["memory_bytes"])
                    for process, container in gpu_processes
                    if container is not None
                    and _container_category(container.name) == "evaluated_system"
                )
                self.totals["peak_provider_gpu_memory_bytes"] = max(
                    self.totals["peak_provider_gpu_memory_bytes"],
                    concurrent_provider_memory,
                )
                for process, container in gpu_processes:
                    if container is None:
                        continue
                    memory_bytes = int(process["memory_bytes"])
                    memory_byte_seconds = memory_bytes * gpu_elapsed
                    category = _container_category(container.name)
                    if category == "evaluated_system":
                        self.totals["provider_gpu_memory_byte_seconds"] += memory_byte_seconds
                    elif category == "disturbance_workload":
                        self.totals["disturbance_gpu_memory_byte_seconds"] += memory_byte_seconds
                    self._append(
                        node_id=f"gpu-process:{container.name}:{process['pid']}",
                        cpu_utilization=0.0,
                        cpu_time_seconds=0.0,
                        memory_bytes=0,
                        gpu_utilization=0.0,
                        gpu_memory_bytes=memory_bytes,
                        gpu_time_seconds=0.0,
                        gpu_energy_joules=0.0,
                        network_rx_bytes=0,
                        network_tx_bytes=0,
                        metadata={
                            "measurement_kind": "nvidia_smi_process_gpu_residency",
                            "attribution": "cuda_process_to_container",
                            "container_id": container.container_id,
                            "container_name": container.name,
                            "container_category": category,
                            "host_pid": process["pid"],
                            "gpu_uuid": process["gpu_uuid"],
                            "memory_byte_seconds": memory_byte_seconds,
                            "concurrent_provider_gpu_memory_bytes": concurrent_provider_memory,
                            "interval_seconds": gpu_elapsed,
                            "note": "VRAM residency; zero GPU busy time is intentional",
                        },
                    )
                for gpu in _gpu_rows():
                    utilization = float(gpu["utilization"])
                    power = float(gpu["power_watts"])
                    gpu_seconds = utilization * gpu_elapsed
                    gpu_energy = power * gpu_elapsed
                    self.totals["host_gpu_seconds"] += gpu_seconds
                    self.totals["host_gpu_energy_joules"] += gpu_energy
                    gpu_index = int(gpu["index"])
                    tier = "device_tier" if gpu_index == 0 else "site_local" if gpu_index == 1 else "other"
                    if tier != "other":
                        self.totals[f"{tier}_host_gpu_seconds"] += gpu_seconds
                        self.totals[f"{tier}_host_gpu_energy_joules"] += gpu_energy
                    self._append(
                        node_id=f"gpu:{gpu['uuid']}",
                        cpu_utilization=0.0,
                        cpu_time_seconds=0.0,
                        memory_bytes=0,
                        gpu_utilization=utilization,
                        gpu_memory_bytes=int(gpu["memory_bytes"]),
                        gpu_time_seconds=gpu_seconds,
                        gpu_energy_joules=gpu_energy,
                        network_rx_bytes=0,
                        network_tx_bytes=0,
                        metadata={
                            "measurement_kind": "nvidia_smi_host_gpu_interval",
                            "attribution": "shared_host_gpu_total",
                            "gpu_uuid": gpu["uuid"],
                            "gpu_index": gpu_index,
                            "gpu_tier": tier,
                            "gpu_power_watts": power,
                            "interval_seconds": gpu_elapsed,
                        },
                    )
            except Exception as exc:  # Instrumentation must not kill a CE run.
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self._stop.wait(max(0.05, self.interval_seconds - (time.monotonic() - sampled_at)))
