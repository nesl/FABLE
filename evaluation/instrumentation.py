"""Low-overhead host resource samples independent of provider semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
from typing import Callable


@dataclass(frozen=True)
class ResourceSample:
    sampled_at: str
    process_id: int
    process_cpu_seconds: float
    process_rss_bytes: int
    host_tx_bytes: int
    host_rx_bytes: int
    gpu_utilization_percent: float | None = None
    gpu_memory_used_bytes: int | None = None
    gpu_power_watts: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _process_sample(pid: int) -> tuple[float, int]:
    stat = Path(f"/proc/{pid}/stat").read_text().split()
    ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    cpu = (int(stat[13]) + int(stat[14])) / ticks
    pages = int(stat[23])
    return cpu, pages * os.sysconf("SC_PAGE_SIZE")


def _network_bytes() -> tuple[int, int]:
    tx = rx = 0
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        interface, values = line.split(":", 1)
        if interface.strip() == "lo":
            continue
        fields = values.split()
        rx += int(fields[0])
        tx += int(fields[8])
    return tx, rx


def _nvml_sample(device_index: int) -> tuple[float, int, float] | None:
    try:
        import pynvml
    except ImportError:
        return None
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        utilization = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        memory = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
        power = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        return utilization, memory, power
    finally:
        pynvml.nvmlShutdown()


def sample_resources(pid: int | None = None, *, gpu_index: int = 0) -> ResourceSample:
    target = pid or os.getpid()
    cpu, rss = _process_sample(target)
    tx, rx = _network_bytes()
    gpu = _nvml_sample(gpu_index)
    return ResourceSample(
        datetime.now(timezone.utc).isoformat(),
        target,
        cpu,
        rss,
        tx,
        rx,
        None if gpu is None else gpu[0],
        None if gpu is None else gpu[1],
        None if gpu is None else gpu[2],
    )


def resource_delta(start: ResourceSample, end: ResourceSample) -> dict[str, float | int | None]:
    return {
        "process_cpu_seconds": max(0.0, end.process_cpu_seconds - start.process_cpu_seconds),
        "tx_bytes": max(0, end.host_tx_bytes - start.host_tx_bytes),
        "rx_bytes": max(0, end.host_rx_bytes - start.host_rx_bytes),
        "peak_process_rss_bytes": max(start.process_rss_bytes, end.process_rss_bytes),
        "gpu_utilization_percent_end": end.gpu_utilization_percent,
        "gpu_memory_used_bytes_end": end.gpu_memory_used_bytes,
        "gpu_power_watts_end": end.gpu_power_watts,
    }
