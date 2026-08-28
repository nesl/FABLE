"""Bounded, read-only validation of applied network and compute conditions."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Callable

from pydantic import Field, model_validator

from fable.common.base import FrozenFableModel


_PING_RTT = re.compile(
    r"(?:rtt|round-trip) min/avg/max(?:/(?:mdev|stddev))? = "
    r"[0-9.]+/(?P<average>[0-9.]+)/[0-9.]+(?:/[0-9.]+)? ms"
)


class NetworkPathProbe(FrozenFableModel):
    source_container: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    destination_ip: str = Field(
        pattern=r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
    )
    maximum_average_rtt_ms: float = Field(gt=0)
    minimum_throughput_mbps: float = Field(gt=0)
    ping_count: int = Field(default=3, ge=1, le=10)
    iperf_seconds: int = Field(default=2, ge=1, le=10)

    @model_validator(mode="after")
    def _validate_ip_octets(self):
        if any(int(part) > 255 for part in self.destination_ip.split(".")):
            raise ValueError("destination_ip contains an invalid octet")
        return self


CommandRunner = Callable[[tuple[str, ...], float], subprocess.CompletedProcess[str]]


def _run_command(
    argv: tuple[str, ...],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def validate_network_path(
    probe: NetworkPathProbe,
    *,
    command_runner: CommandRunner = _run_command,
) -> dict[str, int | float | str | bool]:
    """Run fixed ping/iperf3 clients inside an allowlisted anchor container."""

    ping_argv = (
        "docker",
        "exec",
        probe.source_container,
        "ping",
        "-n",
        "-c",
        str(probe.ping_count),
        "-W",
        "2",
        probe.destination_ip,
    )
    ping = command_runner(ping_argv, float(3 + probe.ping_count * 2))
    if ping.returncode != 0:
        raise RuntimeError(
            f"ping validation failed with status {ping.returncode}"
        )
    match = _PING_RTT.search(ping.stdout)
    if match is None:
        raise RuntimeError("ping output did not contain an RTT summary")
    average_rtt_ms = float(match.group("average"))

    iperf_argv = (
        "docker",
        "exec",
        probe.source_container,
        "iperf3",
        "--client",
        probe.destination_ip,
        "--json",
        "--time",
        str(probe.iperf_seconds),
    )
    iperf = command_runner(iperf_argv, float(5 + probe.iperf_seconds))
    if iperf.returncode != 0:
        raise RuntimeError(
            f"iperf3 validation failed with status {iperf.returncode}"
        )
    try:
        document = json.loads(iperf.stdout)
        bits_per_second = float(document["end"]["sum_received"]["bits_per_second"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("iperf3 output lacks received throughput") from exc
    throughput_mbps = bits_per_second / 1_000_000
    rtt_validated = average_rtt_ms <= probe.maximum_average_rtt_ms
    throughput_validated = throughput_mbps >= probe.minimum_throughput_mbps
    if not rtt_validated or not throughput_validated:
        raise RuntimeError(
            "network path does not satisfy the configured validation bounds"
        )
    return {
        "path_validated": True,
        "source_container": probe.source_container,
        "destination_ip": probe.destination_ip,
        "average_rtt_ms": average_rtt_ms,
        "throughput_mbps": throughput_mbps,
        "ping_samples": probe.ping_count,
        "iperf_seconds": probe.iperf_seconds,
    }


class CgroupExpectation(FrozenFableModel):
    cgroup_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    cpu_quota_us: int | None = Field(default=None, gt=0)
    cpu_period_us: int = Field(default=100_000, gt=0)
    memory_max_bytes: int | None = Field(default=None, gt=0)


def validate_cgroup_state(
    expectation: CgroupExpectation,
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, int | float | str | bool]:
    """Read a fixed evaluation cgroup on either cgroup v1 or v2."""

    root = cgroup_root.resolve(strict=True)
    direct = root / expectation.cgroup_name
    nested = root / "fable-evaluation" / expectation.cgroup_name
    if (direct / "cpu.max").is_file():
        target = direct.resolve(strict=True)
        if target.parent != root:
            raise RuntimeError("cgroup target escapes or is not a direct child")
        cpu_parts = (target / "cpu.max").read_text().strip().split()
        if len(cpu_parts) != 2:
            raise RuntimeError("invalid cpu.max state")
        quota_text, period_text = cpu_parts
        observed_period = int(period_text)
        observed_quota = None if quota_text == "max" else int(quota_text)
        memory_text = (target / "memory.max").read_text().strip()
        observed_memory = None if memory_text == "max" else int(memory_text)
        cgroup_version = 2
    elif (nested / "cpu.max").is_file():
        target = nested.resolve(strict=True)
        if target.parent != (root / "fable-evaluation").resolve(strict=True):
            raise RuntimeError("cgroup target escapes evaluation root")
        cpu_parts = (target / "cpu.max").read_text().strip().split()
        quota_text, period_text = cpu_parts
        observed_period = int(period_text)
        observed_quota = None if quota_text == "max" else int(quota_text)
        memory_text = (target / "memory.max").read_text().strip()
        observed_memory = None if memory_text == "max" else int(memory_text)
        cgroup_version = 2
    else:
        cpu_target = (
            root
            / "cpu,cpuacct"
            / "fable-evaluation"
            / expectation.cgroup_name
        ).resolve(strict=True)
        memory_target = (
            root / "memory" / "fable-evaluation" / expectation.cgroup_name
        ).resolve(strict=True)
        expected_cpu_parent = (
            root / "cpu,cpuacct" / "fable-evaluation"
        ).resolve(strict=True)
        expected_memory_parent = (
            root / "memory" / "fable-evaluation"
        ).resolve(strict=True)
        if (
            cpu_target.parent != expected_cpu_parent
            or memory_target.parent != expected_memory_parent
        ):
            raise RuntimeError("cgroup-v1 target escapes evaluation roots")
        observed_period = int(
            (cpu_target / "cpu.cfs_period_us").read_text().strip()
        )
        quota = int((cpu_target / "cpu.cfs_quota_us").read_text().strip())
        observed_quota = None if quota < 0 else quota
        memory = int(
            (memory_target / "memory.limit_in_bytes").read_text().strip()
        )
        observed_memory = None if memory >= 2**60 else memory
        cgroup_version = 1

    if observed_period != expectation.cpu_period_us:
        raise RuntimeError("cgroup CPU period does not match expectation")
    if observed_quota != expectation.cpu_quota_us:
        raise RuntimeError("cgroup CPU quota does not match expectation")
    if observed_memory != expectation.memory_max_bytes:
        raise RuntimeError("cgroup memory limit does not match expectation")
    return {
        "cgroup_validated": True,
        "cgroup_version": cgroup_version,
        "cgroup_name": expectation.cgroup_name,
        "cpu_quota_us": observed_quota if observed_quota is not None else -1,
        "cpu_period_us": observed_period,
        "cpu_capacity_cores": (
            observed_quota / observed_period
            if observed_quota is not None
            else -1.0
        ),
        "memory_max_bytes": (
            observed_memory if observed_memory is not None else -1
        ),
    }
