from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from evaluation.host_validation import (
    CgroupExpectation,
    NetworkPathProbe,
    validate_cgroup_state,
    validate_network_path,
)


def test_network_validation_uses_fixed_bounded_argv() -> None:
    calls = []

    def runner(argv, timeout):
        calls.append((argv, timeout))
        if argv[3] == "ping":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=(
                    "3 packets transmitted, 3 received, 0% packet loss\n"
                    "rtt min/avg/max/mdev = 49.0/51.5/53.0/1.0 ms\n"
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                {"end": {"sum_received": {"bits_per_second": 120_000_000}}}
            ),
            stderr="",
        )

    result = validate_network_path(
        NetworkPathProbe(
            source_container="netwaggle-node-site-local",
            destination_ip="10.255.250.2",
            maximum_average_rtt_ms=60,
            minimum_throughput_mbps=100,
        ),
        command_runner=runner,
    )

    assert result["path_validated"]
    assert result["average_rtt_ms"] == 51.5
    assert result["throughput_mbps"] == 120
    assert calls[0][0] == (
        "docker",
        "exec",
        "netwaggle-node-site-local",
        "ping",
        "-n",
        "-c",
        "3",
        "-W",
        "2",
        "10.255.250.2",
    )
    assert calls[1][0][3] == "iperf3"
    assert all(timeout <= 11 for _, timeout in calls)


def test_network_validation_fails_closed_on_measured_bounds() -> None:
    def runner(argv, _timeout):
        if argv[3] == "ping":
            stdout = "rtt min/avg/max/mdev = 99/100/101/1 ms\n"
        else:
            stdout = json.dumps(
                {"end": {"sum_received": {"bits_per_second": 1_000_000}}}
            )
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    with pytest.raises(RuntimeError, match="validation bounds"):
        validate_network_path(
            NetworkPathProbe(
                source_container="anchor",
                destination_ip="10.255.0.1",
                maximum_average_rtt_ms=50,
                minimum_throughput_mbps=10,
            ),
            command_runner=runner,
        )


def test_cgroup_state_is_read_back_from_direct_child(tmp_path: Path) -> None:
    target = tmp_path / "site_local"
    target.mkdir()
    (target / "cpu.max").write_text("300000 100000\n", encoding="utf-8")
    (target / "memory.max").write_text("4294967296\n", encoding="utf-8")

    result = validate_cgroup_state(
        CgroupExpectation(
            cgroup_name="site_local",
            cpu_quota_us=300_000,
            memory_max_bytes=4_294_967_296,
        ),
        cgroup_root=tmp_path,
    )

    assert result["cgroup_validated"]
    assert result["cpu_capacity_cores"] == 3


def test_cgroup_state_rejects_mismatch_and_escape(tmp_path: Path) -> None:
    target = tmp_path / "site_local"
    target.mkdir()
    (target / "cpu.max").write_text("300000 100000\n", encoding="utf-8")
    (target / "memory.max").write_text("max\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="CPU quota"):
        validate_cgroup_state(
            CgroupExpectation(cgroup_name="site_local", cpu_quota_us=100_000),
            cgroup_root=tmp_path,
        )
    with pytest.raises(ValueError):
        CgroupExpectation(cgroup_name="../outside")
