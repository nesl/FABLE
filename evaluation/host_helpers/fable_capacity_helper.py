#!/usr/bin/env python3
"""Root-installed, fixed-policy cgroup-v1 helper for FABLE evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


CPU_ROOT = Path("/sys/fs/cgroup/cpu,cpuacct")
MEMORY_ROOT = Path("/sys/fs/cgroup/memory")
GROUP_RELATIVE = Path("fable-evaluation/x86server")
STATE_PATH = Path("/run/fable-evaluation/cgroup-state.json")
ALLOWED_CONTAINERS = (
    "fable-orchestrator",
    "fable-agent-x86server",
    "fable-identity-x86server",
    "fable-evaluation-logger",
    "complex-event-detector",
)
CPU_PERIOD_US = 100_000
E1_CPU_QUOTA_US = 600_000
E1_MEMORY_MAX_BYTES = 48 * 1024**3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("CAPACITY_PROFILE",), required=True)
    parser.add_argument("--target", choices=("x86server",), required=True)
    parser.add_argument("--condition", choices=("E1", "N0"), required=True)
    parser.add_argument("--action", choices=("APPLY", "RESTORE"), required=True)
    parser.add_argument("--condition-epoch", type=int, required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error("capacity helper must run as root")
    if args.condition_epoch < 0:
        parser.error("condition epoch cannot be negative")
    if not CPU_ROOT.is_dir() or not MEMORY_ROOT.is_dir():
        raise RuntimeError("required cgroup-v1 CPU or memory controller is absent")

    if args.action == "APPLY":
        if args.condition != "E1":
            parser.error("APPLY accepts only E1")
        measurements = apply_e1()
    else:
        if args.condition != "N0":
            parser.error("RESTORE accepts only N0")
        measurements = restore_n0()
    print(
        json.dumps(
            {
                "schema_version": "fable.capacity_control_result.v1",
                "validated": True,
                "condition": args.condition,
                "condition_epoch": args.condition_epoch,
                "measurements": measurements,
                "reason": "fixed evaluation cgroup state applied and read back",
            },
            sort_keys=True,
        )
    )
    return 0


def apply_e1() -> dict:
    pids = _running_container_pids()
    if not pids:
        raise RuntimeError("no allowlisted x86 evaluation containers are running")
    originals = {
        str(pid): {
            "cpu": _current_cgroup(pid, "cpu"),
            "memory": _current_cgroup(pid, "memory"),
        }
        for pid in pids
    }
    _write_state({"schema_version": "fable.capacity_state.v1", "pids": originals})
    cpu_group = CPU_ROOT / GROUP_RELATIVE
    memory_group = MEMORY_ROOT / GROUP_RELATIVE
    cpu_group.mkdir(parents=True, exist_ok=True)
    memory_group.mkdir(parents=True, exist_ok=True)
    _write(cpu_group / "cpu.cfs_period_us", str(CPU_PERIOD_US))
    _write(cpu_group / "cpu.cfs_quota_us", str(E1_CPU_QUOTA_US))
    _write(memory_group / "memory.limit_in_bytes", str(E1_MEMORY_MAX_BYTES))
    for pid in pids:
        _write(cpu_group / "cgroup.procs", str(pid))
        _write(memory_group / "cgroup.procs", str(pid))
    return _readback("E1", pids)


def restore_n0() -> dict:
    state = _read_state()
    restored = []
    for pid_text, controllers in state.get("pids", {}).items():
        pid = int(pid_text)
        if not Path(f"/proc/{pid}").exists():
            continue
        _restore_pid(pid, CPU_ROOT, str(controllers["cpu"]))
        _restore_pid(pid, MEMORY_ROOT, str(controllers["memory"]))
        restored.append(pid)
    cpu_group = CPU_ROOT / GROUP_RELATIVE
    memory_group = MEMORY_ROOT / GROUP_RELATIVE
    if cpu_group.is_dir():
        _write(cpu_group / "cpu.cfs_period_us", str(CPU_PERIOD_US))
        _write(cpu_group / "cpu.cfs_quota_us", "-1")
    if memory_group.is_dir():
        _write(memory_group / "memory.limit_in_bytes", "-1")
    STATE_PATH.unlink(missing_ok=True)
    return _readback("N0", restored)


def _running_container_pids() -> list[int]:
    pids = []
    for name in ALLOWED_CONTAINERS:
        completed = subprocess.run(
            (
                "/usr/bin/docker",
                "inspect",
                "--format",
                "{{.State.Pid}} {{.State.Running}}",
                name,
            ),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode != 0:
            continue
        parts = completed.stdout.strip().split()
        if len(parts) == 2 and parts[1].lower() == "true" and int(parts[0]) > 1:
            pids.append(int(parts[0]))
    return pids


def _current_cgroup(pid: int, controller: str) -> str:
    for line in Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines():
        _, controllers, relative = line.split(":", 2)
        if controller in controllers.split(","):
            return relative
    raise RuntimeError(f"PID {pid} has no {controller} cgroup")


def _restore_pid(pid: int, root: Path, relative: str) -> None:
    if not relative.startswith("/"):
        raise RuntimeError("recorded cgroup path is not absolute")
    target = (root / relative.lstrip("/")).resolve(strict=True)
    if root.resolve() not in (target, *target.parents):
        raise RuntimeError("recorded cgroup path escapes controller root")
    _write(target / "cgroup.procs", str(pid))


def _readback(condition: str, pids: list[int]) -> dict:
    cpu_group = CPU_ROOT / GROUP_RELATIVE
    memory_group = MEMORY_ROOT / GROUP_RELATIVE
    quota = int((cpu_group / "cpu.cfs_quota_us").read_text().strip())
    period = int((cpu_group / "cpu.cfs_period_us").read_text().strip())
    memory = int((memory_group / "memory.limit_in_bytes").read_text().strip())
    expected_quota = E1_CPU_QUOTA_US if condition == "E1" else -1
    if quota != expected_quota or period != CPU_PERIOD_US:
        raise RuntimeError("CPU cgroup readback mismatch")
    if condition == "E1" and memory != E1_MEMORY_MAX_BYTES:
        raise RuntimeError("memory cgroup readback mismatch")
    return {
        "cgroup_version": 1,
        "profile_id": condition,
        "cpu_quota_us": quota,
        "cpu_period_us": period,
        "cpu_capacity_cores": quota / period if quota > 0 else -1,
        "memory_max_bytes": memory,
        "moved_pid_count": len(pids),
        "allowlisted_container_count": len(ALLOWED_CONTAINERS),
    }


def _write(path: Path, value: str) -> None:
    path.write_text(value + "\n", encoding="ascii")


def _write_state(document: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(STATE_PATH)


def _read_state() -> dict:
    if not STATE_PATH.is_file():
        raise RuntimeError("capacity restore state is absent")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if state.get("schema_version") != "fable.capacity_state.v1":
        raise RuntimeError("capacity restore state schema mismatch")
    return state


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": "fable.capacity_control_result.v1",
                    "validated": False,
                    "measurements": {},
                    "reason": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            )
        )
        raise SystemExit(4)
