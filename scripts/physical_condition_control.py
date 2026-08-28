#!/usr/bin/env python3
"""Apply bounded allowlisted physical E4 conditions; dry-run by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]
NETWORK_HELPER = "/usr/local/sbin/fable-physical-net"


def parse_compute_clear_output(stdout: str) -> dict:
    """Extract durable workload and tegrastats evidence from remote cleanup."""

    parsed: dict[str, object] = {}
    lines = stdout.splitlines()
    if "CONTENTION_RESULT" in lines:
        index = lines.index("CONTENTION_RESULT") + 1
        if index < len(lines) and lines[index].startswith("{"):
            try:
                parsed["contention_result"] = json.loads(lines[index])
            except json.JSONDecodeError:
                parsed["contention_result"] = None
    if "TEGRASTATS_TAIL" in lines:
        index = lines.index("TEGRASTATS_TAIL") + 1
        parsed["tegrastats_samples"] = [
            line for line in lines[index:] if line.strip()
        ]
    utilization = []
    for line in parsed.get("tegrastats_samples") or ():
        match = re.search(r"\bGR3D_FREQ\s+(\d+(?:\.\d+)?)%", str(line))
        if match:
            utilization.append(float(match.group(1)))
    if utilization:
        ordered = sorted(utilization)
        parsed["gpu_utilization"] = {
            "sample_count": len(utilization),
            "mean_percent": sum(utilization) / len(utilization),
            "p95_percent": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "maximum_percent": max(utilization),
        }
    contention = parsed.get("contention_result") or {}
    gpu = parsed.get("gpu_utilization") or {}
    parsed["measurement_validated"] = bool(
        isinstance(contention, dict)
        and contention.get("iterations", 0) > 0
        and contention.get("active_seconds", 0) > 0
        and gpu.get("sample_count", 0) >= 8
        and 65.0 <= gpu.get("mean_percent", 0.0) <= 95.0
    )
    return parsed


def remote(identity: Path, host: str, command: str, *, execute: bool) -> dict:
    argv = ["ssh", "-i", str(identity), "-o", "BatchMode=yes", host, command]
    if not execute:
        return {"argv": argv, "executed": False}
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    return {
        "argv": argv,
        "executed": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "apply-compute", "clear-compute", "apply-network", "disconnect-network", "restore-network"))
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--target", choices=("physical_jetson", "rpi_to_jetson"), required=True)
    parser.add_argument("--profile", default="N0")
    parser.add_argument("--duration-seconds", type=int, default=45)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.duration_seconds <= 300:
        parser.error("duration must be in [1, 300]")

    config = yaml.safe_load((ROOT / "config/physical_network_profiles.yaml").read_text())
    if args.target == "physical_jetson":
        host = "jetson"
        if args.action == "check":
            command = (
                "test -x /usr/bin/tegrastats && "
                "test -f /home/nesl/FABLE/scripts/physical_jetson_load.py && "
                "/usr/bin/python3 -c 'import torch; assert torch.cuda.is_available()'"
            )
        elif args.action == "apply-compute":
            command = (
                "mkdir -p /home/nesl/FABLE/state /home/nesl/FABLE/logs; "
                "touch /home/nesl/FABLE/state/physical-yolo-relay-only; "
                "rm -f /home/nesl/FABLE/state/physical-contention.ready.json "
                "/home/nesl/FABLE/state/physical-contention.result.json "
                "/home/nesl/FABLE/logs/physical-contention-tegrastats.log; "
                "nohup /usr/bin/python3 /home/nesl/FABLE/scripts/physical_jetson_load.py "
                f"--duration-seconds {args.duration_seconds} --seed 34001 "
                "--ready-file /home/nesl/FABLE/state/physical-contention.ready.json "
                "--result-file /home/nesl/FABLE/state/physical-contention.result.json "
                "</dev/null >/home/nesl/FABLE/logs/physical-contention.log 2>&1 & "
                "echo $! > /home/nesl/FABLE/state/physical-contention.pid; "
                "p=$(cat /home/nesl/FABLE/state/physical-contention.pid); "
                "i=0; while test $i -lt 80 && "
                "! test -s /home/nesl/FABLE/state/physical-contention.ready.json; "
                "do kill -0 $p 2>/dev/null || exit 21; sleep 0.1; i=$((i+1)); done; "
                "test -s /home/nesl/FABLE/state/physical-contention.ready.json || exit 22; "
                "kill -0 $p 2>/dev/null || exit 23; "
                "nohup /usr/bin/tegrastats --interval 250 "
                "--logfile /home/nesl/FABLE/logs/physical-contention-tegrastats.log "
                "</dev/null >/dev/null 2>&1 & "
                "echo $! > /home/nesl/FABLE/state/physical-contention-tegrastats.pid; "
                "cat /home/nesl/FABLE/state/physical-contention.ready.json"
            )
        elif args.action == "clear-compute":
            command = (
                "rm -f /home/nesl/FABLE/state/physical-yolo-relay-only; "
                "p=$(cat /home/nesl/FABLE/state/physical-contention.pid 2>/dev/null); "
                "test -n \"$p\" && kill \"$p\" 2>/dev/null || true; "
                "i=0; while ! test -s /home/nesl/FABLE/state/physical-contention.result.json "
                "&& test $i -lt 50; do sleep 0.1; i=$((i+1)); done; "
                "t=$(cat /home/nesl/FABLE/state/physical-contention-tegrastats.pid 2>/dev/null); "
                "test -n \"$t\" && kill \"$t\" 2>/dev/null || true; "
                "printf '%s\\n' 'CONTENTION_RESULT'; "
                "cat /home/nesl/FABLE/state/physical-contention.result.json 2>/dev/null || true; "
                "printf '%s\\n' 'TEGRASTATS_TAIL'; "
                "tail -n 400 /home/nesl/FABLE/logs/physical-contention-tegrastats.log 2>/dev/null || true"
            )
        else:
            parser.error("network actions require target rpi_to_jetson")
    else:
        target = config["targets"]["rpi_to_jetson"]
        host = target["controller_host"]
        interface = target["interface"]
        if args.action == "check":
            command = f"test -x {NETWORK_HELPER} && sudo -n {NETWORK_HELPER} status"
        elif args.action == "apply-network":
            profile = config["profiles"].get(args.profile)
            if profile is None or profile["rate_mbit"] is None:
                parser.error("apply-network requires a bounded degraded profile")
            # The root-owned helper owns the fixed interface, profile values,
            # and automatic restore timer.  No trace-controlled text reaches a
            # privileged shell.
            if args.profile != "P1_JETSON_PATH_DEGRADED":
                parser.error("only P1_JETSON_PATH_DEGRADED is allowlisted")
            command = f"sudo -n {NETWORK_HELPER} apply P1_JETSON_PATH_DEGRADED"
        elif args.action == "restore-network":
            command = f"sudo -n {NETWORK_HELPER} restore"
        elif args.action == "disconnect-network":
            command = f"sudo -n {NETWORK_HELPER} disconnect"
        else:
            parser.error("compute actions require target physical_jetson")
    result = remote(args.identity_file, host, command, execute=args.execute)
    result.update({"action": args.action, "target": args.target, "profile": args.profile})
    if args.action == "apply-compute" and result.get("stdout"):
        try:
            result["readiness"] = json.loads(result["stdout"])
        except json.JSONDecodeError:
            result["readiness"] = None
    if args.action == "clear-compute" and result.get("stdout"):
        result.update(parse_compute_clear_output(result["stdout"]))
    result["validated"] = bool(
        result.get("executed") and result.get("returncode", 1) == 0
    )
    if args.action == "apply-compute":
        result["validated"] = bool(
            result["validated"]
            and isinstance(result.get("readiness"), dict)
            and result["readiness"].get("ready") is True
        )
    elif args.action == "clear-compute":
        result["validated"] = bool(
            result["validated"] and result.get("measurement_validated")
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result.get("returncode", 0) != 0)


if __name__ == "__main__":
    raise SystemExit(main())
