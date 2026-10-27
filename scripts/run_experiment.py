#!/usr/bin/env python3
"""Run one self-contained replay case with one reusable network plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml

from scripts.visualization_contract import IDLE_STATE

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "replay" / "cases"
PLANS = ROOT / "replay" / "plans"
RESOLVED = ROOT / ".fable" / "runs" / "resolved"
PROJECT_PYTHON = ROOT / ".venv" / "bin" / "python"
COMPOSE_BASE = (
    "docker", "compose", "--env-file", ".env", "-p", "fable-e2e",
    "-f", "deployment/compose.example.yaml",
)
COMPOSE_NETWAGGLE = (*COMPOSE_BASE, "-f", "deployment/compose.netwaggle.real.yaml")
COMPOSE_DIRECT = (*COMPOSE_BASE, "-f", "deployment/compose.direct.yaml")


def _write_idle_visualization(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".launcher.tmp")
    temporary.write_text(
        json.dumps(IDLE_STATE, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_cases() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    cases: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for path in sorted(CASES.glob("*.yaml")):
        case = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        name = str(case.get("case_id", path.stem))
        if int(case.get("version", 0)) != 1 or not {"event", "replay"} <= set(case):
            raise ValueError(f"invalid replay case: {path}")
        if name in cases or name in aliases:
            raise ValueError(f"duplicate replay case name: {name}")
        case["_path"] = path
        cases[name] = case
        aliases[name] = name
        for alias in case.get("aliases", ()):
            alias = str(alias)
            if alias in aliases:
                raise ValueError(f"duplicate replay case alias: {alias}")
            aliases[alias] = name
    return cases, aliases


def _load_plan(name: str) -> dict[str, Any]:
    path = PLANS / f"{name}.yaml"
    plan = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if int(plan.get("version", 0)) != 1 or plan.get("mode") not in {"direct", "netwaggle"}:
        raise ValueError(f"invalid network plan: {path}")
    plan["_path"] = path
    return plan


def _materialize(case_name: str, case: dict[str, Any], plan_name: str,
                 plan: dict[str, Any]) -> Path:
    """Write the old runner's combined input as ignored generated state."""
    RESOLVED.mkdir(parents=True, exist_ok=True)
    event_path = RESOLVED / f"{case_name}.event.yaml"
    replay_path = RESOLVED / f"{case_name}.replay.yaml"
    experiment_path = RESOLVED / f"{case_name}-{plan_name}.yaml"
    event_path.write_text(yaml.safe_dump(case["event"], sort_keys=False), encoding="utf-8")
    replay_path.write_text(yaml.safe_dump(case["replay"], sort_keys=False), encoding="utf-8")
    experiment = {
        # The legacy low-level runner always accepts a topology. Direct mode
        # never starts it, so keep that implementation default out of the
        # user-facing direct plan.
        "topology": str(plan.get("topology", "netwaggle/topologies/sensor_edge.yaml")),
        "event": str(event_path.relative_to(ROOT)),
        "replay_manifest": str(replay_path.relative_to(ROOT)),
        "timeout_seconds": float(case.get("timeout_seconds", 120)),
        "require_completion": bool(case.get("require_completion", False)),
        "conditions": plan.get("conditions", []),
    }
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8")
    return experiment_path


def _command(experiment: Path, *, direct: bool, dry_run: bool, visualize: bool,
             output: Path | None, policy: str, event_only: bool,
             event_evidence: bool) -> list[str]:
    command = [
        str(PROJECT_PYTHON if PROJECT_PYTHON.is_file() else Path(sys.executable)),
        str(ROOT / "scripts" / "run_network_experiment.py"),
        str(experiment.relative_to(ROOT)),
        "--compose-override",
        "deployment/compose.direct.yaml" if direct else "deployment/compose.netwaggle.real.yaml",
    ]
    if direct:
        command.append("--direct")
    if dry_run:
        command.append("--dry-run")
    if visualize:
        command.append("--hold-after-completion")
    if output is not None:
        command.extend(("--output", str(output)))
    if policy != "fable":
        command.extend(("--policy", policy))
    if event_only:
        command.append("--event-only")
    if event_evidence:
        command.append("--event-evidence")
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", dest="case_name", help="self-contained replay case")
    parser.add_argument(
        "--plan", choices=("direct", "network_nominal", "network_disconnect")
    )
    parser.add_argument("--list", action="store_true", help="list cases and reusable network plans")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy", choices=("fable", "b0"), default="fable")
    parser.add_argument("--event-only", action="store_true")
    parser.add_argument("--event-evidence", action="store_true")
    parser.add_argument("action", nargs="?", choices=("stack", "status", "stop"))
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)

    cases, aliases = _load_cases()
    if args.action:
        if args.case_name or args.plan:
            parser.error("stack actions cannot be combined with --case or --plan")
        if args.action == "stack":
            _write_idle_visualization(ROOT / ".fable/runs/visualization-state.json")
            result = subprocess.call([*COMPOSE_NETWAGGLE, "up", "-d", "--build"], cwd=ROOT)
            if result or args.detach:
                return result
            try:
                return subprocess.call([*COMPOSE_NETWAGGLE, "logs", "-f"], cwd=ROOT)
            except KeyboardInterrupt:
                return 130
        command = [*COMPOSE_NETWAGGLE, "ps" if args.action == "status" else "down"]
        return subprocess.call(command, cwd=ROOT)

    if args.list:
        print("Cases (each file contains its CE definition and replay sources):")
        for name, case in cases.items():
            aliases_text = ", ".join(case.get("aliases", ())) or "-"
            print(f"  {name}: {case['_path'].relative_to(ROOT)}")
            print(f"    {case.get('description', '')}")
            print(f"    aliases: {aliases_text}; sources: {len(case['replay'].get('sources', ())) }")
        print("Network plans (reused by every case):")
        for path in sorted(PLANS.glob("*.yaml")):
            plan = _load_plan(path.stem)
            print(f"  {path.stem}: {path.relative_to(ROOT)} ({plan['mode']})")
        return 0

    if not args.case_name or not args.plan:
        parser.error("--case and --plan are required unless --list is used")
    canonical = aliases.get(args.case_name)
    if canonical is None:
        parser.error(f"unknown case {args.case_name!r}; use --list")
    case = cases[canonical]
    plan = _load_plan(args.plan)
    experiment = _materialize(canonical, case, args.plan, plan)
    direct = plan["mode"] == "direct"
    command = _command(
        experiment, direct=direct, dry_run=args.dry_run, visualize=args.visualize,
        output=args.output, policy=args.policy, event_only=args.event_only,
        event_evidence=args.event_evidence,
    )
    print(f"case: {canonical} ({case['_path'].relative_to(ROOT)})")
    print(f"network plan: {args.plan} ({plan['_path'].relative_to(ROOT)})")
    print(f"network mode: {'ordinary Docker networking' if direct else 'NetWaggle'}")
    print("command: " + shlex.join(command), flush=True)
    if not args.dry_run:
        compose = COMPOSE_DIRECT if direct else COMPOSE_NETWAGGLE
        result = subprocess.call([*compose, "up", "-d"], cwd=ROOT)
        if result:
            return result
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
