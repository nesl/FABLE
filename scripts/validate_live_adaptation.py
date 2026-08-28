#!/usr/bin/env python3
"""Bounded apply/probe/restore validation for one live host condition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.adaptation_controller import (  # noqa: E402
    AdaptationControlPolicy,
    AdaptationController,
    AdaptationControllerMode,
    AllowedDisturbanceTarget,
)
from evaluation.disturbance_schedule import (  # noqa: E402
    DisturbanceKind,
    ScheduledDisturbanceAction,
)
from evaluation.host_validation import (  # noqa: E402
    CgroupExpectation,
    NetworkPathProbe,
    validate_cgroup_state,
    validate_network_path,
)
from evaluation.live_validation import validate_apply_restore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("network", "capacity"), required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--restore-condition", required=True)
    parser.add_argument("--condition-probe", type=Path, required=True)
    parser.add_argument("--restore-probe", type=Path, required=True)
    parser.add_argument(
        "--cgroup-root",
        type=Path,
        default=Path("/sys/fs/cgroup"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    kind = (
        DisturbanceKind.NETWORK_PROFILE
        if args.kind == "network"
        else DisturbanceKind.CAPACITY_PROFILE
    )
    policy = AdaptationControlPolicy(
        policy_id="live-validation",
        mode=AdaptationControllerMode.HOST_HELPER,
        helper_path=args.helper,
        targets=(
            AllowedDisturbanceTarget(
                target_id=args.target,
                kinds=(kind,),
                condition_ids=(args.condition, args.restore_condition),
            ),
        ),
    )
    now = datetime.now(timezone.utc)

    def action(verb: str, condition: str) -> ScheduledDisturbanceAction:
        return ScheduledDisturbanceAction(
            step_id="live-validation",
            action=verb,
            kind=kind,
            target_id=args.target,
            condition_id=condition,
            due_at=now,
        )

    def probe(path: Path):
        document = json.loads(path.read_text(encoding="utf-8"))
        if kind == DisturbanceKind.NETWORK_PROFILE:
            probes = tuple(
                NetworkPathProbe.model_validate(item)
                for item in document["probes"]
            )
            results = [validate_network_path(item) for item in probes]
            return {
                "path_count": len(results),
                "paths_validated": sum(
                    item["path_validated"] is True for item in results
                ),
            }
        return validate_cgroup_state(
            CgroupExpectation.model_validate(document["expectation"]),
            cgroup_root=args.cgroup_root,
        )

    result = validate_apply_restore(
        AdaptationController(policy),
        apply_action=action("APPLY", args.condition),
        restore_action=action("RESTORE", args.restore_condition),
        condition_probe=lambda _: probe(args.condition_probe),
        restored_probe=lambda _: probe(args.restore_probe),
        observed_at=now,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    print(result.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
