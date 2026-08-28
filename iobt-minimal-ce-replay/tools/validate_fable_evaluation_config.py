#!/usr/bin/env python3
"""Fail fast when a FABLE systems evaluation is not isolated from legacy/oracle paths."""

from __future__ import annotations

import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    profile = os.environ.get("FABLE_EXECUTION_PROFILE", "").strip().lower()
    if profile != "real":
        raise SystemExit(
            "FABLE systems evaluation requires FABLE_EXECUTION_PROFILE=real; "
            f"got {profile!r}"
        )
    replay = yaml.safe_load((ROOT / "compose.replay.yaml").read_text())
    legacy = replay.get("services", {}).get("complex-event-detector", {})
    profiles = set(legacy.get("profiles", ()))
    if "legacy-ce" not in profiles:
        raise SystemExit(
            "complex-event-detector must be gated behind the legacy-ce Compose profile"
        )
    fable = yaml.safe_load((ROOT / "compose.fable.yaml").read_text())
    env = fable.get("services", {}).get("fable-orchestrator", {}).get("environment", {})
    if "FABLE_EXECUTION_PROFILE" not in env:
        raise SystemExit("compose.fable.yaml does not pass FABLE_EXECUTION_PROFILE")
    print("FABLE evaluation configuration validated: real providers only; legacy CE disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
