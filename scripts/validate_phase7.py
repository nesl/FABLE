#!/usr/bin/env python3
"""Validate Phase 0-7 tests, provider wiring, fixtures, and deterministic demo."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> int:
    run(sys.executable, "-m", "providers.tools.chain_validator")
    run(sys.executable, "scripts/generate_phase7_fixtures.py")
    run(sys.executable, "-m", "pytest", "-q")
    run(sys.executable, "scripts/demo_phase7.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
