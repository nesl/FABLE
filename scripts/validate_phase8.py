#!/usr/bin/env python3
"""Validate Phases 0-8, provider wiring, fixtures, and the Phase-8 demo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> int:
    run(sys.executable, "-m", "providers.tools.chain_validator")
    run(sys.executable, "scripts/generate_phase8_fixtures.py")
    run(sys.executable, "-m", "pytest", "-q")
    run(sys.executable, "scripts/demo_phase8.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
