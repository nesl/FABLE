#!/usr/bin/env python3
"""Validate Phases 0-6 and run the distributed reference demonstration."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> int:
    run(sys.executable, "-m", "pytest", "-q")
    run(sys.executable, "scripts/demo_phase6.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
