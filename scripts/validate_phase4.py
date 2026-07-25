#!/usr/bin/env python3
"""Run the complete test suite and the deterministic Phase-4 demo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    run(sys.executable, "-m", "pytest", "-q")
    run(sys.executable, "scripts/demo_phase4.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
