#!/usr/bin/env python3
"""Validate spatial coordination, request compilation, LLM hooks, and Phases 0-8."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> int:
    run(sys.executable, "-m", "providers.tools.chain_validator")
    run(sys.executable, "-m", "pytest", "-q")
    run(sys.executable, "scripts/demo_spatial_checkpoint.py")
    run(sys.executable, "scripts/compile_request.py", "detect a convoy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
