#!/usr/bin/env python3
"""Run all contract/provider/semantic tests and the deterministic Phase-1 demo."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(
        loader.discover(
            start_dir=str(PROJECT_ROOT / "tests"),
            pattern="test_*.py",
            top_level_dir=str(PROJECT_ROOT),
        )
    )
    suite.addTests(
        loader.discover(
            start_dir=str(PROJECT_ROOT / "providers" / "tests"),
            pattern="*_test.py",
            top_level_dir=str(PROJECT_ROOT),
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    subprocess.run(
        [sys.executable, "scripts/demo_phase1.py"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
