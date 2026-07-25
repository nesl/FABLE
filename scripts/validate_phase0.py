#!/usr/bin/env python3
"""Run the Phase-0 contract and existing provider-catalog test suites."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

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
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
