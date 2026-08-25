#!/usr/bin/env python3
"""Compatibility wrapper for the core FABLE catalog validator.

The validator is owned by :mod:`fable.catalog.chain_validator`; this module is
kept so existing scripts using ``python -m providers.tools.chain_validator``
continue to work.
"""

from fable.catalog.chain_validator import *  # noqa: F401,F403
from fable.catalog.chain_validator import main


if __name__ == "__main__":
    raise SystemExit(main())
