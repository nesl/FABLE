"""Compatibility facade for FABLE data contracts.

Canonical contract definitions are grouped under :mod:`fable.contracts`.  This
module remains as a stable import path for existing callers while behavior is
kept outside the data-contract layer.
"""

from fable.contracts import *  # noqa: F401,F403
