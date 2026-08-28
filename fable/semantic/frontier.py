"""Deprecated compatibility exports for the split semantic evaluators.

The live implementation is deliberately separated by responsibility:
``FrontierDeriver`` coordinates frontier/state propagation,
``CompositeEvaluator`` evaluates graph topology, and ``TemporalEvaluator``
evaluates temporal guards.  This module contains no independent logic.
"""

from .evaluation import CompositeEvaluator
from .frontier_deriver import FrontierDeriver
from .temporal import TemporalEvaluator

__all__ = ["CompositeEvaluator", "FrontierDeriver", "TemporalEvaluator"]
