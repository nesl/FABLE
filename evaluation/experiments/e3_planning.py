"""E3 checkpoint-planning planned-run builder (oracle deferred)."""

from .matrix import build_run_matrix
from .specs import ExperimentQuestion

QUESTION = ExperimentQuestion.RQ2_PLANNING


def build(catalog, **kwargs):
    return build_run_matrix(catalog, QUESTION, **kwargs)
