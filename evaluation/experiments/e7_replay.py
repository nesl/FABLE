"""E7 replay/continuation planned-run builder."""

from .matrix import build_run_matrix
from .specs import ExperimentQuestion

QUESTION = ExperimentQuestion.RQ3_CONTINUATION


def build(catalog, **kwargs):
    return build_run_matrix(catalog, QUESTION, **kwargs)
