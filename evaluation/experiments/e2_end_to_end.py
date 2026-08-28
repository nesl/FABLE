"""E2 nominal end-to-end planned-run builder."""

from .matrix import build_run_matrix
from .specs import ExperimentQuestion

QUESTION = ExperimentQuestion.RQ1_END_TO_END


def build(catalog, **kwargs):
    return build_run_matrix(catalog, QUESTION, **kwargs)
