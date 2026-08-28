"""E4 controlled provider-escalation planned-run builder."""

from .matrix import build_run_matrix
from .specs import ExperimentQuestion

QUESTION = ExperimentQuestion.RQ_PROVIDER_ESCALATION


def build(catalog, **kwargs):
    return build_run_matrix(catalog, QUESTION, **kwargs)
