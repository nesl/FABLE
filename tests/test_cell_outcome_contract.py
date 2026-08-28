from evaluation.cell_outcome import CellOutcome


def test_scientific_false_negative_does_not_abort_safe_campaign() -> None:
    outcome = CellOutcome(
        infrastructure_status="SUCCEEDED", protocol_status="VALID",
        mutation_status="SUCCEEDED", adaptation_status="INFEASIBLE",
        measurement_status="VALID", scientific_classification="FALSE_NEGATIVE",
        cleanup_status="SUCCEEDED",
    )
    assert outcome.safe_to_continue is True


def test_infrastructure_failure_is_not_safe_to_continue() -> None:
    outcome = CellOutcome(
        infrastructure_status="FAILED", protocol_status="UNKNOWN",
        mutation_status="UNKNOWN", adaptation_status="UNKNOWN",
        measurement_status="INVALID", scientific_classification="RUNTIME_ERROR",
        cleanup_status="UNKNOWN",
    )
    assert outcome.safe_to_continue is False
