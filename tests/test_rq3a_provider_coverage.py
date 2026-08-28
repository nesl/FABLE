from pathlib import Path

from evaluation.rq3a_provider_coverage import validate_rq3a_provider_coverage


ROOT = Path(__file__).resolve().parents[1]


def test_primary_rq3a_families_have_non_reference_runtime_paths() -> None:
    result = validate_rq3a_provider_coverage(
        ROOT / "iobt-minimal-ce-replay/config/fable_provider_runtimes.yaml"
    )
    assert result["valid"], result
    assert {item["family_id"] for item in result["families"]} == {
        "pass_follow_clear_convoy",
        "multimodal_robbery",
    }
