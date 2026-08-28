import base64
import json

from evaluation.e4_identity_judging import IdentityEvidenceCapture, summarize_judgments


def _url(content: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(content).decode()


def test_capture_deduplicates_prediction_and_persists_pair_evidence(tmp_path):
    capture = IdentityEvidenceCapture(
        tmp_path, experiment_id="trace", baseline_id="FABLE", maximum_predictions=2
    )
    for source, local, content in (("orin1", "track1", b"left"), ("orin4", "track9", b"right")):
        capture.observe_descriptor({
            "source_id": source,
            "entity_kind": "vehicle",
            "event_time_interval": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T00:00:01Z"},
            "records": [{"local_entity_id": local, "source_context_image_data_urls": [_url(content)]}],
        })
    association = {
        "left_source_id": "orin1", "right_source_id": "orin4", "entity_kind": "vehicle",
        "associations": [{"left_local_entity_id": "track1", "right_local_entity_id": "track9", "confidence": 0.8, "association_basis": "reid"}],
    }
    assert capture.observe_associations(association) == 1
    assert capture.observe_associations(association) == 0
    row = json.loads(capture.manifest.read_text().strip())
    assert row["predicted_same_identity"] is True
    assert row["baseline_id"] == "FABLE"
    capture.finalize()
    assert len(tuple(capture.images.iterdir())) == 2


def test_summary_treats_undetermined_as_coverage_not_error():
    rows = [
        {"baseline_id": "FABLE", "pair_id": "a", "judge_label": "MATCH", "agreement": True},
        {"baseline_id": "FABLE", "pair_id": "b", "judge_label": "UNDETERMINED", "agreement": False},
    ]
    summary = summarize_judgments(rows)
    assert summary["by_baseline"]["FABLE"]["vlm_judged_binding_precision"] == 1.0
    assert summary["by_baseline"]["FABLE"]["undetermined"] == 1
