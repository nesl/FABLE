import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.mixed_workload import MixedRequestWorkload


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_mixed_workload_is_bounded_and_overlapping() -> None:
    workload = MixedRequestWorkload.model_validate_json(
        (ROOT / "evaluation/manifests/workloads/rq3a_mixed_480s.json").read_text()
    )
    assert workload.duration_s == 480
    assert len(workload.episodes) == 5
    assert max(episode.end_offset_s for episode in workload.episodes) <= 480
    assert workload.vlm_mode == "replayed_response"


def test_mixed_workload_rejects_episode_past_end() -> None:
    document = json.loads(
        (ROOT / "evaluation/manifests/workloads/rq3a_mixed_480s.json").read_text()
    )
    document["episodes"][-1]["duration_s"] = 200
    with pytest.raises(ValidationError, match="finish within"):
        MixedRequestWorkload.model_validate(document)
