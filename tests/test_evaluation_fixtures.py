import json
from pathlib import Path

from evaluation.metrics.spatial_coordination import evaluate_spatial_coordination
from evaluation.schemas import CoordinationEpisode

ROOT = Path(__file__).resolve().parents[1]


def test_coordination_fixture_is_schema_valid_and_scope_aware() -> None:
    payload = json.loads(
        (ROOT / "tests/evaluation_fixtures/coordination_episodes.json").read_text()
    )
    episodes = tuple(CoordinationEpisode.model_validate(item) for item in payload)
    metrics = evaluate_spatial_coordination(episodes)
    assert metrics.evaluated_episodes == 1
    assert metrics.excluded_unknown_topology == 1
    assert metrics.excluded_unavailable_mobile_target == 1
