"""Regression specifications derived from the bounded RQ3 outage campaign.

These tests intentionally use durable campaign artifacts. They distinguish
normal post-reconnection live evidence from explicit historical catch-up and
pin the planner/semantic behaviors that must be repaired before the full
continuation campaign is publishable.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT
    / "evaluation/results/rq3_network_single_disconnect_full_20260807/rq3a"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fable_degraded_result(experiment_id: str) -> tuple[Path, dict]:
    matches = tuple(
        RESULTS.glob(
            "rq3-network-single-disconnect-*"
            f"/FABLE/repetition-01/{experiment_id}.json"
        )
    )
    if not matches:
        pytest.skip("durable RQ3 diagnostic artifact is not present")
    assert len(matches) == 1
    return matches[0], _load(matches[0])


def _records(result_path: Path, name: str) -> tuple[dict, ...]:
    path = result_path.with_suffix(".records") / f"{name}.jsonl"
    return tuple(_load_line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() for _load_line in (json.loads(line),))


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _outage_bounds(result_path: Path) -> tuple[datetime, datetime]:
    disturbances = _records(result_path, "disturbance_event")
    failed = next(item for item in disturbances if item["action"] == "FAIL")
    restored = next(item for item in disturbances if item["action"] == "RESTORE")
    return _timestamp(failed["wall_timestamp"]), _timestamp(restored["wall_timestamp"])


def _failed_endpoint(result: dict) -> tuple[str, str]:
    transition = next(
        item
        for item in result["condition_trace"]["transitions"]
        if item["action"] == "FAIL_LINK"
    )
    switch = transition["target_id"].split(":")[1].removeprefix("s_")
    if switch.startswith("orin"):
        return f"dvpg_gq_orin_{switch.removeprefix('orin')}", f"{switch}_camera"
    return (
        f"mobile_archive_{switch.removeprefix('mob')}",
        f"mobile_archive_{switch.removeprefix('mob')}_camera",
    )


def test_drop_offline_b1_does_not_receive_or_explicitly_replay_outage_evidence() -> None:
    """B1's route-convoy TP comes from the live tail, not history replay."""

    experiment_id = "20241008-route-convoy-18-r030"
    matches = tuple(
        RESULTS.glob(
            "rq3-network-single-disconnect-*"
            f"/B1_STATIC_WHOLE_EVENT/repetition-01/{experiment_id}.json"
        )
    )
    if not matches:
        pytest.skip("durable RQ3 diagnostic artifact is not present")
    result_path = matches[0]
    result = _load(result_path)
    failed_at, restored_at = _outage_bounds(result_path)
    _, failed_source = _failed_endpoint(result)
    observations = tuple(
        item
        for item in _records(result_path, "predicate_observation")
        if item.get("sensor_id") == failed_source
    )

    assert observations
    assert not any(
        failed_at <= _timestamp(item["wall_timestamp"]) <= restored_at
        for item in observations
    )
    assert result.get("retrospective_replay_statuses", []) == []
    assert not any(
        "NODE_RECOVERED" in item.get("replan_trigger", "")
        for item in _records(result_path, "plan_decision")
    )
    assert any(_timestamp(item["wall_timestamp"]) > restored_at for item in observations)


@pytest.mark.xfail(
    strict=True,
    reason="durable artifact predates deployment-availability projection repair",
)
@pytest.mark.parametrize(
    "experiment_id",
    (
        "20260415-cross-sensor-robbery-robbery-13-r013",
        "20260414-three-visit-stalking-stalking-30-r030",
        "20241008-route-convoy-18-r030",
    ),
)
def test_fable_never_selects_failed_endpoint_during_validated_outage(
    experiment_id: str,
) -> None:
    result_path, result = _fable_degraded_result(experiment_id)
    failed_at, restored_at = _outage_bounds(result_path)
    failed_node, failed_source = _failed_endpoint(result)
    outage_plans = tuple(
        item
        for item in _records(result_path, "plan_decision")
        if failed_at <= _timestamp(item["wall_timestamp"]) <= restored_at
    )

    assert outage_plans
    assert all(
        failed_node not in item.get("selected_node_ids", ())
        and failed_source not in item.get("selected_source_ids", ())
        for item in outage_plans
    )


@pytest.mark.xfail(
    strict=True,
    reason="durable artifact predates mobile link-state routing repair",
)
@pytest.mark.parametrize(
    "experiment_id",
    (
        "20250812-robbery-with-alarm-burglary-a-r012",
        "20250812-talking-rendezvous-rendezvous-brianjulian-1-r029",
    ),
)
def test_mobile_disconnect_produces_validated_link_state_replan(
    experiment_id: str,
) -> None:
    result_path, _ = _fable_degraded_result(experiment_id)
    triggers = {
        item.get("replan_trigger", "")
        for item in _records(result_path, "plan_decision")
    }
    assert any("VALIDATED_LINK_STATE:link:s_mob" in value for value in triggers)


@pytest.mark.xfail(
    strict=True,
    reason="durable artifact predates recovery-scope frontier preservation repair",
)
def test_three_visit_progression_survives_link_epoch() -> None:
    _, result = _fable_degraded_result(
        "20260414-three-visit-stalking-stalking-30-r030"
    )
    applied = sum(
        item.get("transition_status") == "APPLIED"
        for item in result.get("progress_diagnostics", ())
    )

    assert applied >= 3
    assert result["detected"] is True
    assert result["deadline_missed"] is False
