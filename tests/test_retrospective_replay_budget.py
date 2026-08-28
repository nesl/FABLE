from fable.distributed.node_agent import retrospective_replay_budget


def test_identity_critical_lookback_uses_trackable_frame_density() -> None:
    assert retrospective_replay_budget("PASSES") == (540, 3.0)
    assert retrospective_replay_budget("VEHICLE_PRESENT_BEFORE") == (540, 3.0)
    assert retrospective_replay_budget("PERSON_PRESENT") == (540, 3.0)


def test_non_track_forming_lookback_retains_low_cost_budget() -> None:
    assert retrospective_replay_budget("EXITS") == (180, 1.0)
