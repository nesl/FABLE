from pathlib import Path

from scripts.run_rq3_campaigns import (
    disturbed_cell_preflight,
    execution_groups,
    load_manifest,
    spatial_execution_candidates,
    spatial_execution_nodes,
    summarize,
    topology_shortlist,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "evaluation/results/rq3_all_campaigns_20260731/manifests/rq3b.jsonl"
)


def _run(policy_id: str):
    return next(
        run for run in load_manifest(MANIFEST) if run.baseline_id.value == policy_id
    )


def test_topology_shortlist_is_model_derived_and_bounded_to_available_sensors():
    run = _run("SPATIAL_TOPOLOGY_SHORTLIST")
    selected = topology_shortlist(run)

    assert selected
    assert set(selected) <= {
        sensor.replace("orin_", "orin")
        for sensor in run.replay_supported_sensor_ids
    }
    assert len(selected) < len(run.replay_supported_sensor_ids)


def test_spatial_policies_have_distinct_executable_scopes():
    broadcast = spatial_execution_nodes(_run("SPATIAL_BROADCAST"))
    topology = spatial_execution_nodes(_run("SPATIAL_TOPOLOGY_SHORTLIST"))
    resource = spatial_execution_nodes(_run("SPATIAL_RESOURCE_ONLY"))
    fable = spatial_execution_nodes(_run("SPATIAL_FABLE"))

    assert broadcast == ()
    assert len(topology) > 2
    assert len(resource) == 2
    assert len(fable) == 2
    assert set(fable) <= set(topology)


def test_bounded_policies_retain_ranked_runtime_fallback_candidates():
    for policy in ("SPATIAL_RESOURCE_ONLY", "SPATIAL_FABLE"):
        run = _run(policy)
        selected = spatial_execution_nodes(run)
        candidates = spatial_execution_candidates(run)

        assert candidates[: len(selected)] == selected
        assert len(candidates) >= len(selected)
        assert set(selected) <= set(candidates)


def test_rq3a_execution_runs_network_then_compute_trace_major():
    manifest = (
        ROOT
        / "evaluation/manifests/adaptation/"
        "rq3a_focused_lease_controlled_matrix.jsonl"
    )
    runs = load_manifest(manifest)
    groups, deferred = execution_groups("rq3a", runs)

    assert not deferred
    assert all(len(rows) == 1 for _, rows in groups)
    ordered = [rows[0] for _, rows in groups]
    network = [run for run in runs if "compute" not in run.condition_trace_id]
    compute = [run for run in runs if "compute" in run.condition_trace_id]
    assert [run.run_id for run in ordered] == [
        run.run_id for run in (*network, *compute)
    ]
    for partition in (network, compute):
        for index in range(0, len(partition), 3):
            trace_rows = partition[index : index + 3]
            assert [run.baseline_id.value for run in trace_rows] == [
                "B2_FRONTIER_FIXED_REALIZATION",
                "B3_TASK_RESOURCE_ADAPTIVE",
                "FABLE",
            ]
            assert len({run.experiment_id for run in trace_rows}) == 1


def test_updated_rq3a_matrix_is_case_major_and_balanced():
    manifest = (
        ROOT
        / "evaluation/manifests/adaptation/rq3a_updated/"
        "rq3a_network_updated_55.jsonl"
    )
    runs = load_manifest(manifest)
    groups, deferred = execution_groups("rq3a", runs)

    assert not deferred
    assert len(groups) == 55
    assert all(len(rows) == 1 for _, rows in groups)
    for index in range(0, 55, 5):
        block = [rows[0] for _, rows in groups[index : index + 5]]
        assert len({row.condition_trace_id for row in block}) == 1
        assert [row.baseline_id.value for row in block] == [
            "B1_STATIC_WHOLE_EVENT",
            "B2_FRONTIER_FIXED_REALIZATION",
            "B3_TASK_RESOURCE_ADAPTIVE",
            "B4_GREEDY_FRONTIER",
            "FABLE",
        ]


def test_disturbed_preflight_fails_closed_without_nominal_result(tmp_path):
    runs = load_manifest(
        ROOT
        / "evaluation/manifests/adaptation/rq3a_updated/"
        "rq3a_network_updated_55.jsonl"
    )
    disturbed = next(row for row in runs if row.disturbance_profile_id == "N1")

    report = disturbed_cell_preflight(
        tmp_path, disturbed, disturbed.baseline_id.value
    )

    assert report["applicable"]
    assert not report["valid"]
    assert "nominal control" in report["reason"]


def test_prepared_e3_retrospective_matrix_is_trace_major_and_policy_isolated():
    manifest = (
        ROOT
        / "evaluation/manifests/spatial/e3_prepared_20260827/rq3c.jsonl"
    )
    runs = load_manifest(manifest)

    assert len(runs) == 12
    groups, deferred = execution_groups("rq3c", runs)
    assert not deferred
    assert len(groups) == 12
    assert all(len(rows) == 1 for _, rows in groups)
    for index in range(0, len(runs), 3):
        block = runs[index : index + 3]
        assert len({row.experiment_id for row in block}) == 1
        assert [row.retrospective_policy_id for row in block] == [
            "R0_NO_REPLAY",
            "R1_RAW_REPLAY",
            "R2_FABLE_TYPED_REPLAY",
        ]
        # The labels remain schema-compatible, but the runner executes all
        # three with the same FABLE planning runtime to isolate replay policy.
        assert [row.baseline_id.value for row in block] == [
            "B1_HANDWRITTEN_STATIC",
            "B2_FRONTIER_FIXED_REALIZATION",
            "FABLE",
        ]


def test_prepared_e3_spatial_matrix_excludes_unreplayable_chase_labels():
    manifest = (
        ROOT
        / "evaluation/manifests/spatial/e3_prepared_20260827/rq3b.jsonl"
    )
    runs = load_manifest(manifest)
    trace_ids = {run.experiment_id for run in runs}

    assert len(runs) == 45
    assert len(trace_ids) == 9
    assert not {
        "20241009-two-vehicle-chase-3-r005",
        "20241009-two-vehicle-chase-13-r016",
        "20241009-two-vehicle-chase-14-r017",
    } & trace_ids
    assert {
        "20241009-two-vehicle-chase-17-r020",
        "20241009-two-vehicle-chase-18-r021",
    } <= trace_ids


def test_e3_summary_ignores_stale_results_outside_current_manifest(tmp_path):
    run = load_manifest(
        ROOT / "evaluation/manifests/spatial/e3_prepared_20260827/rq3b.jsonl"
    )[0]
    result_dir = tmp_path / "rq3b" / "good_network" / run.baseline_id.value
    result_dir.mkdir(parents=True)
    current = {
        "experiment_id": run.experiment_id,
        "classification": "TRUE_POSITIVE",
        "detected": True,
        "suite": {
            "evaluation_policy_id": run.baseline_id.value,
            "runner_returncode": 0,
            "replay_nodes": [],
        },
    }
    stale = {
        **current,
        "experiment_id": "superseded-trace",
        "classification": "FALSE_NEGATIVE",
    }
    (result_dir / "current.json").write_text(__import__("json").dumps(current))
    (result_dir / "stale.json").write_text(__import__("json").dumps(stale))

    report = summarize(tmp_path, planned_runs=(run,), deferred=0)

    assert report["completed_result_files"] == 1
    assert report["classification_counts"] == {"TRUE_POSITIVE": 1}
