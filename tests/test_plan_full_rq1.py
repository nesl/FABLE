from __future__ import annotations

import importlib.util
from pathlib import Path

from evaluation.catalog import ExperimentCatalog
from evaluation.schemas import EvaluationMode


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plan_full_rq1.py"


def _module():
    spec = importlib.util.spec_from_file_location("plan_full_rq1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_e1_plan_has_six_adjacent_policies_per_recommended_trace() -> None:
    module = _module()
    catalog = ExperimentCatalog.from_csv(
        ROOT / "evaluation/labels/filtered_complex_event_experiments.csv",
        transition_model_path=(
            ROOT / "evaluation/labels/site_sensor_transition_model_2024_2025.json"
        ),
    )
    rows = module.planned_runs(catalog, seed=17)
    assert len(rows) == (
        len(catalog.recommended()) - len(module.E1_REPLAY_EXCLUSIONS)
    ) * 6
    assert len({row.run_id for row in rows}) == len(rows)
    assert all(row.mode == EvaluationMode.FULL_STACK for row in rows)
    assert all(row.playback_mode == "realtime" for row in rows)
    assert not ({row.experiment_id for row in rows} & set(module.E1_REPLAY_EXCLUSIONS))
    for offset in range(0, len(rows), 6):
        block = rows[offset : offset + 6]
        assert len({row.experiment_id for row in block}) == 1
        assert tuple(row.baseline_id for row in block) == module.ALL_RQ1_POLICIES
