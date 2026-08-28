import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mixed_executor_runs_all_policies_on_one_clock_per_run(tmp_path: Path) -> None:
    completed = subprocess.run([
        str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_rq3a_mixed_workload.py"),
        "--matrix", str(ROOT / "evaluation/manifests/workloads/rq3a_mixed_480s_matrix.jsonl"),
        "--output-dir", str(tmp_path), "--clock-scale", "0",
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / "campaign-report.json").read_text())
    assert report["completed_runs"] == 3
    by_policy = {row["policy_id"]: row for row in report["runs"]}
    assert set(by_policy) == {"B2_FRONTIER_FIXED_REALIZATION", "B3_TASK_RESOURCE_ADAPTIVE", "FABLE"}
    assert all(row["episodes_completed"] == 5 for row in by_policy.values())
    assert all(row["maximum_concurrent_requests"] >= 2 for row in by_policy.values())
    assert by_policy["B2_FRONTIER_FIXED_REALIZATION"]["request_replans"] == 0
    assert by_policy["FABLE"]["request_replans"] > 0
