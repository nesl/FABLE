import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_combined_campaign_preflight(tmp_path: Path) -> None:
    completed = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_rq1_rq3a_unattended.py"),
         "--output-root", str(tmp_path), "--preflight-only"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / "preflight.json").read_text())
    assert report["rq1_rows"] == 45
    assert report["rq3a_single_rows"] == 36
    assert report["mixed_execution_ready"] is True
