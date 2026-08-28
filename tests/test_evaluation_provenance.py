from pathlib import Path

from evaluation.provenance import (
    build_run_provenance,
    discover_input_paths,
    fingerprint_files,
)


def test_fingerprint_is_stable_and_changes_with_content(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("value: one\n", encoding="utf-8")
    first_files, first_digest = fingerprint_files(tmp_path, ("config.yaml",))
    second_files, second_digest = fingerprint_files(tmp_path, ("config.yaml",))

    assert first_files == second_files
    assert first_digest == second_digest

    (tmp_path / "config.yaml").write_text("value: two\n", encoding="utf-8")
    _, changed_digest = fingerprint_files(tmp_path, ("config.yaml",))
    assert changed_digest != first_digest


def test_provenance_records_arguments_without_environment_dump(
    tmp_path: Path,
) -> None:
    (tmp_path / "runner.py").write_text("print('test')\n", encoding="utf-8")
    provenance = build_run_provenance(
        tmp_path,
        runner_arguments={"scenario": "example", "output": Path("result.json")},
        input_paths=("runner.py", "missing.yaml"),
        model_candidates=(),
    )

    assert provenance["runner_arguments"] == {
        "output": "result.json",
        "scenario": "example",
    }
    missing = next(
        item for item in provenance["inputs"] if item["path"] == "missing.yaml"
    )
    assert missing == {
        "path": "missing.yaml",
        "status": "missing",
    }
    assert "environment" not in provenance


def test_input_discovery_excludes_results(tmp_path: Path) -> None:
    (tmp_path / "evaluation/results").mkdir(parents=True)
    (tmp_path / "evaluation/logic.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "evaluation/results/run.json").write_text("{}", encoding="utf-8")

    paths = discover_input_paths(tmp_path)

    assert "evaluation/logic.py" in paths
    assert "evaluation/results/run.json" not in paths
