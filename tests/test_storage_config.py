from pathlib import Path

from fable.common.storage import load_storage_config


ROOT = Path(__file__).resolve().parents[1]


def test_storage_config_resolves_standard_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FABLE_STORAGE_ROOT", str(tmp_path))
    config = load_storage_config(repository_root=ROOT)

    assert config.path("results") == tmp_path / "results"
    assert config.path("runs") == tmp_path / "runs"
    assert config.path("debug") == tmp_path / "debug"
    links = config.link_targets(ROOT)
    assert links[ROOT / "evaluation/results"] == tmp_path / "results"
    assert ROOT / "runs" not in links
