from pathlib import Path

import pytest

from evaluation.instrumentation import resource_delta, sample_resources
from replay.manifest import load_replay_manifest
from replay.discovery import discover_recordings
from evaluation.artifacts import CompletionArtifact, CompletionWriter, load_completions, score_presence
from evaluation.recording_map import load_recording_map
from datetime import datetime, timezone


def test_replay_manifest_rejects_missing_recording(tmp_path: Path) -> None:
    manifest = tmp_path / "replay.yaml"
    manifest.write_text(
        """version: 1
replay_id: test
speed: 1.0
sources:
  - source_id: camera
    modality: video
    path: missing.mp4
    event_time_start: 2026-01-01T00:00:00Z
"""
    )
    with pytest.raises(FileNotFoundError):
        load_replay_manifest(manifest)


def test_resource_sampler_produces_monotonic_delta() -> None:
    start = sample_resources()
    end = sample_resources()
    delta = resource_delta(start, end)
    assert delta["process_cpu_seconds"] >= 0
    assert delta["tx_bytes"] >= 0
    assert delta["rx_bytes"] >= 0


def test_recording_discovery_is_explicit_and_read_only(tmp_path: Path) -> None:
    device = tmp_path / "mobile6"
    device.mkdir()
    recording = device / "recording_20251008_161305.mp4"
    recording.touch()
    (device / "notes.txt").touch()
    rows = discover_recordings(tmp_path)
    assert len(rows) == 1
    assert rows[0].path == recording
    assert rows[0].device == "mobile6"
    assert rows[0].timestamp_token == "20251008_161305"


def test_epoch_recording_timestamp_is_not_truncated(tmp_path: Path) -> None:
    recording = tmp_path / "recording_test_1728390781797.mp4"
    recording.touch()
    assert discover_recordings(tmp_path)[0].timestamp_token == "1728390781797"


def test_completion_artifact_is_durable_and_scoreable(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    artifact = CompletionArtifact("convoy", now, now, "camera1", {"vehicle": "v1"}, "cell-1")
    output = tmp_path / "completions.jsonl"
    CompletionWriter(output).append(artifact)
    loaded = load_completions(output)
    assert loaded == (artifact,)
    score = score_presence(True, loaded)
    assert score.true_positive == 1
    assert score.precision == 1.0
    assert score.recall == 1.0


def test_recording_map_requires_existing_unique_sources(tmp_path: Path) -> None:
    media = tmp_path / "camera.mp4"
    media.touch()
    mapping = tmp_path / "map.yaml"
    mapping.write_text(
        """version: 1
experiments:
  - experiment_id: exp-1
    verified: true
    recordings:
      - source_id: camera1
        path: camera.mp4
        start_offset_seconds: 2.5
""",
        encoding="utf-8",
    )
    rows = load_recording_map(mapping)
    assert rows[0].verified is True
    assert rows[0].recordings[0].path == media
