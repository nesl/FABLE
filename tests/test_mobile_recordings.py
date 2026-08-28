from datetime import datetime
from pathlib import Path

from evaluation import mobile_recordings


def test_resolver_uses_timestamp_and_keeps_archive_identity(tmp_path, monkeypatch):
    archive = tmp_path / "mobile 1 all"
    archive.mkdir()
    short = archive / "recording_spatial_ce1_1_1728396745810.mp4"
    covering = archive / "recording_spatial_ce1_1_1728396746000.mp4"
    short.write_bytes(b"x")
    covering.write_bytes(b"x" * 20)
    monkeypatch.setattr(
        mobile_recordings,
        "_duration",
        lambda path: 1.0 if path == short else 300.0,
    )

    rows = mobile_recordings.resolve_mobile_recordings(
        tmp_path,
        recording_prefix="spatial_ce1_1",
        event_start=datetime.fromisoformat("2024-10-08T10:12:30"),
        event_end=datetime.fromisoformat("2024-10-08T10:13:00"),
    )

    assert len(rows) == 1
    assert rows[0].archive_id == "mobile_archive_1"
    assert rows[0].logical_id == "mobile_archive_1"
    assert rows[0].path == covering.resolve()


def test_resolver_applies_only_explicit_run_alias(tmp_path, monkeypatch):
    archive = tmp_path / "mobile 2 all"
    archive.mkdir()
    path = archive / "recording_temporal_ce1_1_1728400105601.mp4"
    path.write_bytes(b"x")
    monkeypatch.setattr(mobile_recordings, "_duration", lambda _path: 300.0)

    rows = mobile_recordings.resolve_mobile_recordings(
        tmp_path,
        recording_prefix="temporal_ce1_1",
        event_start=datetime.fromisoformat("2024-10-08T11:08:30"),
        event_end=datetime.fromisoformat("2024-10-08T11:09:00"),
        alias_map={"mobile_archive_2": "n3"},
    )

    assert rows[0].logical_id == "n3"


def test_resolver_can_match_legacy_prefixes_by_time_without_name_guess(
    tmp_path, monkeypatch
):
    archive = tmp_path / "mobile 1 all"
    archive.mkdir()
    unrelated = archive / "recording_spatial_ce1_1_1728396746000.mp4"
    covering = archive / "recording_temporal_ce1_13_1728409565000.mp4"
    unrelated.write_bytes(b"old")
    covering.write_bytes(b"covering")
    monkeypatch.setattr(mobile_recordings, "_duration", lambda _path: 300.0)

    rows = mobile_recordings.resolve_mobile_recordings(
        tmp_path,
        recording_prefix=None,
        allow_any_prefix=True,
        event_start=datetime.fromisoformat("2024-10-08T13:49:52"),
        event_end=datetime.fromisoformat("2024-10-08T13:51:26"),
    )

    assert len(rows) == 1
    assert rows[0].path == covering.resolve()


def test_resolver_supports_2025_timestamp_only_recordings(tmp_path, monkeypatch):
    archive = tmp_path / "mobile 6 all"
    archive.mkdir()
    covering = archive / "recording__1755029138219.mp4"
    pending = archive / ".pending-1-recording__1755029138219.mp4"
    covering.write_bytes(b"complete")
    pending.write_bytes(b"partial")
    monkeypatch.setattr(mobile_recordings, "_duration", lambda _path: 180.0)

    rows = mobile_recordings.resolve_mobile_recordings(
        tmp_path,
        recording_prefix=None,
        event_start=datetime.fromisoformat("2025-08-12T16:05:35"),
        event_end=datetime.fromisoformat("2025-08-12T16:05:56"),
    )

    assert len(rows) == 1
    assert rows[0].archive_id == "mobile_archive_6"
    assert rows[0].path == covering.resolve()


def test_resolver_combines_adjacent_timestamp_recordings(tmp_path, monkeypatch):
    archive = tmp_path / "mobile 6 all"
    archive.mkdir()
    first = archive / "recording__1755029585000.mp4"
    second = archive / "recording__1755029635000.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    monkeypatch.setattr(mobile_recordings, "_duration", lambda _path: 50.0)

    rows = mobile_recordings.resolve_mobile_recordings(
        tmp_path,
        recording_prefix=None,
        event_start=datetime.fromisoformat("2025-08-12T16:13:05"),
        event_end=datetime.fromisoformat("2025-08-12T16:14:35"),
    )

    assert len(rows) == 1
    assert [item.path for item in rows[0].segments] == [
        first.resolve(),
        second.resolve(),
    ]
    assert sum(
        item.trim_end_seconds - item.trim_start_seconds
        for item in rows[0].segments
    ) == 90.0


def test_resolver_does_not_reject_minute_of_event_evidence_due_to_long_padding(
    tmp_path, monkeypatch
):
    archive = tmp_path / "mobile 6 all"
    archive.mkdir()
    first = archive / "recording__1755029593000.mp4"
    second = archive / "recording__1755029635000.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    monkeypatch.setattr(
        mobile_recordings,
        "_duration",
        lambda path: 42.0 if path == first else 30.0,
    )

    rows = mobile_recordings.resolve_mobile_recordings(
        tmp_path,
        recording_prefix=None,
        event_start=datetime.fromisoformat("2025-08-12T16:13:05"),
        event_end=datetime.fromisoformat("2025-08-12T16:15:30"),
    )

    assert len(rows) == 1
    assert len(rows[0].segments) == 2


def test_resolver_skips_directory_named_like_mp4(tmp_path, monkeypatch):
    archive = tmp_path / "mobile 4 all"
    archive.mkdir()
    malformed = archive / "recording__1755029095975.mp4"
    malformed.mkdir()
    covering = archive / "recording__1755029096000.mp4"
    covering.write_bytes(b"valid")
    monkeypatch.setattr(mobile_recordings, "_duration", lambda _path: 300.0)

    rows = mobile_recordings.resolve_mobile_recordings(
        tmp_path,
        recording_prefix=None,
        event_start=datetime.fromisoformat("2025-08-12T16:05:35"),
        event_end=datetime.fromisoformat("2025-08-12T16:05:56"),
    )

    assert len(rows) == 1
    assert rows[0].path == covering.resolve()


def test_resolver_skips_unprobeable_recording(tmp_path, monkeypatch):
    archive = tmp_path / "mobile 4 all"
    archive.mkdir()
    malformed = archive / "recording__1755029095975.mp4"
    covering = archive / "recording__1755029096000.mp4"
    malformed.write_bytes(b"bad")
    covering.write_bytes(b"valid")

    def duration(path):
        if path == malformed:
            raise ValueError("invalid duration")
        return 300.0

    monkeypatch.setattr(mobile_recordings, "_duration", duration)
    rows = mobile_recordings.resolve_mobile_recordings(
        tmp_path,
        recording_prefix=None,
        event_start=datetime.fromisoformat("2025-08-12T16:05:35"),
        event_end=datetime.fromisoformat("2025-08-12T16:05:56"),
    )

    assert len(rows) == 1
    assert rows[0].path == covering.resolve()
