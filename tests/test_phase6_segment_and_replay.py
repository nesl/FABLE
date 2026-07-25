from __future__ import annotations

from datetime import timedelta
import json

from fable.common.examples import BASE_TIME
from fable.common.time import EventTimeInterval
from fable.distributed.heartbeat import ReplaySourceProgressTracker
from fable.distributed.models import SegmentRef
from fable.distributed.segment_store import SegmentStore


def test_segment_store_interval_query_coverage_and_expiry(tmp_path):
    media = tmp_path / "segment.wav"
    media.write_bytes(b"fake audio")
    store = SegmentStore(tmp_path / "segments.sqlite")
    interval = EventTimeInterval(start=BASE_TIME, end=BASE_TIME + timedelta(seconds=10))
    segment = SegmentRef(
        source_id="mic",
        path=str(media),
        event_time_interval=interval,
        bytes=media.stat().st_size,
        created_at=BASE_TIME,
        expires_at=BASE_TIME + timedelta(minutes=1),
    )
    assert store.register(segment)
    assert store.covers(source_id="mic", interval=interval, now=BASE_TIME + timedelta(seconds=30))
    assert store.query(source_id="mic", interval=EventTimeInterval(start=BASE_TIME + timedelta(seconds=2), end=BASE_TIME + timedelta(seconds=3)), now=BASE_TIME + timedelta(seconds=30)) == (segment,)
    expired = store.expire(now=BASE_TIME + timedelta(minutes=2), delete_files=True)
    assert expired == (segment.segment_id,)
    assert not media.exists()


def test_replay_progress_tracker_understands_existing_status_and_detector_topics(tmp_path):
    store = SegmentStore(tmp_path / "segments.sqlite")
    tracker = ReplaySourceProgressTracker(node_id="dvpg_gq_orin_11", segment_store=store)
    status = {
        "start_time": "2026-04-14T18:19:51+00:00",
        "end_time": "2026-04-14T18:20:51+00:00",
        "current": 12.5,
        "event": "progress",
    }
    source = tracker.update(
        "/replay/status/zed/dvpg_gq_orin_11", json.dumps(status).encode()
    )
    assert source.source_id == "dvpg_gq_orin_11:camera"
    assert source.raw_buffer_interval.start == BASE_TIME
    assert source.latest_event_time == BASE_TIME + timedelta(seconds=12.5)

    yolo = tracker.update(
        "/dvpg_gq_orin_11/analytics/yolo/bbox",
        json.dumps([{"t": "2026/04/14 18:20:04.000000", "class": "car"}]).encode(),
    )
    assert yolo.latest_event_time == BASE_TIME + timedelta(seconds=13)

    audio = tracker.update(
        "/dvpg_gq_orin_11/audio_detector/detections",
        json.dumps({"t": BASE_TIME.timestamp() + 14, "event": "loud_audio"}).encode(),
    )
    assert audio.source_id == "dvpg_gq_orin_11:audio"
    assert audio.latest_event_time == BASE_TIME + timedelta(seconds=14)
