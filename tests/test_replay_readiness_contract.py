from datetime import UTC, datetime
import json
from uuid import uuid4

from fable.distributed.models import ReplayReadiness
from scripts.run_replay_accuracy import decode_replay_readiness


def test_typed_readiness_round_trip() -> None:
    expected = ReplayReadiness(
        message_id=uuid4(), node_id="dvpg_gq_orin_11", service_id="zed",
        process_instance_id="process-1", generation=2, ready=True,
        scenario="20260414-three-visit-stalking", replay_id="replay-1",
    )
    decoded = decode_replay_readiness(
        "/readiness/dvpg_gq_orin_11/zed", expected.model_dump_json().encode()
    )
    assert decoded == expected


def test_legacy_readiness_is_confined_to_explicit_adapter() -> None:
    decoded = decode_replay_readiness(
        "/readiness/dvpg_gq_orin_11/zed",
        json.dumps({
            "node": "dvpg_gq_orin_11", "service": "zed", "ready": True,
            "pid": 42, "t": datetime(2026, 8, 21, tzinfo=UTC).timestamp(),
            "scenario": "scenario-1", "video_file": "/data/example.svo2",
        }).encode(),
    )
    assert decoded.process_instance_id == "legacy:42"
    assert decoded.metadata["video_file"] == "/data/example.svo2"

