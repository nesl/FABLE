from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "iobt-minimal-ce-replay/services/analytics/yolo_detector/app/remote_frame.py"
)
SPEC = importlib.util.spec_from_file_location("fable_yolo_remote_frame", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
decode_remote_frame = MODULE.decode_remote_frame


def test_typed_remote_frame_decodes_image_and_preserves_replay_provenance() -> None:
    image = b"\xff\xd8synthetic-jpeg\xff\xd9"
    document = {
        "schema_version": "fable.remote_camera_frame.v1",
        "data": base64.b64encode(image).decode("ascii"),
        "t": 1_728_403_916.25,
        "replay_id": "replay-r013",
        "frame_number": 37,
    }

    decoded = decode_remote_frame(json.dumps(document))

    assert decoded == {
        "image": image,
        "event_time": 1_728_403_916.25,
        "replay_id": "replay-r013",
        "frame_number": 37,
        "schema_version": "fable.remote_camera_frame.v1",
    }


def test_legacy_binary_and_base64_frames_remain_supported() -> None:
    image = b"\x89PNG\r\nsynthetic"

    assert decode_remote_frame(image)["image"] == image
    assert decode_remote_frame(base64.b64encode(image))["image"] == image
