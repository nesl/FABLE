from datetime import datetime, timezone

import pytest

from fable.execution import HysteresisInputGate, InputGateConfig
from fable.providers.data_models import AudioWindow, VideoFrame


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_video_gate_uses_separate_open_and_close_thresholds() -> None:
    np = pytest.importorskip("numpy")
    gate = HysteresisInputGate(
        InputGateConfig("video_frame_difference", 0.20, 0.05)
    )
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    white = np.full((8, 8, 3), 255, dtype=np.uint8)
    mid = np.full((8, 8, 3), 230, dtype=np.uint8)

    assert not gate.accept(VideoFrame("camera", NOW, black, "0"))
    assert gate.accept(VideoFrame("camera", NOW, white, "1"))
    # Difference is between the off and on thresholds, so the open state holds.
    assert gate.accept(VideoFrame("camera", NOW, mid, "2"))
    assert not gate.accept(VideoFrame("camera", NOW, mid, "3"))


def test_audio_gate_uses_rms_hysteresis() -> None:
    gate = HysteresisInputGate(InputGateConfig("audio_rms", 0.20, 0.05))
    window = lambda samples: AudioWindow("mic", NOW, tuple(samples), 16_000)

    assert not gate.accept(window((0.0, 0.0)))
    assert gate.accept(window((0.3, -0.3)))
    assert gate.accept(window((0.1, -0.1)))
    assert not gate.accept(window((0.01, -0.01)))


def test_gate_rejects_inverted_thresholds() -> None:
    try:
        InputGateConfig("audio_rms", 0.1, 0.1)
    except ValueError as exc:
        assert "off_threshold" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid thresholds were accepted")
