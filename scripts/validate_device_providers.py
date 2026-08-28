#!/usr/bin/env python3
"""Run dependency-light real provider implementations on physical edge devices."""

from __future__ import annotations

import argparse
from datetime import timedelta
from time import perf_counter

import numpy as np

from fable.common.time import EventTimeInterval, utc_now
from providers.multimodal.audio import SpectralRuleAudioBackend
from providers.multimodal.localization import GccPhatAudioLocalizer
from providers.multimodal.models import (
    AudioWindow,
    MicrophoneArrayGeometry,
    MicrophonePosition,
)
from providers.vehicle.geometry import MotionStateEvaluator
from providers.vehicle.models import (
    BoundingBox,
    Point2D,
    TrackObservation,
    TrackSet,
    scoped_track_identity,
)


def validate_pi() -> None:
    sample_rate = 16_000
    count = sample_rate
    impulse = np.zeros(count, dtype=np.float32)
    impulse[count // 2] = 1.0
    delayed = np.roll(impulse, 2)
    now = utc_now()
    window = AudioWindow(
        source_id="physical_rpi:synthetic_audio",
        event_time_interval=EventTimeInterval(start=now, end=now + timedelta(seconds=1)),
        sample_rate_hz=sample_rate,
        channel_ids=("mic0", "mic1"),
        waveform=(tuple(impulse), tuple(delayed)),
    )
    started = perf_counter()
    scores = SpectralRuleAudioBackend().score(window)
    spectral_ms = (perf_counter() - started) * 1000
    geometry = MicrophoneArrayGeometry(
        array_id="synthetic_two_mic",
        coordinate_frame_id="device",
        microphones=(
            MicrophonePosition(microphone_id="mic0", x_m=0.0, y_m=0.0),
            MicrophonePosition(microphone_id="mic1", x_m=0.08, y_m=0.0),
        ),
        reference_microphone_id="mic0",
    )
    started = perf_counter()
    location = GccPhatAudioLocalizer(geometry).localize(window)
    gcc_ms = (perf_counter() - started) * 1000
    print(
        f"PI_PROVIDER_OK spectral_ms={spectral_ms:.3f} gcc_phat_ms={gcc_ms:.3f} "
        f"scores={scores} azimuth_deg={location.azimuth_deg:.2f}"
    )


def _track(second: int, x: float) -> TrackObservation:
    event_time = utc_now() + timedelta(seconds=second)
    return TrackObservation(
        local_track_id=1,
        scoped_track_id=scoped_track_identity(
            "physical_jetson:fixture_camera", "fixture", 1
        ),
        source_id="physical_jetson:fixture_camera",
        tracker_session_id="fixture",
        class_name="car",
        confidence=0.95,
        bbox=BoundingBox(x1=x, y1=0, x2=x + 2, y2=4),
        event_time=event_time,
        world_point=Point2D(x=x, y=0, coordinate_frame_id="world"),
    )


def validate_jetson() -> None:
    evaluator = MotionStateEvaluator(minimum_window_s=1.0)
    outputs = ()
    started = perf_counter()
    for second, x in ((0, 0.0), (2, 4.0)):
        track = _track(second, x)
        outputs = evaluator.update(
            TrackSet(
                source_id=track.source_id,
                tracker_family="fixture",
                tracker_version="1",
                tracker_session_id="fixture",
                event_time=track.event_time,
                tracks=(track,),
            )
        )
    elapsed_ms = (perf_counter() - started) * 1000
    predicates = [item.predicate_id for item in outputs]
    if "MOVING" not in predicates:
        raise RuntimeError(f"motion evaluator produced {predicates!r}, expected MOVING")
    print(f"JETSON_PROVIDER_OK motion_ms={elapsed_ms:.3f} predicates={predicates}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("pi", "jetson"))
    args = parser.parse_args()
    validate_pi() if args.role == "pi" else validate_jetson()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
