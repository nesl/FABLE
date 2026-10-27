# Replay

Replay adapts recorded media to FABLE's live source interfaces:

```text
video file -> OpenCVVideoSourceAdapter -> VideoFrame
WAV file   -> WaveAudioSourceAdapter   -> AudioWindow
```

Fixed cameras and mobile phones use the same video path. A source may specify
rotation, an event timestamp, a media offset, and multiple consecutive file
segments. All sources share one release barrier and event-time clock.

Phone audio embedded in video must currently be extracted to mono PCM WAV.
GPS and other generic mobile telemetry are not replay inputs.

## Cases

Each file in [`cases/`](cases) is one complete runnable case. It contains:

- `event`: the complex event FABLE should detect;
- `replay`: the synchronized video/audio source manifest;
- case metadata such as aliases, timeout, and completion requirement.

There is no separate case catalog or event-specific network experiment file.
The launcher discovers case files directly.

Media paths are container paths. `/recordings` is mounted from
`RECORDING_ROOT`, `/gq-recordings` from `GQ_RECORDING_ROOT`, and
`/replay-assets` from `.fable/runs/replay-assets`.

Network behavior is deliberately separate from cases. Reusable choices are in
[`plans`](plans).
