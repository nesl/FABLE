# Replay integration

This directory is the destination for the IoBT recording/testbed port. Replay
is intentionally an adapter around `fable.execution.SourceAdapter`, not a
second semantic runtime.

Initial mapping:

```text
video/RTSP recording -> OpenCVVideoSourceAdapter -> VideoFrame
WAV recording        -> WaveAudioSourceAdapter   -> AudioWindow
provider result      -> PredicateMatch           -> FableRuntime
```

The large legacy Docker tree is not copied here yet because its orchestrator,
NodeAgent, provider contracts, and campaign scripts target the old FABLE API.
Only recording discovery, clock synchronization, and source-adapter behavior
should be retained during that port.

`replay.discovery.discover_recordings()` indexes media under a caller-supplied
root. It never guesses mount points or mounts, moves, or rewrites recordings.
