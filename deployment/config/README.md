# Deployment configuration

These examples document the YAML consumed by `scripts/run_fable.py` and
`scripts/run_node_agent.py`. They are deliberately outside `fable/`: a
deployment describes one installation, not complex-event semantics.

- `deployment.example.yaml` describes planning nodes, sensing sources, links,
  and node-agent control endpoints.
- `node_agent.example.yaml` describes the providers and local source adapters
  owned by one node agent.
- `deployment.mqtt.example.yaml` and `node_agent.mqtt.example.yaml` select the
  MQTT command/result plane. MQTT carries compact messages, not raw media.
Copy an example before adding machine-specific addresses or recording paths.
Do not commit credentials.

`historical_store.database` enables a node-local SQLite time index. By default
the agent records compact detection and track streams. Media segments can be
registered through `HistoricalStore.add_segment()` and retrieved by source and
time interval without copying media into the database.

## Replay input gating

A NodeAgent may gate replay input before expensive inference. Gating is off by
default. Configure it by source ID with two hysteresis thresholds:

```yaml
input_gates:
  camera1:
    type: video_frame_difference
    on_threshold: 0.04
    off_threshold: 0.02
  microphone1:
    type: audio_rms
    on_threshold: 0.03
    off_threshold: 0.015
```

The gate opens at or above `on_threshold`, stays in its previous state between
the thresholds, and closes at or below `off_threshold`. Video scores are mean
absolute grayscale frame differences normalized to `[0, 1]`; audio scores are
linear PCM RMS amplitudes in `[0, 1]`. These are replay transport policies and
do not change predicates, CE definitions, or provider output contracts.
