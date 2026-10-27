# FABLE architecture

The repository has three implementation layers and a small operational shell:

```text
fable/       reusable CE language, providers, planner, runtime, execution, UI
replay/      self-contained cases and synchronized source adapters
netwaggle/   network topology, measurement, and condition control
deployment/ containers, workers, broker settings, and configuration
scripts/     command implementations invoked by ./run
```

## Dependency boundary

`fable/` is the reusable core. It must not import `replay`, `netwaggle`, or
`deployment`.

```text
replay ─────┐
netwaggle ──┼──> fable public APIs
deployment ─┘
```

Replay and network experiments are integration layers. Generated state,
recordings, models, caches, and run outputs remain outside source-controlled
packages under the ignored `.fable/` directory or external recording mounts.

## Execution boundary

The controller compiles a complex event and turns its current semantic
frontier into a physical execution plan. Node agents start, keep, or stop
node-local providers in response to that plan.

MQTT carries compact lifecycle commands, predicate matches, identity
associations, and status. Raw video/audio and intermediate detections stay on
the node running the provider dataflow. A node-local SQLite index may retain
compact historical evidence and references to externally stored media.

## Replay boundary

`replay/` adapts files to the same source interfaces used by live execution.
Video becomes `VideoFrame`; WAV audio becomes `AudioWindow`. A shared start
barrier and event timestamps synchronize fixed cameras, mobile recordings, and
audio without creating a second CE runtime.

Each `replay/cases/*.yaml` file is the only case-specific configuration: its
`event` block defines the CE and its `replay` block defines the media sources.
Reusable `replay/plans/*.yaml` files independently select direct networking
or a NetWaggle condition.

## Deliberate limitation

Tightly coupled intermediates remain on one selected compute node. Sources may
be remote, but arbitrary detector-on-sensor to tracker-on-edge pipelines are
outside the current execution model.
