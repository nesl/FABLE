# FABLE execution runtime

This package closes the loop between a selected physical plan and **live provider computation**.
It keeps CE semantics, physical planning, and execution separate.

## Live control/data loop

```text
sensor / replay source
        ↓
provider workers on a node
        ↓
intermediate streams
        ↓
PredicateMatch / ReID association
        ↓
controller result transport
        ↓
IdentityResolver
        ↓
CEInstanceManager
        ↓
ActiveFrontier
        ↓
PhysicalPlanner
        ↓
ExecutionPlan
        ↓
PlanReconciler
        ↓
START / KEEP / STOP
        ↓
NodeAgent(s)
```

A runtime-condition change follows the same loop from `RuntimeState` onward. There is no separate replanning algorithm: update the state and call the planner again.

## Files

- `fable_runtime.py` — top-level closed-loop controller.
- `plan_reconciler.py` — pure `ExecutionPlan → START/KEEP/STOP` diff. New chains start downstream-first so consumers subscribe before producers begin emitting.
- `node_agent.py` — lifecycle API for one compute node plus an optional Linux resource probe.
- `dataflow_runtime.py` — canonical live provider backend: creates long-running provider workers and connects them on one node.
- `provider_worker.py` — small adapter between recovered provider classes and typed streams.
- `stream_bus.py` — same-node typed publish/subscribe bus. It is intentionally not a distributed streaming framework.
- `source_adapters.py` — manual/replay sources, OpenCV video/RTSP input, and WAV audio replay.
- `result_transport.py` — asynchronous provider-result path from node workers back to the controller.
- `provider_runtime.py` — simpler in-process, subprocess, and Docker lifecycle backends retained for tests/external processes.
- `command_transport.py` — direct transport plus the TCP JSON START/STOP/STATUS control plane.
- `network_monitor.py` — online ping, on-demand iperf3, and passive throughput.
- `identity_resolver.py` / `reid.py` — object identity canonicalization and ReID.
- `local_runner.py` — compact local/replay execution path used by tests.

## Provider workers and the same-node data plane

A planned provider no longer means only "a process exists." Under `DataflowProviderRuntime`, it becomes a live worker:

```text
video_frame
   ↓
yolo_full_context_960
   ↓ detections
multi_object_tracker
   ↓ tracks
enters_basic / near_geometry / follows_local_geometry
   ↓
PredicateMatch
```

Workers subscribe to their declared input types from `provider_capabilities.yaml` and publish their declared output type. Shared prefixes are physically shared: one `TrackFrame` can feed several predicate workers.

The same-node data plane uses the existing small provider models (`VideoFrame`, `DetectionFrame`, `TrackFrame`, `AudioWindow`, embeddings, etc.). There is no new artifact hierarchy.

### Current deliberate boundary

Provider-produced intermediate stages are kept on the same compute node by default. This avoids adding arbitrary cross-node serialization, buffering, backpressure, and recovery machinery. Raw camera/audio sources may be remotely accessible through ordinary deployment URIs (for example RTSP), allowing an edge node to run the selected chain without FABLE inventing a video transport protocol.

If a future experiment truly requires detector@sensor → tracker@edge → predicate@cloud, that can be added later. It is not required for the current FABLE system claim.

## Source adapters

`source_adapters.py` provides:

- `ManualSourceAdapter` — tests and external replay harnesses can push already-constructed source values.
- `IterableSourceAdapter` — finite scripted replay.
- `OpenCVVideoSourceAdapter` — camera index, video file, or RTSP/other OpenCV URI → `VideoFrame`.
- `WaveAudioSourceAdapter` — mono 16-bit PCM WAV → `AudioWindow`.

These are deployment adapters, not CE semantics.

## Provider result transport

Lifecycle commands and results use separate channels:

```text
controller → node
    START / STOP / STATUS

node → controller
    PredicateMatch
    IdentityAssociation
```

`DirectResultTransport` is used in tests/single-process deployments. `TcpResultTransport` and `ResultTCPServer` provide a tiny newline-delimited JSON result channel for distributed nodes.

Terminal predicate results call `FableRuntime.handle_predicate_match()` automatically. ReID results call the identity resolver separately; ReID still does **not** imply CE-instance deduplication.

## Provider readiness and replacement

`DataflowProviderRuntime.start()` returns only after the worker is created, subscribed to all of its inputs, and ready to receive values. `NodeAgent.start()` reports `ready` for this backend.

When changing plans, FABLE starts every replacement provider before stopping obsolete work. If replacement startup fails, old work is not intentionally removed.

## Provider instance identity

A running provider remains identified by only:

```text
(provider_id, node_id, source_ids)
```

If the same key occurs in the old state and new plan it is `KEEP`; new-only is `START`; old-only is `STOP`.

## Node-agent configuration

The standard node-agent path now uses `runtime: dataflow`:

```yaml
node:
  node_id: edge1
  node_type: edge

runtime: dataflow

controller_results:
  host: 10.0.0.10
  port: 8766

sources:
  camera3:
    type: video
    uri: rtsp://camera3/stream
```

Then:

```bash
python scripts/run_node_agent.py node_agent.yaml
```

`runtime: subprocess` remains available for externally managed provider executables.

The controller starts a result receiver automatically:

```bash
python scripts/run_fable.py ce_definitions/<event>.yaml deployment.yaml
```

## Network measurement

Network measurement lives in execution because measurement is runtime telemetry, not planning.

Normal real-world operation should use:

1. **ping** reasonably often for reachability and RTT;
2. **passive transfer throughput** from FABLE's real transfers for the steady-state throughput estimate;
3. **iperf3 occasionally/on demand** when a fresh active throughput estimate is useful.

`iperf3` measures achievable throughput, but consumes network capacity while doing so. The planner sees only `LinkState`; it never imports ping/iperf code.
