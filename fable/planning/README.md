# FABLE physical planning

The planner answers one question:

> Given the current semantic frontier and runtime conditions, what provider
> computations should run, on which nodes, using which sources?

## Provider search

`provider_capabilities.yaml` remains an ordinary YAML catalog. It is **not**
converted into a persistent graph object.

`ProviderSearcher` builds a temporary producer index and performs bounded
backward AND/OR search over declared input/output types. For example:

```text
need predicate_match:near
        ↑
  near_geometry
        ↑
      tracks
        ↑
multi_object_tracker
        ↑
    detections
        ↑
       YOLO
        ↑
   video_frame
```

Adding a provider with compatible `inputs` and `outputs` automatically makes it
a candidate; no authored chain is required.

## Placement and plan selection

For each discovered provider recipe, `PhysicalPlanner` enumerates feasible source
and node placements and rejects alternatives that cannot run with current node,
network, resource, or semantic-expiry constraints.

Provider-produced intermediate stages are co-located by default. M13 now executes
those same-node provider DAGs as live workers; raw sensor input may be accessed by
the chosen compute node through its configured source adapter/URI.

Multiple frontier needs are combined with a bounded beam search. Plans are ranked
lexicographically by:

1. predicted completion time;
2. semantic-expiry slack;
3. number of new providers that must start;
4. transferred bytes;
5. peak resource pressure;
6. quality;
7. deterministic tie-break.

There is no weighted objective with arbitrary cross-unit coefficients.

## Runtime conditions

`runtime_state.py` intentionally contains only:

- `NodeState`
- `SourceState`
- `LinkState`
- `RunningProvider`
- `ProviderProfile`
- `RuntimeState`

Network measurement itself lives in `fable.execution.network_monitor`; planning
only consumes the resulting `LinkState` values.
