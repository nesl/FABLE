# Repository architecture

The repository contains a stable FABLE core plus replaceable research-system
integration layers.

```text
ce_definitions/  authored complex-event semantics
fable/           language, semantic runtime, planning, execution, providers
config/          typed example deployment and node-agent configuration
evaluation/      manifests, baselines, measurements, and reports
netwaggle/       network emulation and runtime-link telemetry integration
replay/          dataset/replay adapters (ported separately from the old stack)
scripts/         small human-facing entry points
tests/           core and integration regression tests
```

## Dependency rule

Integration layers may import public objects from `fable`. Code under `fable/`
must not import `evaluation`, `netwaggle`, or `replay`.

```text
evaluation ─┐
netwaggle ──┼──> fable public APIs
replay ─────┘
```

The old repository remains migration input, not a runtime dependency. New code
must not add `/home/brianw/Documents/FABLE_old` to `PYTHONPATH` or import its
packages. Results, caches, model checkpoints, recordings, and generated files
remain outside source-controlled packages.

## Porting policy

1. Copy immutable data only after documenting its provenance.
2. Convert old executable code to the new public contracts instead of adding
   compatibility aliases to the core.
3. Keep experiment manifests separate from generated run state and results.
4. Validate every integration with a small smoke test before porting a full
   campaign.
5. Record the core source digest before and after integration work.
