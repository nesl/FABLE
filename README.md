# FABLE minimal rebuild

This repository rebuilds FABLE as a small closed-loop system whose semantic event progress controls real distributed perception execution.

Implemented path:

```text
CE YAML
  → parse + compile
  → CEInstanceManager
       → persistent discovery frontier
       → candidate-specific continuation frontiers
  → dynamic typed provider search
  → runtime-aware physical placement + beam selection
  → ExecutionPlan
  → START / KEEP / STOP reconciliation
  → node agents
  → live same-node provider dataflow
       source → detector → tracker/features → predicate implementation
  → PredicateMatch / ReID association
  → controller result transport
  → identity canonicalization
  → CEInstanceManager again
  → completed CE instances
```

Main folders:

- `fable/language/` — CE grammar, parser, predicate vocabulary, and capability-aware compiler.
- `fable/providers/` — perception/intermediate providers, predicate implementations, label/capability declarations, ReID backends, and `PredicateMatch`.
- `fable/runtime/` — candidate CE instances, discovery/continuation frontier semantics, and instance management.
- `fable/planning/` — dynamic provider search, runtime conditions, placement enumeration, and beam plan selection.
- `fable/execution/` — live provider workers/dataflow, identity/ReID, plan reconciliation, node agents, control/result transport, online network measurement, and the top-level runtime.
- `ce_definitions/` — authored complex-event YAML files.
- `evaluation/` — manifests, baseline policies, metrics, and generated reports.
- `netwaggle/` — network-emulation integration and measured link-state updates.
- `replay/` — recording/testbed adapters around the execution source API.
- `config/` — deployment and node-agent configuration examples.

The core/integration dependency policy is documented in `ARCHITECTURE.md`.

The frontier is the semantic-to-physical interface. There is no separate `EvidenceDemand` layer and there is no hand-authored provider-chain catalog.

## Installation

The project uses **only `pyproject.toml`** for Python dependency/configuration metadata.

Core/tests:

```bash
pip install -e ".[dev]"
```

Vision providers and video source adapters:

```bash
pip install -e ".[vision]"
```

ReID providers:

```bash
pip install -e ".[reid]"
python scripts/provision_reid_models.py
```

`ping` and `iperf3` are operating-system tools. Runtime networking uses ping as the lightweight probe, passive throughput from real transfers as the normal bandwidth signal, and iperf3 only as an explicit/occasional throughput calibration.

## Distributed controller/agent entry points

Start each node agent with a YAML configuration:

```bash
python scripts/run_node_agent.py node_agent.yaml
```

`runtime: dataflow` is the normal mode: the node agent creates long-running provider workers, connects their typed streams locally, and sends terminal results back to the controller.

Start the controller with:

```bash
python scripts/run_fable.py ce_definitions/<event>.yaml deployment.yaml
```

The controller exposes a small result receiver (default port 8766); node-agent configs point `controller_results` at it.

## Deliberate boundary

Provider-produced intermediates remain on one selected compute node. Raw sensor streams may be accessed remotely through deployment URIs such as RTSP. Arbitrary detector@sensor → tracker@edge → predicate@cloud pipelines are intentionally not implemented unless an experiment proves they are necessary.

With M13, the core FABLE architecture is complete. Remaining work should primarily be replay/testbed/NetWaggle integration, measured provider profiling, logging, baselines, and experiments rather than additional framework machinery.
