# FABLE Runtime Prototype

FABLE is a frontier-driven runtime for distributed complex-event recognition. The repository now contains the complete control-plane path from an authored complex-event request to semantic frontier tracking, physical planning, multi-tenant admission, distributed provider execution, and feedback from predicate results into the next frontier.

## Runtime architecture

<!-- Runtime architecture image placeholder. -->

_A runtime architecture figure will be inserted here._

## Quick start

FABLE requires Python 3.11 or newer. A local editable installation with the
development dependencies is enough to run the deterministic examples and test
the complete pipeline; Docker, MQTT, GPUs, videos, and physical devices are not
required for this first test.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

### Inspect one pipeline pass

The architecture walkthrough prints each intermediate object as a request
moves through semantic frontier creation, demand compilation, physical
alternative generation, bounded planning, admission, provider leasing, and
predicate-result feedback:

```bash
python examples/fable_architecture_walkthrough.py
```

The output is numbered from the selected semantic graph through the updated
hypothesis and frontier. This command is deliberately deterministic and uses
in-process fixtures, making it useful for understanding the data passed between
the major packages.

### Run the end-to-end smoke test

The following test exercises the closed control and data path: typed event
submission, request compilation, semantic frontier creation, demand
compilation, planning, scheduling, orchestration, an in-memory transport, a
node agent, reference-provider execution, predicate-result ingestion,
replanning for the next frontier, and final complex-event emission.

```bash
pytest -q \
  tests/test_architecture_upgrade.py::test_event_request_api_runs_semantic_planning_feedback_loop
```

A successful run ends with `1 passed`. The provider and message broker are
deterministic in-process implementations, so this validates FABLE itself
without claiming that a physical sensor, model, or network testbed was used.

For an interactive version that exposes the event graph, active frontier,
predicate demand, execution plan, scheduler decision, node-agent command,
fake replay result, and updated hypothesis, install the notebook dependencies
and open the architecture walkthrough. FABLE requires Python 3.11 or newer;
create the environment with an explicit 3.11 interpreter if the system
`python3` command is older:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebook]'
python -m jupyter lab examples/fable_architecture_pipeline.ipynb
```

The notebook locates the repository checkout automatically, so it does not
require a separately registered `FABLE` Jupyter kernel. Its selected Python
environment still needs the project dependencies; the editable install above
provides both those dependencies and JupyterLab.

Do not launch a user- or system-level `jupyter` executable backed by Python
3.10. Python 3.10 does not provide `enum.StrEnum`, and the import cell will fail
with `cannot import name 'StrEnum' from 'enum'`. Launching Jupyter as
`python -m jupyter` from the activated environment ensures that the server and
kernel use the intended interpreter. Verify the active kernel in the first
notebook cell:

```python
import sys
print(sys.executable)
print(sys.version)
```

The executable should end in `FABLE/.venv/bin/python`, and the reported version
must be 3.11 or newer. If it does not, stop the Jupyter server, activate
`.venv`, and relaunch it with `python -m jupyter` as shown above.

To run every unit and integration test that does not require an externally
configured physical deployment:

```bash
pytest -q
```

## Major packages

- `fable/contracts/` — canonical transportable data contracts, grouped by architectural ownership. `fable/common/schemas.py` is now only a compatibility facade.
- `fable/catalog/` — data-driven predicate/provider catalogs plus generic catalog loading and validation.
- `fable/semantic/` — hypotheses, transitions, composite/temporal evaluation, frontiers, and checkpoints. Authored event families live in `fable/events/`.
- `fable/planning/` — demand compilation, provider/data-type registry, artifact catalog, physical alternatives, bounded planning, and live deployment overlays. Alternative construction and bounded search are decomposed into focused subpackages.
- `fable/scheduling/` — admission, capacity accounting, provider lifecycle, lease sharing, cancellation, and checkpoint control. Scheduling consumes a self-contained `ExecutionPlan` rather than planner-internal physical alternatives.
- `fable/distributed/` — transport, durable outbox/state, node agents, worker/runtime mapping, dependency-aware physical-plan dispatch, heartbeats, failure detection, and recovery. The node agent is provider-agnostic.
- `fable/orchestration/` — deployed closed-loop controller joining semantic, planning, scheduling, and distributed execution.
- `fable/integrations/` — testbed/provider-specific adapters such as replay output normalization, NetWaggle telemetry, and synthetic reference execution.
- `fable/debug/` and `fable/testing/` — explicit low-level demo/test support that is kept out of runtime packages.
- `providers/` — concrete provider implementations; these depend on FABLE's generic catalog/adapter contracts rather than being imported by the core planner.
- `evaluation/` — planner/system evaluation support.
- `iobt-minimal-ce-replay/` — Docker replay/testbed integration and FABLE service deployment.
- `netwaggle/` — network shaping/profile publication used by runtime replanning.

## Network testbed dependency

FABLE's emulated-network experiments build on the physical/virtual host model
from [IoCT-Testbed-Simulation](https://github.com/nesl/IoCT-Testbed-Simulation).
That project supplies the underlying Mininet/Mininet-WiFi testbed design: a
coordinator machine places virtual hosts and externally attached physical
devices behind configurable switches or access points, while forwarders and
Linux network namespaces ensure application traffic traverses the emulated
links. This is the mechanism that lets an otherwise unmodified distributed
application experience controlled latency, bandwidth, loss, topology, and
mobility conditions.

The local `netwaggle/` package is FABLE's experiment-specific integration of
that approach. It:

- maps FABLE logical nodes and Docker service groups onto Mininet-attached
  network namespaces;
- supports the host, Raspberry Pi, and Jetson/Orin as physical execution
  tiers, alongside virtual nodes used during development;
- applies repeatable wired link profiles and scheduled condition changes;
- publishes link/profile state into FABLE's runtime deployment view so network
  changes can trigger physical replanning; and
- records interface, queueing-discipline, latency, and MQTT traffic metrics for
  paired evaluation runs.

FABLE does not vendor or replace the complete upstream simulator. The code in
`netwaggle/` contains the narrower wired Mininet/traffic-control path and
Compose integration needed by this repository. The upstream project remains
the reference for the general physical-host forwarder architecture,
Mininet-WiFi mobility support, PcapPlusPlus setup, and topology configuration
format. See [`netwaggle/README.md`](netwaggle/README.md) for the FABLE-specific
bring-up and cleanup commands.

## Execution profiles

The deployed FABLE service accepts `FABLE_EXECUTION_PROFILE`:

- `development` — normal development behavior; reference providers may be used.
- `plumbing` — intended for control-plane/distributed-substrate tests with synthetic/reference providers.
- `real` — experimental mode. Reference runtimes are rejected, and multi-step alternatives that require an unimplemented cross-worker intermediate-data transfer are pruned rather than pretending they are executable.

The evaluation launcher defaults to `real` and validates the Compose configuration before starting.

## Physical workers vs. logical providers

A provider runtime may declare a `worker_id` and `worker_resource_limits`. Multiple logical provider capabilities can therefore share one physical process/container. Capacity is charged once to the physical worker while demand-specific leases remain independently attachable/cancellable. This matches the replay vehicle and multimodal services, which expose multiple logical operations from one warm container.

## Live replanning inputs

Planning uses a `RuntimeDeploymentView` layered over the authored deployment:

- node heartbeats cap planning capacity by currently observed free CPU/RAM/GPU;
- node availability/failure state removes unavailable placements;
- a generic network-telemetry interface updates current latency/bandwidth links; the testbed injects a NetWaggle adapter;
- admission still accounts for FABLE's own active reservations.

## Event submission

The replay stack includes `iobt-minimal-ce-replay/tools/fable_submit_event.py`, which publishes a typed complex-event request to the orchestrator. Existing `fable_submit_*` plan-candidate tools remain useful for dispatch/plumbing tests but intentionally bypass semantic compilation/planning.

## Validation

Run the full suite from the repository root:

```bash
pytest -q
```

For a real evaluation configuration:

```bash
cd iobt-minimal-ce-replay
FABLE_EXECUTION_PROFILE=real python tools/validate_fable_evaluation_config.py
```

See `ARCHITECTURE_UPGRADE.md` for the closed-loop runtime upgrade and `ARCHITECTURE_REFACTOR.md` for the separation-of-concerns refactor.
