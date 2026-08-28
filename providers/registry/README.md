# Provider registry

A **provider** is one executable, typed computation. A **provider chain** is a
DAG wiring provider ports into a complete physical implementation of a semantic
predicate. Predicate IDs appear under `implements.predicates` because a
`PredicateDemand` names a semantic predicate and
`ProviderRegistry.candidate_chains(demand)` must discover chains capable of
returning its accepted result.

## Adding a provider in five steps

1. Add its contract and a complete chain to `catalog.yaml`.
2. Add genuinely new port types to `data_types.yaml`.
3. Add measured/default resource profiles for every eligible node class.
4. Map `(provider_id, node_id)` to a process, container, or reference runtime in
   deployment runtime configuration.
5. Add an output adapter only when native output is not already a normalized
   `PredicateResult`/declared artifact.

`examples/fake_object_detector.catalog.yaml` and its profile provide a small,
validated onboarding example without adding a fake option to production plans.

## Fields

- `description`: optional human documentation.
- `provider_family`: optional grouping/calibration metadata.
- `implements.predicates`: required for chain discovery/planning.
- `implements.role_capabilities`: required when results introduce/validate CE
  roles; otherwise optional.
- `ports.inputs/outputs`: required planning and execution interface.
- port `name`: wiring identity (`frames`, `detections`, `result`).
- port `type`: data compatibility (`raw_video_frames.v1`,
  `predicate_match.v1`).
- port `purpose`: descriptive/planning artifact-retention category.
- port `required`: execution wiring requirement; omitted inputs are required by
  default in the current adapter.
- `parameters`: execution configuration schema and planner validation metadata.
- `execution_capabilities.modes`: required execution-mode compatibility.
- `supports_shared_execution`: required for scheduler reuse decisions.
- `accepted_input_access`: required planning/execution constraint:
  `local` reads node-local data, `transferred` consumes copied bytes, and
  `remote_reference` dereferences a remotely hosted source/artifact. Advertise
  only modes implemented by the runtime.
- chain `external_inputs`, `steps`, `bind`, and `outputs`: required typed DAG.
- `compatibility_groups`: optional compatibility/reuse metadata.
- `eligible_node_classes`: optional hard placement restriction.

Resource profiles are planning inputs (startup/execution latency, CPU, RAM, GPU
memory, quality). Runtime mappings and output adapters are execution-only.
