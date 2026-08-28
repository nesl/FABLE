# FABLE contracts

This package defines the typed schemas exchanged between architecture modules.
Contracts describe data, identity, and compatibility; they do not implement CE
progression, planning, scheduling, or execution. `fable.common.schemas` is a
compatibility facade that preserves the same canonical class identities.

| Module | Principal concepts | Typical producer | Typical consumer |
|---|---|---|---|
| `_shared.py` | shared scalar types, enums, validators | contract modules | all modules |
| `semantic.py` | roles, predicates, graph nodes/edges, `SemanticGraph` | CE compiler | semantic runtime |
| `hypothesis.py` | bindings, hypothesis progress, checkpoint, frontier snapshot | semantic runtime | demand compiler/controller |
| `demand.py` | provider-independent `PredicateDemand` | demand compiler | physical planner |
| `provider.py` | provider ports/contracts/families and catalog data | provider registry | planner/executor |
| `artifact.py` | retained/available evidence references | node agents/artifact catalog | planner/executor |
| `execution.py` | concrete steps, costs, labels, `ExecutionPlan` | physical planner | scheduler/node execution |
| `scheduling.py` | demand-scoped `ProviderLease` | lifecycle manager | node agent |
| `result.py` | normalized predicate evidence and terminal CE output | provider adapter/runtime | semantic runtime/application |
| `telemetry.py` | heartbeat, capacity, source progress, runtime updates | node agent/monitor | runtime deployment view/planner |

## Identity and version vocabulary

- `provider_id`: stable logical implementation, such as
  `yolo_vehicle_fast_640`.
- `provider_contract_version`: version of that implementation's declared
  interface and semantics. It is **not** a trained-model version.
- `provider_instance_id`: one currently running or warm logical instance.
- `model_id` / `model_version`: actual ML model and weights identity, when used.
- `lease_id`: one demand's permission to use a provider instance.
- `plan_id`: concrete physical execution plan that created the lease.
- `demand_id`: provider-independent semantic evidence request being served.
- `attempt_id`: one activation/execution attempt for the lease.
- `graph_hash`: content identity of the immutable static CE definition.
- `hypothesis_version`: optimistic version of mutable occurrence progress.
- `checkpoint_id`: one active planning/time/cancellation boundary.
- `configuration_hash`: provider settings identity used to permit safe reuse.
- `sharing_key`: semantic equivalence key for safely sharing demand work.
- `cancellation_scope`: smallest semantic unit invalidated when work is cancelled.
- `continuation_requirements`: artifacts that must survive a checkpoint for later
  consumers or retrospective reasoning.
- `resource_epoch`: version of the effective node/link resource view; changing it
  can trigger replanning without changing CE meaning.
