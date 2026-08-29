# FABLE CE instances and frontiers

This folder is the candidate-management layer of the rebuilt FABLE. It turns
provider-produced `PredicateMatch` records into a set of plausible complex-event
occurrences and exposes the semantic evidence that could matter next.

The core model is:

```text
                         compiled CE
                             │
                   persistent discovery
                        frontier
                             │
        PredicateMatch ──────┼──────────────┐
             │               │              │
             ▼               ▼              ▼
      create CEInstance   advance H1     advance H2 ...
             │               │              │
             └───────────────┴──────────────┘
                             │
                    continuation frontiers
                             │
                             ▼
                     physical planner
```

The semantic runtime does not choose YOLO, a camera, a compute node, a
container, or a network route.

## Files

### `ce_instance.py`

Defines `CEInstance`, one plausible occurrence (hypothesis) of one authored CE.
It stores only:

- `matched_at`: the event time of the observation that seeded the candidate,
- `matched_source`: the source/sensor of that seed observation,
- `matched_predicate` / `matched_path`: what part of the CE seeded it,
- logical role bindings (`VEHICLE -> object_17`),
- satisfied AST paths and their event times,
- structure-operator activation and semantic expiry times,
- small `for:` duration bookkeeping.

`instance_id` is only a local dictionary/logging key. It is not the semantic
identity of the candidate.

Two candidates can therefore bind the same vehicle but still represent separate
occurrences:

```text
H1: VEHICLE=V7, matched_at=12:00:01, source=camera_1
H2: VEHICLE=V7, matched_at=12:30:07, source=camera_1
```

### `frontier.py`

Implements the runtime meaning of:

- `seq`
- `all`
- `any`
- `k_of_n`
- `within`
- `for`

and derives two kinds of semantic frontier.

`derive_discovery_frontier(event)` returns the predicates that can start a new
candidate. This frontier remains logically active for as long as the CE query
is active.

`derive_continuation_frontier(event, instance, now)` returns the predicates that
could advance one particular candidate.

Both return the same `FrontierItem` representation. A frontier item is already
resolved against the candidate bindings, e.g.:

```text
predicate: follows
arguments:
    leader: vehicle_17
    follower: None
classes:
    leader: vehicle
    follower: vehicle
parameters:
    max_gap_m: 30
expires_at: None
required_duration_ms: 3000
```

There is deliberately no separate `EvidenceDemand` object. The frontier itself
is the semantic-to-physical interface.

`expires_at` is a **semantic usefulness window**, not a scheduler/QoS deadline.
It currently comes only from the CE structure:

- `all`: first satisfied child + join window (5 minutes by default),
- `k_of_n`: first satisfied child + join window,
- `within`: the authored maximum,
- `seq` / `any`: no implicit expiry.

### `instance_manager.py`

`CEInstanceManager` owns the active candidate set.

For each `PredicateMatch`, it independently asks:

1. Can this observation advance any existing candidate?
2. Can this observation start a new candidate from the discovery frontier?

A single observation may do both and may advance several candidates.

When a match introduces a **new role binding**, the manager keeps the earlier
partial candidate alive as well. This allows later objects to form alternative
valid CEs from the same prefix. For example:

```text
t=1  V1 enters      -> H1 waits for PERSON until t=11
t=5  V2 enters      -> H2 waits for PERSON until t=15
t=12 P1 enters      -> H1 expires; H2 can branch to (V2,P1)
```

If P2 also enters before t=15, `(V2,P2)` can coexist with `(V2,P1)` rather than
being discarded merely because the first person match already advanced one
branch.

When a match only validates roles that were already bound, the advanced branch
replaces the old prefix. For example, once `exits(V2)` completes a candidate,
the old candidate is not left waiting forever for another exit of the same
already-bound vehicle.

This remains correctness-oriented around identity choices and can produce many
semantic instances. A later physical planner should coalesce identical provider
work across those instances rather than merging the semantic hypotheses themselves.

Object ReID is handled outside this folder. Predicate implementations first emit
`PredicateMatch` records with their local IDs; `fable/execution/IdentityResolver`
canonicalizes those IDs before they reach the instance manager. ReID does not by
itself imply that two CE instances are the same occurrence.

## Predicate result classes

`PredicateMatch` now optionally carries `classes`, e.g.:

```python
PredicateMatch(
    predicate="enters",
    arguments={"object": "camera_1:track_9"},
    classes={"object": "car"},
    ...,
)
```

This prevents a generic predicate such as `enters` from accidentally binding a
vehicle observation to a CE role whose authored class is `person`. The provider
capability catalog handles semantic aliases such as `car -> vehicle`.
