# E2 redesign: joint planning under bounded resource competition

## Claim boundary

E2 tests planning quality, not perception accuracy or fault recovery. Raw video
remains at its originating sensing tier, and no redundant camera view is
assumed. Network degradation is applied only to compact derived evidence.
Sensor disconnection remains an availability/fault case for E4 rather than a
placement crossover for E2.

The primary claim is:

> Under simultaneous feasible demands that compete for bounded compute and
> network resources, checkpoint-joint FABLE planning avoids infeasible or
> deadline-poor independent choices and remains close to an exhaustive oracle.

## Why the previous matrix was degenerate

Every evaluated raw-derived chain was cheapest when its detector remained at
the source. Resource pools were not independent, remote reads were undercharged,
and bounded alternative enumeration could omit a cheaper tier. After correcting
those defects, B4, FABLE, and O1 select equal-cost realizations for all audited
corpus frontiers. Stronger network degradation alone cannot change that result.

## Experiment structure

### E2-A: intra-checkpoint mechanism validation

Use one checkpoint containing two to four simultaneously grounded demands.
Each demand retains its real source and authored predicate. Construct the
checkpoint from real exported frontier records; do not invent an alternate
camera capable of satisfying the same predicate.

Candidate cases, in priority order:

1. Robbery-with-alarm joint departure and exit demands.
2. Cross-sensor robbery with identity/continuation work active alongside a
   local vehicle predicate.
3. A same-entity checkpoint paired with another compact-evidence consumer.

The nominal pools must be the independently profiled sensing device, Jetson
site tier, and host PC tier. Add reservations representing already-running
work, rather than changing provider costs arbitrarily:

- `R0`: no background reservation.
- `RJ50`: 50% of Jetson GPU memory/compute reserved.
- `RJCPU75`: 75% of Jetson CPU reserved, with GPU and memory left available.
  This models concurrent CPU-side tracking/projection work and is the measured
  crossover condition: raw inference remains on the Jetson while compact
  terminal work can move to the host.
- `RJ80`: 80% of Jetson GPU memory/compute reserved.
- `RS50`: 50% of sensing-tier compute reserved.
- `RPC50`: 50% of PC CPU/memory reserved.

Apply two network states only after a non-local compact-evidence realization
exists:

- `N0`: nominal physical link profile.
- `NC`: constrained bandwidth for compact evidence; raw transfer remains
  forbidden.

Compare B4, FABLE, and O1. Retain B2 as a fixed-realization control only when
the exported state contains an admission-time realization.

FABLE is considered close to O1 when both predicted completion and transfer
are within 1% of O1. Exact equality is retained as a diagnostic but is not a
headline gate.

### E2-B: concurrent-hypothesis admission

Replay real checkpoint states at the control plane without executing video.
Submit batches of `H = 1, 2, 4, 8, 16` active hypotheses with deterministic
arrival times. Different hypotheses may refer to the same physical source and
may share an already-active provider, but they must not claim redundant sensor
coverage.

This requires a bounded batch-admission layer. The current per-request policy
API does not jointly account for reservations made by other requests. The new
layer must either:

1. jointly search the batch while sharing one resource footprint, or
2. admit requests in a declared order, committing each selected plan's
   reservations before planning the next request.

Both modes should be recorded. Joint batch search is the primary FABLE
condition; sequential independent admission is the concurrency baseline.

## Required workload properties

A case is headline-eligible only if all of the following hold before repeated
runs:

- At least two demands compete for the same bounded resource pool.
- Each demand has at least two feasible physical realizations.
- At least one globally feasible combination differs from the independently
  cheapest combination.
- B4's independent selection is infeasible, misses a deadline, or has a
  strictly worse cost vector than O1 in at least one preregistered condition.
- O1 completes within its enumeration cap.
- The crossover survives corrected transfer accounting and full three-tier
  alternative enumeration.

If no real exported frontier satisfies these gates, report E2-A as a negative
result. A synthetic fixture may remain a correctness test, but must not become
headline empirical evidence.

## Metrics

Primary:

- feasible joint-plan rate;
- checkpoint deadline-miss rate;
- completion time or makespan;
- FABLE-to-O1 completion and transfer gap;
- excess resource use and transfer bytes relative to O1.

Secondary:

- planning latency and peak memory;
- labels generated, dominated, and beam-pruned;
- provider reuse count;
- resource-pool utilization and reservation rejection count;
- queueing delay under concurrent admission.

Report placement changes only as mechanism evidence, not as the outcome by
itself.

## Minimal matrix

Run a discrimination pilot before the full sweep:

| Dimension | Pilot values |
|---|---|
| Cases | first two headline-eligible exported states |
| Policies | B4, FABLE, O1 |
| Reservations | R0, RJ50, RJ80 |
| Network | N0, NC where applicable |
| Hypotheses | 1, 2, 4 |
| Repetitions | 1 deterministic audit repetition |

Only expand to 10 repetitions and the larger concurrency sweep after the pilot
passes the structural gates and produces a nonzero policy or oracle gap.

## Implementation gates

1. Export immutable frontier states including active reservations and provider
   reuse state.
2. Add a resource-reservation snapshot to deployment/planning input without
   mutating calibrated capacities.
3. Implement batch or sequential committed admission across hypotheses.
4. Add a structural discriminator that rejects a degenerate matrix before
   provider replay.
5. Verify B4's independent choices against joint feasibility instead of merely
   summing their reported costs.
6. Verify FABLE and O1 use identical hard-feasibility rules.
7. Run the planning-only pilot.
8. Proceed to physical execution only if the pilot passes.

Compute disturbance during a live recording remains E4. E2 uses measured
capacity/reservation snapshots to isolate planner quality from disturbance
controller behavior.
