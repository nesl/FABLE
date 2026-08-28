# Evaluation changelog

## 2026-08-05 — Reintroduce B0 as CE-specific all-node placement

- Reintroduced `B0_PRODUCE_ALL` with new semantics: it selects the same
  complex-event-specific authored provider-chain set as B1, starts that set
  with the seed watch, and freezes it for the request.
- B0 deliberately ignores B1's trace-authored node/source placement and
  broadcasts each selected sensor-executable chain to every eligible replay
  node. It does not activate providers unrelated to the requested complex
  event and does not restore the historical all-registry-provider behavior.
- Restored B0 to the default RQ1 matrix and replay/planning CLIs. FABLE policy,
  ranking, fan-out, and re-planning behavior are unchanged.

## 2026-08-05 — B1 authored trace placements

- Defined B1 as an authored, frozen whole-event realization rather than a
  family-level cheapest-chain selector. Nine pilot traces now carry explicit
  chain, execution-node, and source allowlists derived from their validated
  successful FABLE runs.
- B1 activates its complete structurally executable pipeline during the seed
  watch, so fixed providers process evidence from the beginning of replay.
  Identity comparisons remain late-bound because concrete identity pairs do
  not exist at startup, but use the pre-authored chain and placement when the
  pair becomes available.
- Restricted B1 fan-out to the authored nodes and chains. This is isolated to
  `B1_STATIC_WHOLE_EVENT`; FABLE planning and coverage behavior are unchanged.

## 2026-08-05 — Retire B0 from prospective evaluation

_Historical decision, superseded later on 2026-08-05 by the CE-specific
all-node B0 definition above._

- Removed `B0_PRODUCE_ALL` from the default RQ1 matrix, live replay CLI choices,
  matching-pilot defaults, and failed-cell repair campaign. Historical B0
  manifests and results remain readable for reproducibility.
- Future RQ1 comparisons start at `B1_STATIC_WHOLE_EVENT`; B0 is not an
  experimental treatment because it is strictly dominated by B1 for the
  intended comparison.

## 2026-08-01 — calibrated bounded RQ4 scaling campaign

- Added a reproducible RQ4 scaling-profile derivation from 105 measured E0
  observations, 135 RQ2 planning decisions, 469,975 RQ3 artifact records,
  58,663 live resource samples, the consolidated RQ1 FABLE recall, and
  content-addressed source provenance.
- Corrected profile-driven scaling so `good_network` and `cloud_degraded`
  affect control latency using the checked-in NetWaggle RTT, bandwidth, and
  loss parameters. Unknown network profiles now fail closed.
- Corrected sustainable-load reporting to stratify by both policy and network
  profile rather than pooling incompatible conditions.
- Validated low, standard, and combined-stress points across all three
  policies and both profiles, then completed all 1,380 bounded OFAT runs (23
  points, 3 policies, 2 profiles, 10 seeds). All runs emitted calibrated typed
  results and common plan/resource records.
- Under good network, every policy sustained through 512 generated labels and
  failed the SLO at the 8,192-label stress point. At stress, FABLE reduced
  effective invocations to 4,096 versus 6,144 for B3 and 8,192 without
  sharing, with corresponding timely recall of 0.556, 0.384, and 0.211.
- Under cloud degradation, the measured 256 ms RTT alone exceeded the
  predeclared 250 ms control-latency SLO, so no workload satisfied the joint
  SLO in that condition.
- Added and executed a fixed-stress saturation refinement at 1,024, 2,048,
  4,096, 6,144, and 8,192 labels (3 policies, 2 profiles, 10 seeds; 300/300
  completed). Under good network, FABLE sustained 2,048 labels, while B3 and
  FABLE without sharing sustained 1,024. At 2,048 labels, FABLE retained 0.815
  timely recall with 1,024 effective invocations; B3 and no-sharing fell to
  0.771/1,536 and 0.728/2,048 respectively.

## 2026-07-30 — live replan and NetWaggle runtime profile control

- Identified that bundled NetWaggle previously selected a profile only during
  topology startup. Restarting it for W1/N0 would destroy namespaces and
  invalidate a dynamic-adaptation run.
- Added an in-process, versioned Unix-socket endpoint that reconfigures every
  existing Mininet `TCLink` interface in place. Runtime profiles must preserve
  the complete topology and cannot add or omit links.
- Added a root-runner-owned condition map for N0, W1, W2, and L1 plus a narrow
  helper client. The protocol rejects unknown fields, kinds, targets,
  conditions, actions, topology changes, oversized requests, and unvalidated
  results.
- The privileged helper uses a fixed socket when running as root; environment
  variables cannot redirect it. It still must be installed as a root-owned,
  non-writable copy outside the Codex-writable repository before privileged
  use.
- Connected distributed resource/failure callbacks to active B3, B4, and FABLE
  requests. Matching active demands now advance the resource epoch and re-enter
  controlled planning with a `RESOURCE_EPOCH` trigger; static policies do not
  replan.
- Non-privileged integration validation passed through a real Unix-socket
  request/response using fake `TCLink` interfaces, and NetWaggle's real
  topology/profile/condition map passed runner dry-run validation.
- Focused tests passed (35/35) and the complete suite passed (404/404). No
  Mininet, OVS, namespace, route, interface, or `tc` mutation was performed.

## 2026-07-30 — Milestone 3 typed dynamic-adaptation controller

- Added an allowlisted adaptation controller with `DRY_RUN`, `PROFILED`, and
  `HOST_HELPER` modes. Scheduled actions contain no command or shell field.
  Targets, disturbance kinds, and condition IDs must all match an immutable
  policy.
- `HOST_HELPER` accepts only one absolute executable, verifies that it is
  root-owned, executable, and not group/other writable, invokes it with a fixed
  argument array and `shell=False`, applies a timeout, and requires validated
  JSON output. No privileged helper was installed or executed.
- Added profiled network actions that rebuild planner-visible deployment links
  from NetWaggle profiles and emit exact `NetworkCondition` epochs.
- Added profiled compute actions that rebuild one target node's planner-visible
  capacity, execution multiplier, and queue delay and emit a profiled
  `ResourceSample`.
- Added a composite typed action router, schedule/controller/common-record
  integration, runner-side disturbance persistence, and the canonical W1+E1
  profiled manifest.
- Non-privileged validation applied W1 and E1 one second after plan dispatch,
  restored both after 20 seconds, advanced condition epochs 1–4, emitted four
  validated disturbance records, six network-condition records, and two
  capacity samples, then observed ten post-restoration seconds. No host
  mutation occurred.
- The complete repository suite passed (399/399). Results are under
  `evaluation/results/milestone3_profiled_adaptation_20260730/`.

## 2026-07-30 — Milestone 2 cross-baseline common-record validation

- Added `--quiet` to the bounded replay driver so large diagnostic result JSON
  can be written without duplicating it on stdout.
- Extended the common-baseline reporter with per-family JSONL counts, native
  versus measurement-window attribution counts, request-isolation checks,
  normalization-error counts, and a required-record completeness verdict.
- Executed the same 2026 convoy trace through B0, B1, B2, B3, B4, and FABLE
  against one persistent perception stack. All six were true positives, all
  six passed request isolation, all six emitted the required common record
  families, and no normalization errors occurred.
- Wall times were tightly grouped from 45.024 to 45.252 seconds. Each run
  emitted 2–3 demands, 2 plans, 8 typed commands, 8 leases, 14–25 lifecycle
  records, 87–88 resource samples, and 2 predicate observations.
- Artifact counts varied from 171 to 1,436, but all were explicitly classified
  as ambient measurement-window rolling-buffer artifacts. They must not be
  interpreted as request-attributed artifact cost.
- Results and the machine-readable CSV/JSON audit are under
  `evaluation/results/milestone2_common_records_all_baselines_20260730/`.

## 2026-07-30 — native lifecycle and artifact request attribution

- Added backward-compatible request attribution to distributed provider-status
  and artifact-announcement envelopes.
- Node agents derive all active lifecycle request IDs from typed lease commands.
  Terminal cancellation status preserves the request attached to the lease
  that was just removed. Request-created retrospective/reference artifacts
  carry their originating request directly; ambient rolling buffer artifacts
  remain explicitly measurement-window attributed.
- The evaluation normalizer rejects lifecycle/artifact envelopes natively
  attributed to another request. Older or ambient envelopes remain compatible
  and retain explicit measurement-window attribution.
- Focused distributed/runtime tests passed (27/27), followed by the complete
  repository suite (391/391).
- A bounded 2026 FABLE convoy remained a true positive with F1 1.0 in 45.303
  seconds. All 45 captured provider lifecycle events used native request
  attribution. The 734 artifact events were ambient rolling track-buffer
  artifacts, so measurement-window attribution was correct for this case.
- The run also captured 90 resource samples, 2 demands, 2 plans, 8 commands,
  8 leases, and 4 predicate observations. Results are under
  `evaluation/results/milestone2_live_records_20260730/`.

## 2026-07-30 — canonical heartbeat resource capture

- Fixed the live evaluation normalizer to accept the canonical
  `fable.node_heartbeat.v1` payload actually published by node agents and
  consumed by the orchestrator. It previously recognized only the unused
  `fable.reliable_node_heartbeat.v1` wrapper.
- Preserved wrapper compatibility and added heartbeat schema, sequence,
  session, and explicit measurement-window attribution metadata to every
  resource sample. Lifecycle and artifact records now also label their
  measurement-window attribution explicitly.
- Focused validation passed (22/22), followed by the complete repository suite
  (390/390).
- The bounded 2026 convoy validation remained a true positive with F1 1.0 and
  completed in 45.087 seconds. It captured 90 resource samples: sequences
  7–51 from both `dvpg_gq_orin_14` and `x86server`, all using the canonical
  heartbeat schema.
- The same run captured 2 demands, 2 plans, 8 commands, 8 leases, 41 lifecycle
  records, 811 artifact records, and 3 predicate observations, with no
  normalization errors. Results are under
  `evaluation/results/milestone2_live_records_20260730/`.

## 2026-07-30 — request-scoped live common-record instrumentation

- Added a live planning-boundary normalizer for predicate demands, plan
  decisions, typed provider activation commands, and provider lease events.
  Admission and successor-frontier replans now use the same emission path.
- The orchestrator publishes those typed records on a request-scoped MQTT
  topic. The replay-accuracy runner now starts a bounded subscriber and writes
  one JSONL file per common record family beside its result JSON (or at an
  explicit `--common-record-dir`).
- Direct common records and predicate results are rejected by the subscriber
  when their request ID does not match the active run. Provider status,
  heartbeat, and artifact messages remain measurement-window observations
  because their current distributed envelopes do not carry a request ID.
- Focused tests passed (21/21), followed by the full repository suite
  (389/389).
- Live FABLE validation on 2026 convoy case
  `20260413-pass-follow-clear-convoy-c1-test-r003` was a true positive in
  50.057 seconds. Captured JSONL totals were: 2 demands, 2 plan decisions,
  8 provider commands, 8 leases, 25 lifecycle events, 370 artifact events,
  and 2 predicate observations. No normalization errors occurred.
- No resource samples were observed in this initial smoke; the subsequent
  canonical-heartbeat fix above closes that gap. Results are under
  `evaluation/results/milestone2_live_records_20260730/`.

## 2026-07-30 — common-baseline convoy milestone

- Enabled the replay-accuracy driver for canonical B0, B1, B2, B3, B4 and
  FABLE rather than restricting it to B4/FABLE.
- Ran all six systems on
  `20260413-pass-follow-clear-convoy-c1-test-r003` with bounded max-throughput
  replay. The first pass produced four true positives; B0 and FABLE observed
  both distinct `PASSES` intervals but rejected the second because its start
  exactly equaled the first interval's end.
- Removed the unsupported 250 ms lower gap from the sequential-pass evaluation
  profile. Distinct vehicle bindings and the 60-second maximum remain in force.
  This prevents inference cadence from deciding whether adjacent tracker
  intervals pass.
- After rebuilding the orchestrator, B0 and FABLE reruns were true positives.
  The final common result is 6/6 true positives. Results and the CSV summary are
  under `evaluation/results/milestone2_common_convoy_20260730/`.
- Added a reusable common-baseline trace CSV reporter. These replay-driver
  artifacts cover event outcomes and evidence counts; complete common
  plan/resource JSONL callback emission remains a separate instrumentation
  milestone.

## 2026-07-30 — structural task universe and canonical B2 admission

- Added a structural task-demand universe used by whole-event offline
  baselines. It traverses all remaining executable branches and includes
  retrospective templates before their trigger time is known.
- Consume-only roles whose upstream alternative has not yet introduced an
  identity receive an explicit `__structural_unbound__` planning placeholder.
  These projections are restricted to offline resource planning and cannot be
  dispatched as live semantic evidence.
- Restricted B3 to a task/resource planning view that omits semantic frontier,
  binding, hypothesis and semantic-epoch information.
- Enabled canonical B2 live admission through its fixed-realization manifest.
  B0, B1 and B3 remain rejected at live admission until the common semantic
  matcher can retain and later apply observations produced before their graph
  node reaches the authoritative frontier.
- Refined B4's deterministic greedy ordering to prefer active-provider reuse
  and incremental resource cost before transfer bytes and quality.
- Added the bounded baseline-neutral early-observation buffer. It preserves
  provider evidence and provenance while replacing only the stale
  demand/frontier execution envelope after a compatible grounded demand
  becomes active. Matching is request-, graph-, node-, predicate-, source-,
  event-time- and identity-aware; structural placeholder identities are never
  eligible to match.
- Wired canonical B0, B1 and B3 through typed live admission and the structural
  whole-event graph. Their provider observations are buffered until a
  compatible authoritative frontier is active; normal FABLE/B2/B4 result
  progression remains direct.
- Made strict B1 fail closed when its manifest omits any complete-task demand,
  and completed the preferred-chain manifest for the currently compiled
  convoy, chase, robbery, rendezvous, convergence and repeated-visit profiles.
- Added typed, non-mutating semantic disturbance schedules with bounded
  apply/restore actions. Host-level Mininet, capacity and provider mutations
  remain delegated to an allowlisted controller.
- Added replay and scaling metric reducers, a bounded OFAT E8 scaling builder,
  adaptation disturbance IDs in planned runs, and per-run CSV metric output.

## 2026-07-29 — 2025 mobile archives 4–6

- Added timestamp-only Android archive discovery for files named
  `recording__<epoch_ms>.mp4`; partial `.pending-*` files remain excluded.
- The 2025 full-suite path now automatically matches mobile archives 4–6
  against each fixed-camera scenario interval. Missing mobile coverage is
  optional and does not prevent a fixed-camera evaluation.
- Corrected readiness handling so mobile adapters satisfy the camera side of
  the `zed,yolo` barrier as `mobile,yolo`, and included ready mobile nodes in
  the live request's execution scope.
- Mobile execution augmentation is opt-in for the 2025 stalking and rendezvous
  variants. Alarm robbery retains its established fixed audio/escape placement;
  allowing mobile candidates to replace that path caused a sampled regression.
- Corrected real-time mobile pacing to use MP4 presentation timestamps rather
  than unreliable Android nominal frame-rate metadata.
- Probed 560 complete recordings: every file exposed a valid H.264 720x1280
  video stream, with no ffprobe failures; representative frames decoded from
  all three archives.
- Recorded-speed validation:
  `evaluation/results/mobile456_2025_scoped_validation_20260729/`.
  The 2025 UCLA rendezvous example passed after the mobile nodes were admitted
  to the execution scope. The 2025 stalking example remained negative because
  the added footage supplied only one mobile `PASSES` observation, not two
  identity-consistent visits. A mobile-enabled alarm-robbery regression also
  remained positive on its intentionally fixed-sensor execution scope.

## 2026-07-29 — reference-diverse repeated-visit seeds

- Added an opt-in `reference_diverse` seed-admission policy for the two- and
  three-visit stalking profiles. Tracker fragments from a camera that already
  owns a live seed slot no longer crowd other views out of the four-hypothesis
  budget.
- Kept `first_distinct` as the default for every non-stalking event family.
  Convoy, chase, robbery, rendezvous and convergence request behavior is
  unchanged.
- Duplicate-fragment diagnostics are emitted once per concrete camera
  reference and then suppressed, preventing diagnostic traffic from becoming
  an orchestration load source.
- A recorded-speed 2026 three-visit stalking replay completed as
  `TRUE_POSITIVE` in 170.532 seconds after the orchestrator image was rebuilt:
  `evaluation/results/reference_diverse_stalking_rebuilt_20260729/`.
- The first verification attempt used a stale orchestrator image and timed out
  before watch registration. It emitted no predicates and is an infrastructure
  schema-mismatch artifact, not an accuracy result.

## 2026-07-29 — bounded seed fan-out and concrete reference binding

- Added opt-in bounded seed fan-out. Susceptible convoy/chase/stalking
  evaluation profiles may retain at most four distinct seed hypotheses in one
  semantic runtime; all other requests retain the previous single-seed
  behavior.
- Seed candidates are deduplicated by occurrence, source, concrete reference,
  and bound identity. Cancellation now retires both an open seed watch and its
  active execution.
- Added structured seed and semantic-transition diagnostics to replay results:
  seed occurrence/action, active hypothesis count, result ID, transition
  status/reason, hypothesis IDs, dispatch count and terminal lifecycle.
- Corrected sequential and uncalibrated repeated-pass graphs to bind the
  provider's concrete camera reference from the seed. Later observations must
  match that same view; the runtime no longer compares provider references
  against unusable literals such as `sequence_gate`.
- Catalog-driven replays now scope source playback to five seconds before the
  labeled interval through the label deadline. Recording-time reconstruction
  includes this replay offset, preventing an earlier event in the same source
  scenario from consuming the seed budget or being scored as the requested
  event.
- Final recorded-speed validation:
  `evaluation/results/multiseed_final_validation_20260729/`.
  Route convoy, two-vehicle chase, alarm robbery, visual rendezvous and
  pass-follow convoy passed. The questionable 2025 two-visit stalking example
  remained negative. The 2026 three-visit stalking example was nondeterministic:
  it passed in `multiseed_validation_20260729` but timed out in the final
  repetition, with different seed identities and fewer applied transitions.
- The established 2026 robbery-as-convoy negative control remained
  `NOT_DETECTED`, so bounded fan-out did not reduce specificity in that
  control.

## 2026-07-29 — year-scoped data-aligned definitions

- Added explicit evaluation profiles instead of weakening the default event
  definitions globally.
- `2024` route convoy and two-vehicle chase, plus the `2026`
  pass-follow-clear convoy profile, now accept two distinct vehicle crossings
  separated by at most 60 seconds. This represents sequential convoy members
  that are not simultaneously visible.
- `2025` two-visit stalking now uses two same-identity camera-view `PASSES`
  observations separated by 30 seconds to five minutes. It no longer requires
  calibrated `INSIDE` geometry and does not use `FOLLOWS`.
- `2025` robbery-with-alarm now requires a confirmed alarm followed by an
  observed vehicle departure, without claiming retrospective cross-camera
  identity linkage. The stricter cross-sensor robbery definition is unchanged.
- `2025` talking/rendezvous now has a visual-presence evaluation profile:
  vehicle arrival followed by a real person track. This does not claim that a
  conversation was detected.
- Added `PERSON_PRESENT` as a typed state predicate emitted from person tracks.
- Replay results and suite output now label experiments as
  `year | complex-event variant | experiment ID`; the full-suite runner also
  supports repeatable `--experiment-id` selection.
- Validation run:
  `evaluation/results/definition_simplification_validation_20260729/`.
  Seven recorded-speed examples produced two true positives (2025 alarm
  robbery and 2025 visual rendezvous) and five false negatives. The remaining
  sequential/repeated-visit negatives contained abundant source predicates
  but did not complete the semantic graph, so they are classified as
  `SEMANTIC_GRAPH`/seed-and-identity progression issues rather than absent
  detections.

## 2026-07-28 — restore artifact planning and bound VLM identity comparisons

- Fixed live planner feasibility validation to compare deployment artifact
  families and schema versions independently. Valid artifacts such as
  `camera_calibration/camera_calibration.v1` and
  `route_graph/route_graph.v1` are no longer rejected by comparing their
  family directly with the versioned data-type name.
- Prevented the bounded VLM ReID fallback from comparing an unchanged
  sensor-local track with itself across descriptor snapshots.
- Added non-secret VLM attempt/result logging with replay ID, entity kind,
  source pair, budget usage, acceptance, and confidence.
- Added an explicit live execution-node scope to typed evaluation requests.
  Eligible sources and provider placements are constrained at every semantic
  frontier, and the constraint is applied before the physical alternative
  enumeration cap so out-of-scope placements cannot crowd out the valid local
  plan.
- Fixed replay provenance loss between YOLO and the vehicle provider. When the
  IoBT-MAX detection transport omits the per-row `replay_id`, the vehicle
  processor now restores the active ID established by `/replay/sync`; YOLO
  independently rejects frames from any other replay before inference.
  Explicitly mismatched detection replay IDs continue to fail closed.
- Fixed live node scoping to clear and recompute each demand's derived
  `sharing_key` after its hard node constraints change. Previously the seed
  could be received correctly but admission failed Pydantic validation while
  constructing the scheduling candidate.
- Vehicle predicate samples in evaluation artifacts now include `replay_id`
  so stale/cross-run observations can be distinguished directly in reports.
- Aligned the vehicle-centric robbery graph with its revised semantics:
  retrospectively recovered vehicles now complete on a canonical-equivalent
  `EXITS` observation instead of requiring a person/vehicle
  `DEPARTURE_OR_ESCAPE` interaction.
- Strengthened repeated-visit stalking so each return requires an `ENTERS`
  transition at least 30 seconds after the preceding `EXITS`. This prevents
  short detector/tracker dropouts from counting as additional visits.
- Live semantic progression now immediately releases the satisfied demand's
  idle physical instances before planning the next frontier. Instances shared
  by another live lease are preserved. This prevents short sequential graphs
  from exhausting node capacity on unreusable idle-grace reservations.
- Changed the identity consistency pilot to real-time playback after
  max-throughput replay reordered or dropped evidence needed by the live
  event-time graph.

## 2026-07-28 — phase timing and repeatability diagnostics

- Live replay accuracy results now include structured timing for watch
  registration, replay configuration/readiness, sync-to-admission,
  sync-to-terminal-or-cutoff, cleanup, total wall time, labeled-event
  duration, the observed event-time span, and its effective wall-time ratio.
- The scenario catalog's cross-sensor timestamp envelope is explicitly marked
  as such rather than called recording duration; known ZED clock offsets can
  make that envelope much larger than the underlying media.
- Bounded-pilot rows retain the outer subprocess wall time, and pilot reports
  record total pilot wall time. Container startup is explicitly reported as
  unmeasured when the stack was started outside the runner.
- These fields make accelerated replay backlog and graph-processing latency
  distinguishable from the duration of the source recording.

This file records changes that can alter evaluation results. Evaluation result
JSON files also contain a `provenance` object with exact runner arguments and
SHA-256 fingerprints of the evaluation inputs and model artifacts. Git commits
are not required to compare runs.

For every result-affecting change, add an entry with:

- date and a short change identifier;
- affected event families and scenarios;
- semantic, predicate, model, routing, replay, or metric changes;
- expected compatibility with earlier results;
- canonical positive and negative cases tested;
- repetitions, pass rate, and observed failure layer.

Failure layers are:

1. `SENSOR_EVIDENCE`: required source evidence was absent;
2. `PREDICATE`: source evidence existed but required predicates did not;
3. `SEMANTIC_GRAPH`: predicates existed but the graph did not complete;
4. `METRIC_MATCH`: a detection existed but did not match ground truth;
5. `RUNTIME`: readiness, cleanup, timeout, transport, or infrastructure failure.

## 2026-07-29 — 2025-vehicle-rendezvous-relabel-v1

- Relabelled source rows 26–28 from `Two-visit stalking` to
  `Vehicle rendezvous` after manual review of all available mobile-4/5/6
  recordings showed CAR1 and ATV2 converging in shared views.
- Retained the legacy `Staking-BrianJulian-*` source-row names only as raw-log
  provenance; new stable experiment IDs use `vehicle-rendezvous`.
- Vehicle rendezvous compiles to the two-vehicle convergence graph
  (`PASSES` seed followed by `DISTANCE_LT`) rather than repeated-identity
  stalking or talking/person-presence semantics.
- Earlier stalking accuracy results for these three rows are invalid as
  stalking measurements and must not be included in future stalking reports.

## 2026-07-27 — provenance-v1

- Added Git-independent provenance to replay-accuracy results.
- Captures runner arguments, Python runtime, evaluation/FABLE/provider source
  and configuration hashes, Compose/runtime configuration hashes, and
  available YOLO model hashes.
- Does not capture environment dumps or secrets.
- Earlier result files have no provenance block and must be compared manually.

## 2026-07-28 — bounded-pilot-v1

- Added a manifest-driven, sequential pilot runner with internal and external
  per-run time limits, explicit repetitions, controls, and result isolation.
- Added aggregation by family, positive recall, control specificity,
  configuration/model consistency, and failure layer.
- Pilot configuration: two positive cases with two repetitions plus one
  cross-family control for robbery, stalking, convoy, and rendezvous.
- Added an explicit `model_id` runner argument because the active model may be
  packaged inside a container rather than present as a host repository file.

## 2026-07-28 — replay-determinism-and-sequential-follows-v1

- Made identity-service replay reset idempotent; duplicate supervisor sync
  rebroadcasts no longer erase ReID history for the active replay.
- Added target-node replay configuration and fresh readiness checks to prevent
  unrelated or retained camera state from entering a run.
- Multi-node runs now require fresh, scenario-matching readiness from every
  selected node; partial camera availability fails before replay sync instead
  of silently evaluating a degraded single-camera graph.
- Renamed robbery's retrospective person predicate to
  `PERSON_PRESENT_BEFORE` to match its actual camera-observation semantics.
  The historical executor now waits for a qualifying person TrackSet instead
  of failing permanently when the first retained TrackSet contains vehicles
  only. `PERSON_ENTERED_BEFORE` remains registered as a compatibility alias.
- Mapped the labeled `Cross-sensor robbery` variant to the trigger-directed
  robbery graph seeded by gunshot. It now performs retrospective person
  recovery and subsequent departure association instead of using the separate
  drive-up-shooting graph.
- Audio seed observations without acoustic localization now bind a conservative
  `sensor_scene:<source>` location. This permits retrospective scene recovery
  without claiming doorway, building, or metric localization.
- `PERSON_PRESENT_BEFORE` is now a topology-free fan-out demand: one local
  retrospective realization is dispatched to every selected camera node.
  Candidate person bindings are canonicalized through the live identity map
  when available. No 2025 placement or route prior is used for 2026 runs.
- The bound-person `DEPARTURE_OR_ESCAPE` continuation uses the same selected-node
  fan-out. The live alternative builder may enumerate up to 16 candidate nodes
  per step so dynamic 2026 deployments are not truncated to the first two
  nodes; normal bounded alternative caps still apply.
- ReSpeaker replay now carries both the active `replay_id` and the original
  playback-CSV timestamp into typed audio events. Seed watches and node-agent
  audio caches reject events from other replays, and classifier debounce state
  resets on replay changes. Accuracy results report per-node audio counts and
  playback-time person-to-nearest-gunshot deltas.
- Added sequential image-traversal `FOLLOWS` for bounded, same-path,
  same-direction vehicle passages that do not overlap in one frame.
- Added a low-weight HSV appearance component to pretrained ReID embeddings for
  vehicle reacquisition across tracker resets. Color alone is not canonical
  identity evidence.

## 2026-07-28 — bounded-vlm-reid-fallback-v1

- Added an opt-in OpenAI vision fallback for person or vehicle pairs left
  unmatched by calibrated ReID.
- The fallback compares full camera frames annotated with the corresponding
  YOLO target box. Normal ReID still uses the tight YOLO crop, and the VLM runs
  only after ordinary ReID has returned no association.
- Calls are hard-capped at four per replay across both entity kinds. The budget
  and attempted-pair cache reset only on a new replay ID.
- The default pinned model is `gpt-4o-mini-2024-07-18`; API credentials are
  supplied only through `OPENAI_API_KEY` and are not written to artifacts.
- VLM-derived associations are explicitly labeled with
  `association_basis=vlm_fallback` and the model ID, keeping them distinct from
  calibrated embedding associations.
- The fallback remains disabled unless `FABLE_VLM_REID_ENABLED=true`; missing
  credentials fail closed without affecting normal ReID.
- Offline regression coverage: 47 focused tests passed, including both failed
  person-ReID and failed vehicle-ReID VLM paths. No paid API call or
  replay was performed because `OPENAI_API_KEY` was unset.

## 2026-07-29 — replay-mqtt-and-projected-convergence-v1

- Fixed an IoBT-MAX startup race by initializing MQTT callback state before
  starting the network loop. Mobile and ZED replay services no longer
  intermittently lose their control/configuration subscriptions.
- Added an explicitly non-metric vehicle-convergence fallback for uncalibrated
  cameras. It compares same-camera vehicle bbox scale and center separation,
  requiring either a clear approach into the proximity envelope or sustained
  close support. Diagnostics distinguish this projection from metric distance.
- Replayed 2025 row 27 at real-time speed after rebuilding the affected images:
  `TRUE_POSITIVE`, with convergence evidence from Orin 1 and projected evidence
  also emitted by mobile archives 4 and 6.

## 2026-07-29 — 2025-mobile456-cross-family-sample-v1

- Enabled mobile archive 4–6 replay for `Robbery with alarm`, in addition to
  the existing rendezvous profiles.
- A five-example real-time sample produced two true positives: robbery row 23
  and talking/rendezvous row 3. Talking/rendezvous row 3 used Mobile 5 for its
  arrival seed and an Orin person observation for the interaction.
- Robbery row 8 included all three mobile archives but stalled after alarm
  admission despite person evidence. August 12 talking rows 29 and 31 remained
  Orin-only because their handset coverage is split across adjacent MP4 files;
  the current resolver accepts only one file per archive.

## 2026-07-29 — alarm-exit-and-multisegment-mobile-v1

- The legacy alarm robbery graph now accepts either an identity-associated
  `DEPARTURE_OR_ESCAPE` or an ordinary `EXITS` observation after the alarm.
  `EXITS` may introduce an otherwise unbound vehicle in this alternative.
- Mobile recording resolution now assembles an event window from multiple
  adjacent timestamped MP4 files. Generated bundles carry a typed segment
  manifest, mount every selected source file, trim overlaps, and replay the
  segments against one event-time timeline.
- Focused regression validation passed: 93 tests.
- Real-time reruns of 2025 rows 8, 29, and 31 completed without infrastructure
  failures but remained false negatives. Row 8 emitted 13 `EXITS` and 6
  `DEPARTURE_OR_ESCAPE` observations but no audio event, so its alarm seed was
  never admitted. Rows 29 and 31 admitted on `PASSES` and emitted mobile
  vehicle predicates from their multi-file archives, but emitted no
  interaction/convergence predicate.
- Results are stored in
  `evaluation/results/2025_alarm_exit_multisegment_mobile_fix_20260729`.

## 2026-07-29 — synchronized-mobile-audio-and-rendezvous-lookback-v1

- Mobile MP4 replay now decodes mono 16 kHz PCM audio with `ffmpeg` and
  publishes it through the same local ReSpeaker IPC contract as fixed sensors.
  Audio chunks and video frames share the recording timestamp, trim offset,
  replay ID, wall-clock anchor, and playback speed.
- Generated mobile nodes now include a local multimodal service connected to
  the mobile replay service. The service uses the mono audio channel and a
  five-second person-track gap tolerance suitable for the archive frame rate.
- Mobile recording selection now caps the required overlap at 60 seconds.
  This includes Mobile 6's two adjacent `161305` clips without pretending that
  uncovered padding is recorded evidence.
- Rendezvous conversation, visual proximity, and transfer outcomes retain a
  60-second activation lookback. `PASSES` is finalized when a vehicle leaves
  the view, so interaction observed while that pass is still open is replayed
  into the newly activated frontier.
- Focused regression validation passed: 39 tests.
- The corrected 2025 `161305` row 29 replay selected Mobile 4, 5, and 6, with
  three synchronized segments per archive, and completed as `TRUE_POSITIVE`.
  Mobile 6 supplied the two-person interaction evidence. Experiment processing
  took 87.321 seconds after 17.753 seconds of stack startup, and the suite
  reported that all resources were released.
- Results are stored in
  `evaluation/results/2025_mobile_audio_sync_lookback_fix_corrected_20260729`.

## 2026-07-29 — all-ce-two-valid-examples-pilot-v1

- Ran two recommended examples from each of the nine current CE variants,
  excluding failed/incomplete catalog rows, invalid 2025 rendezvous `162612`,
  and the obsolete stalking interpretation of rows 26–28.
- Sixteen of eighteen selected examples were replayable: ten true positives
  and six false negatives. Both 2024 two-vehicle-chase examples are coverage
  gaps because the replay scenario catalog has no matching recordings.
- Both examples passed for 2024 vehicle convergence, 2024 route convoy, 2025
  robbery with alarm, 2025 vehicle rendezvous, and 2026 cross-sensor robbery.
- Both examples were false negatives for 2025 talking/rendezvous, 2026
  pass-follow-clear convoy, and 2026 three-visit stalking. These were admitted
  executions rather than infrastructure failures. Rendezvous emitted person
  presence but no proximity/conversation; convoy emitted PASSES and FOLLOWS
  but did not complete the clear-scene sequence; stalking advanced but
  accumulated stale-version and minimum-delay rejections.
- Total pilot wall time was 2586.723 seconds. Every runnable experiment was
  bounded to 120 seconds in max-playback mode, and the suite reported all
  resources released.
- Results are stored in
  `evaluation/results/all_ce_two_each_valid_20260729`.

## 2026-07-29 — failing-cases-realtime-five-minute-rerun-v1

- Re-ran the six accelerated-pilot false negatives at real-time speed with a
  five-minute bound, plus two resolvable 2024 chase examples. Earlier chase
  rows fall inside long source recordings but are rejected by the current
  three-minute scenario-start matching policy.
- Four of eight runs were true positives: 2024 chase row 17, 2025 August 13
  talking/rendezvous row 3, and 2026 stalking rows 28 and 29.
- Four runs remained false negatives: 2024 chase row 13, 2025 `161305`
  talking/rendezvous row 29, and both April 13 convoy examples.
- `161305` emitted ten sustained `PERSON_PROXIMITY` observations from Mobile 6
  before its Orin `PASSES` admission but still did not advance. This isolates
  the remaining fault to outcome deployment/cache routing rather than mobile
  audio, person detection, or the proximity evaluator.
- Both convoy failures emitted abundant PASSES and FOLLOWS evidence but only
  duplicate runtime transitions, so they did not reach the clear-scene
  watermark stage. Chase row 13 showed the same evidence-routing symptom.
- Total wall time was 2271.891 seconds. The suite reported no coverage gaps,
  no infrastructure failures, and all resources released.
- Results are stored in
  `evaluation/results/failing_cases_realtime_5min_20260729`.
## 2026-07-29 — hypothesis-scoped deduplication and node-diverse planning

- Scoped deployed predicate occurrence suppression to
  `(hypothesis_id, graph_node_id, occurrence_id)` while retaining request-wide
  seed suppression and exact `result_id` idempotency.
- Balanced external-input alternatives across live sensor nodes before filling
  the bounded enumeration budget, preventing lexicographically early sensors
  from consuming every candidate slot.
- Added a regression proving that one physical observation can advance two
  distinct hypotheses without allowing duplicate delivery to the same target.
- Focused validation: 68 tests passed.
- Real-time rerun:
  `evaluation/results/hypothesis_dedup_node_diverse_fanout_realtime_20260729`.
  Both 2026 convoy cases changed from false negatives to true positives
  (90.734 s and 152.631 s). The 2024 chase graph also completed and emitted a
  detection, but evaluation scored it FN+FP because its predicted interval
  began 8.5 seconds after the labeled interval ended. The 2025 `161305`
  rendezvous remained a false negative: Mobile6 emitted ten replay-scoped
  `PERSON_PROXIMITY` observations, but admission still dispatched only one
  three-command realization and no semantic progress was returned.
- The suite released all resources.
## 2026-07-29 — source-local live fan-out for rendezvous

- Keyed live sensor fan-out by `LIVE_SOURCE` node and required the
  semantic-emitting step to be colocated with that node.
- Prevented retained alternatives from displacing live realizations for the
  same replay demand.
- Distributed each chain's bounded placement budget across external-source
  assignments and prioritized source-local final evaluator placements.
- Made orchestrator alternative limits configurable and increased the
  multi-node defaults to 64 external assignments, 256 alternatives per chain,
  and 1024 total alternatives.
- Focused validation: 69 tests passed.
- The first bounded `161305` run demonstrated that source-local selection alone
  was insufficient because the usable Mobile6 alternative was still truncated.
  After expanding the quota-balanced graph,
  `20250812-talking-rendezvous-rendezvous-brianjulian-1-r029` completed as a
  true positive in 78.277 seconds. The accepted graph bound the arrival vehicle
  and participant from Mobile6 and transitioned `FORKED` to a completed
  rendezvous hypothesis.
- Result directory:
  `evaluation/results/rendezvous_161305_expanded_source_local_fanout_realtime_20260729`.
  All resources were released.
## 2026-07-29 — nine-variant cross-family real-time smoke

- Added `evaluation/manifests/workloads/cross_family_smoke_20260729.yaml` with
  one recommended positive example for every catalog CE variant across
  2024–2026.
- Ran all nine cases at real-time playback with a five-minute per-case ceiling.
  Seven were true positives: Vehicle convergence, Route convoy, Two-vehicle
  chase, Robbery with alarm, Vehicle rendezvous, Pass-follow-clear convoy, and
  Cross-sensor robbery.
- `161305` Talking/rendezvous regressed to false negative in the multi-scenario
  run despite its immediately preceding isolated true positive. It again
  admitted only one three-command realization, emitted nine Mobile6
  `PERSON_PROXIMITY` observations, and returned no semantic progress. Its
  event-time/wall ratio also fell to 0.710, confirming unresolved
  placement/load nondeterminism rather than missing source evidence.
- Three-visit stalking row 28 advanced seven times but emitted 39 stale results
  against superseded hypothesis versions and never completed. Its cancellation
  response timed out, although the enclosing suite teardown released all
  resources.
- Aggregate result: 7 TP / 2 FN; all nine results completed, with no coverage
  gaps and suite-level `resources_released: true`.
- Result directory:
  `evaluation/results/cross_family_smoke_realtime_20260729`.

## 2026-07-29 — deterministic live fan-out and evidence-cache isolation

- Completed selected-node fan-out for source-local live predicate chains,
  including explicit local alternatives when bounded graph enumeration omits a
  selected node. `PERSON_PRESENT` now follows the same fan-out policy as
  proximity and conversation evidence.
- Only accepted runtime transitions advance a semantic epoch. Before an
  accepted result completes a demand, active sibling leases are explicitly
  cancelled so superseded providers cannot continue owning that frontier.
- Split sparse interaction observations from the high-volume vehicle evidence
  cache in each node agent. This prevents vehicle predicates from evicting
  earlier Mobile interaction evidence while a later frontier is being
  activated.
- Focused validation: 71 tests passed. Rebuilt
  `fable/orchestrator:phase6` and `fable/node-agent:phase6`.
- Ran two independent real-time repetitions of 2025 Talking/rendezvous
  `161305` and 2026 Three-visit stalking row 28, with a five-minute per-case
  ceiling. All four classifications were false negatives.
- Rendezvous was stable at 249.8 seconds in both reports. It deployed five
  source-local alternatives (15 commands) and observed 33–35
  `PERSON_PRESENT` plus 6–11 `PERSON_PROXIMITY` events, but returned no
  semantic transition. The interaction occurs before the `PASSES` seed opens
  the successor demand, so the successor event-time interval rejects the
  preserved evidence.
- Stalking was stable at roughly 305.3 seconds. It accepted 7–8 transitions,
  but produced 25–29 stale sibling results; repeat 2 also rejected four
  observations that began before the authored minimum-delay guard. No
  hypothesis assembled the terminal three-visit sequence.
- Suite teardown reported all resources released for both repetitions. The
  per-request stalking cancellation response still exceeded its five-second
  acknowledgement timeout.
- Results:
  `evaluation/results/rendezvous_stalking_cache_isolation_repeat1_20260729`
  and
  `evaluation/results/rendezvous_stalking_cache_isolation_repeat2_20260729`.

## 2026-07-30 — presence-seeded rendezvous and uncalibrated stalking

- Changed the 2025 visual rendezvous profile to seed from typed vehicle
  `INSIDE` presence instead of image-traversal `PASSES`. This matches recordings
  where vehicles are already visible before people meet and avoids treating
  handheld-camera motion as a vehicle crossing.
- Changed the three-visit stalking evaluation template to
  `evaluation_profile: uncalibrated_passes`. The graph now requires three
  temporally separated `PASSES` observations for one bound vehicle identity
  and no longer invents calibrated `INSIDE/EXITS/ENTERS` site-zone semantics
  for the dynamic 2026 deployment.
- Focused validation: 114 tests passed. Rebuilt
  `fable/orchestrator:phase6`.
- Real-time, five-minute-bounded validation:
  - 2025 Talking/rendezvous `161305`: true positive in 78.288 seconds.
  - 2026 Three-visit stalking row 28: true positive in 231.439 seconds.
- The stalking run produced four accepted transitions, five early-delay
  rejections, and no stale-version results. Both requests cancelled cleanly
  and suite teardown reported all resources released.
- Result directory:
  `evaluation/results/rendezvous_presence_stalking_uncalibrated_realtime_20260730`.

## 2026-07-30 — post-fix nine-variant cross-family smoke

- Re-ran one positive representative for every catalog CE variant at real-time
  speed with a five-minute per-case ceiling.
- All nine cases were true positives:
  - 2024 Vehicle convergence: 53.876 seconds.
  - 2024 Route convoy: 72.077 seconds.
  - 2024 Two-vehicle chase: 73.165 seconds.
  - 2025 Robbery with alarm: 59.722 seconds.
  - 2025 Vehicle rendezvous: 104.806 seconds.
  - 2025 Talking/rendezvous: 79.054 seconds.
  - 2026 Pass-follow-clear convoy: 93.334 seconds.
  - 2026 Three-visit stalking: 230.232 seconds.
  - 2026 Cross-sensor robbery: 254.197 seconds.
- The corrected 2025 presence-seeded rendezvous and 2026 uncalibrated stalking
  definitions both remained true positive outside their isolated validation.
- Aggregate wall time was 1646.150 seconds (27 minutes 26 seconds). Experiment
  subprocesses accounted for 1030.187 seconds; the remainder was bundle
  generation, nine scenario-specific stack startups/readiness probes, stack
  teardown, and final cleanup.
- Suite-level cleanup reported all resources released. The cross-sensor
  robbery's per-request cancellation acknowledgement exceeded five seconds
  after its terminal detection, but enclosing teardown succeeded.
- Result directory:
  `evaluation/results/cross_family_semantic_fixes_realtime_20260730`.
## 2026-07-30 — scalable NetWaggle topology and qdisc readback

- Added deterministic generation of the canonical 5–20-device site-local
  deployment. Each logical sensor receives a unique anchor namespace, IP
  address, access switch, and service bundle.
- Checked in `site_local_20node.yaml` and the matching NetWaggle JSON topology.
  N0, W1, W2, L1, and their condition map are generated from the same topology
  to prevent incomplete runtime profiles.
- Dynamic network updates now read `tc -j qdisc` for both interfaces of every
  configured link. Empty or failed readback makes the typed socket response
  fail instead of reporting unverified success.
- Added focused generation, uniqueness, topology-coverage, and qdisc-readback
  tests.

## 2026-07-30 — bounded path and cgroup validation

- Added read-only network validation using fixed, shell-free ping and iperf3
  client commands inside an explicitly named anchor container. Both commands
  are time-bounded, their outputs are parsed fail-closed, and measured RTT and
  throughput must meet declared bounds.
- Added confined cgroup v2 readback. Only a validated direct child of the
  configured cgroup root is accepted, and observed `cpu.max` and `memory.max`
  values must exactly match the typed expectation.
- Kept mutations out of this layer. The deployed root-owned helper must first
  apply its fixed policy, then invoke these validators and return their scalar
  measurements through the existing adaptation result protocol.
- Replaced the implicit plain-Alpine anchor dependency with a pinned
  `Dockerfile.anchor` that installs iputils, iproute2, and iperf3. Generated
  anchors start a namespace-local iperf3 server and health-check both probe
  executables, avoiding late “package not installed” failures.

## 2026-07-30 — shared attribution and derived lifecycle records

- Resource heartbeats now retain the complete active-demand count and
  fractionally attribute capacity to the current request by intersecting
  heartbeat demands with its active lease demands. Unallocated heartbeats
  remain explicitly measurement-window observations.
- Typed retrospective failures now automatically produce common artifact
  `BUFFER_EXPIRED` or `COMPATIBILITY_FAILURE` records, preserving the attempt,
  artifact, node, type, and expiration reason.
- Added a stateful coordination tracker over common records. It opens from an
  upstream predicate observation, accumulates plan selections, commands and
  provider-ready milestones, and closes on downstream evidence or explicit
  expiry. This avoids deriving coordination metrics from planner-private state.

## 2026-07-30 — common hosted-VLM contract and profiled replay

- Promoted the bounded identity fallback into the shared provider catalog as
  `hosted_vlm_identity_comparator`, with detector-boxed full-context frame
  inputs and a distinct `SAME_ENTITY` fallback chain.
- Added hard node-capability placement: the provider requires `hosted_vlm`,
  which is declared only on `cloud1`, not the site-local x86 server.
- Added explicit LIVE/PROFILED modes, a four-invocation per-run gate,
  confidence and ambiguity thresholds, and secret-name-only validation that
  never stores or returns API-key values.
- Added deterministic single-use PROFILED replay manifests. The example is a
  schema/runner fixture, not an E0 calibration result.
- Kept the legacy x86 identity service separate. A dedicated cloud proxy and
  runtime mapping are still required before the common LIVE provider can be
  selected in a full-stack run.

## 2026-07-30 — secret-isolated hosted-VLM proxy

- Added a dedicated cloud proxy process and a secret-free remote comparator.
  `OPENAI_API_KEY` is now passed only to `fable-vlm-cloud1`, not to the x86
  identity association service.
- Requests use a fixed typed schema, inline image data URLs, replay-scoped
  invocation IDs, and no credential fields. The proxy rejects unknown fields,
  filesystem/remote image references, duplicate IDs, and the fifth call in a
  replay.
- The proxy stays health-checkable when LIVE credentials are absent and fails
  comparison requests closed. This permits non-VLM evaluations to bring up the
  stack without silently enabling external calls.
- Direct planner-command/result integration on the cloud node agent remains
  separate from this now-functional legacy-association proxy path.

## 2026-07-30 — direct cloud hosted-identity results

- Added a distinct `fable-agent-cloud1` service and an adopted, non-reference
  runtime mapping for `hosted_vlm_identity_comparator`.
- Added the narrow `IDENTITY_ASSOCIATION` output adapter. It accepts only the
  typed canonical entity-map schema, a `SAME_ENTITY` demand, an overlapping
  event-time window, and associations consistent with already-bound roles.
  Unbound roles receive the canonical identity; arbitrary proxy output cannot
  satisfy another predicate.
- Added the cloud agent and proxy to the generated NetWaggle cloud bundle and
  regenerated the canonical 20-sensor topology and matched profiles.

## 2026-07-30 — bounded hosted proxy/container smoke

- Built `fable/node-agent:phase6` and `fable/vehicle-stack:phase7` within the
  two-minute build ceiling.
- Started only MQTT, MongoDB, orchestrator, hosted proxy, and cloud agent. The
  proxy became healthy and the cloud agent emitted an `AVAILABLE` heartbeat
  with `node_id=cloud1`.
- From the cloud-agent container, reached the proxy health endpoint and sent a
  typed request containing a forbidden filesystem image reference. The proxy
  returned HTTP 400 with the expected inline-image-only validation error; no
  OpenAI request was made.
- Stopped and removed all five smoke-test containers after validation. The
  newly built images and named state volumes remain available.

## 2026-07-30 — broad evaluation completion tranche

- Added inventory-driven E0 calibration planning. The checked-in provider
  catalog and deployment currently resolve to 115 feasible
  provider/tier/input targets; the default plan creates 30 warm and 10 cold
  repetitions per target without starting services.
- Added typed E0 observation reduction and deterministic measured-profile
  output for success rate, cold startup, warm execution, quality, and
  ambiguity.
- Embedded campaign year, replay-supported sensors, unavailable mobile
  sensors, and topology deployment IDs in immutable planned runs. The live
  MQTT logger can now construct its E6 coordination record deriver from one
  exact manifest run.
- Added 95% confidence intervals, trace-paired FABLE/baseline comparisons,
  predeclared E8 service-level objectives, and maximum-sustainable-load CSV
  and JSON reports.

## 2026-07-30 — executable E8 no-sharing control

- Added `FABLE_NO_SHARING` as a first-class controlled baseline and included it
  alongside B3 and FABLE at all E8 scaling points.
- Disabled warm-provider reuse and retained produced-artifact realization in
  its physical graphs, so planning costs do not receive sharing benefits.
- Added a request-scoped scheduling policy that gives every no-sharing demand
  a distinct provider compatibility token. This prevents the lifecycle layer
  from silently attaching multiple demands to one provider after planning.
- Preserved ordinary FABLE behavior and deployment artifacts; the new controls
  are selected only by the no-sharing baseline.

## 2026-07-30 — bounded profile-driven E8 execution

- Added a versioned scaling execution profile and deterministic executor for
  large PROFILE_DRIVEN points. Each run writes a typed result and common
  aggregate `PlanDecision` and `ResourceSample` records.
- Added a bounded campaign CLI with a wall-clock deadline, optional run cap,
  campaign CSV/JSON output, and per-baseline maximum-sustainable-load reports.
- Made unmeasured profiles fail closed. The checked-in example requires an
  explicit implementation-validation override and labels every result
  `calibrated=false`.
- Executed all 69 one-repetition N0 runs with the illustrative profile. All
  emitted their expected artifacts in under one second; these values validate
  implementation behavior only and are not scientific measurements.

## 2026-07-30 — corrected E0 execution boundary

- Corrected E0 input classes so all required ports of a multi-input provider
  form one calibration signature. Optional inputs are varied individually.
  This reduced the invalid 115-target plan to 79 executable-shape targets.
- Added a read-only readiness report over the exact deployment and runtime
  map. It distinguishes actual container runtimes from reference-delay
  fixtures, preventing simulated delay entries from becoming hardware
  calibration results.
- The current boundary is 29 container-ready targets, one hosted-bounded
  target, 22 reference-only targets, and 27 missing tier-specific runtime
  targets across 36 providers.
- Added the bounded E0 campaign executor. Each exact target requires a typed
  adapter; cold runs create a fresh provider, warm runs retain one instance,
  and every adapter runs in a terminable child process. Missing adapters,
  timeouts, and invocation failures are recorded as exclusions rather than
  substituted with reference-delay values.
- Added a shell-free JSON calibration-worker protocol and versioned fixture
  manifest. Workers receive only the exact target, invocation number, and
  typed fixture payload; response schema, quality, and ambiguity are validated
  before an observation can be recorded.
- Added initial provider worker operations for pairwise distance and the
  dependency-light spectral audio validation backend. Warm-only command
  adapters now explicitly exclude cold runs instead of reporting subprocess
  construction time as provider cold-start latency.
- Rebuilt the vehicle analytics image and executed 10 bounded pairwise worker
  requests in an isolated container. All succeeded. Provider-internal time was
  approximately 0.246 ms mean versus approximately 895 ms wall time dominated
  by `docker exec` and Python startup. The checked-in result is classified as
  desktop x86 container compatibility; it does not make an Orin performance
  claim.
- Clarified the desktop-only evaluation topology: sensor/server are logical
  placement roles sharing one x86 host. Added persistent JSONL worker mode and
  reran 30 warm pairwise requests successfully. Mean provider time was about
  0.130 ms, p95 0.157 ms, and steady-state request/response overhead about
  0.465 ms; the one-time worker import/start round trip was about 847 ms.
## 2026-07-30 — Expanded typed E0 provider worker coverage

- Added typed worker operations for motion state, route matching, zone
  membership, and stateful local follows using the production provider
  implementations.
- Added fixture-level correctness tests, including multi-frame state
  preservation for follows.
- Added a machine-readable capability inventory that explicitly labels the
  spectral audio backend as implementation-validation-only.
- Retained logical sensor/server placement on the single desktop x86 host;
  these labels no longer imply separate Orin hardware measurements.
- Expanded exact worker coverage to 21 production-measured target signatures
  and three explicitly validation-only signatures.
- Added bounded media-backed workers for the two deployed YOLO variants,
  package filtering, and the configured Roboflow tracker.
- Added fail-closed desktop profile promotion with complete warm/cold coverage,
  success-rate, and exact-signature gates.
- Added bounded live apply/probe/restore validation for NetWaggle paths and
  confined cgroups, with restoration guaranteed after a failed condition
  probe.
- Added concurrent-run acceptance checks for many-request provider sharing,
  fractional heartbeat attribution, and request-joined retrospective artifact
  failures.

## 2026-07-30 — pre-matrix live validation

- Corrected the generated 20-device topology for Linux interface-name limits
  and added generator-side validation of every derived interface name.
- Live-validated all 22 logical NetWaggle anchors: each had its unique expected
  IP address and could reach its namespace gateway.
- Ran a normal-speed paired FABLE convoy while applying W1 network and E1
  compute profiles. Both profiles passed live readback, restored to N0, and
  the CE produced a true positive.
- Ran two concurrent requests against one shared replay. Both produced true
  positives; validation found eight shared provider instances and four
  correctly fractionally attributed heartbeat windows.
- Completed 105 successful desktop E0 observations over all 21 eligible
  production target signatures and promoted all 21. Eight non-production or
  missing worker signatures remain explicitly excluded from measured claims.
- Added calibrated planner-profile conversion and a read-only orchestrator
  profile mount. The output contains 36 exact measured logical-placement
  profiles and 17 explicitly marked uncalibrated fallbacks for mixed graphs.
- Ran normal-speed calibrated FABLE and B3 convoy experiments. Both produced
  true positives and completed request/lease cleanup.
- Wrote a consolidated pre-matrix report and stopped before executing the full
  repeated matrix.

## 2026-07-30 — authored CE completion boundaries

- Strengthened vehicle convergence with a bounded group dwell and asynchronous
  identity-preserving exit requirements for all bound vehicles.
- Strengthened full talking rendezvous with same-participant boarding and
  bound-arrival-vehicle departure while retaining the co-presence-only proxy.
- Strengthened package exchange with executable post-transfer separation and
  bound receiver departure stages.
- Made the uncalibrated repeated-pass contract explicitly require prior track
  termination plus a configurable absence gap; calibrated visits retain the
  `ENTERS -> EXITS -> ENTERS` boundary.
- Changed drive-up shooting completion from generic movement to a bound vehicle
  exit and added the documented `require_boarding=false` variant.
- Added common scene-clear rearm metadata with `TRIAL_RESET` labeling outside
  semantic completion.
- Bounded evaluation-only physical alternative enumeration after the new
  identity-preserving stages exposed redundant synthetic placement growth.

## 2026-07-30 — updated-definition cross-family replay

- Reran the historical nine-case cross-family pilot at normal speed with FABLE.
- Corrected the talking-rendezvous pilot template, which still routed to the
  visual-presence proxy, to select the strengthened full-talking graph.
- Changed metric and image-space distance providers to emit continuous
  proximity intervals, allowing authored dwell guards to observe duration.
- Final corrected result: six true positives and three false negatives, with
  no infrastructure failures or coverage gaps.
- Three-visit stalking improved from the historical false negative to a true
  positive with explicit finalized-pass and absence boundaries.
- The remaining failures are stage-specific: incomplete bound exits for 2024
  convergence, no qualifying dwell for the active 2025 vehicle-rendezvous
  identity pair, and no validated interaction result for full talking
  rendezvous.
- Reclaimed 135 GB of unused Docker build layers after a fresh vehicle image
  exhausted the host filesystem; tagged runtime images and repository data
  were retained.
## 2026-07-30 — departure-only completion simplification

- Restored the pre-strengthening core event sequences while retaining the
  requested final vehicle-departure boundaries.
- Vehicle convergence now requires the original pass and convergence evidence,
  followed by asynchronous exits from both bound vehicles; the added
  three-second dwell stage was removed.
- Full talking rendezvous now requires arrival, conversation or visual
  proximity, and exit of the bound arrival vehicle; the added participant
  boarding/ReID stage was removed.
- Package exchange now requires both arrivals, transfer, and exit of the bound
  receiving vehicle; the extra source-vehicle separation stage was removed.
- Repeated-visit occurrence boundaries and drive-up shooting's existing
  boarding variant were retained because they predate or are independent of
  the added completion strengthening. The final drive-up vehicle exit remains.
- Focused semantic, planning, and live-request regression tests passed (58/58).

## 2026-07-30 — bounded known-false-negative identity and event-time repair

- Added conservative provider-side canonicalization for short, non-overlapping
  ByteTrack vehicle fragments. Simultaneously visible tracks are never merged;
  stitching requires compatible class, ReID, image position, scale, source,
  and tracker session.
- Prevented the identity service from associating overlapping same-camera
  snapshots, which had incorrectly collapsed distinct co-visible vehicles.
- Allowed the existing four-call bounded VLM fallback to compare annotated
  full frames across incompatible Orin/Mobile ReID feature spaces. Calibrated
  numeric descriptors remain fail-closed across incompatible models.
- Added 60 seconds of bounded event-time lookback to convergence dwell and
  bound-member departure stages. Traversal-based `PASSES` is emitted only when
  a track finishes crossing the image, while valid convergence and departure
  evidence can occur before that provider result arrives.
- The first post-lookback normal-speed 2024 convergence replay advanced from
  only `NOOP` results to one `FORKED` hypothesis and five canonical `MERGED`
  updates. It remained negative at the subsequent bound-exit stage; exit
  lookback was then added and unit-tested, but not live-rerun in this bounded
  iteration.
- Focused regression result: 79 tests passed. Targeted artifacts are under
  `evaluation/results/identity_continuity_targeted_max_20260730`,
  `evaluation/results/identity_continuity_vehicle_realtime_20260730`, and
  `evaluation/results/convergence_lookback_realtime_20260730`.

## 2026-07-31 — canonical-merge planning suppression

- Stopped evidence-only `MERGED` and `NOOP` semantic transitions from
  incrementing the semantic epoch or dispatching an unchanged successor
  frontier. The satisfied input demand is still retired normally.
- This prevents progressive interval evidence from amplifying one active
  rendezvous hypothesis into repeated, identical exit activations.
- Added a regression guard for the planning-status classification; the focused
  live execution, live request, multimodal, and CE definition suites passed
  37/37 tests.
- Normal-speed 2025 talking/rendezvous
  `20250812-talking-rendezvous-rendezvous-brianjulian-1-r029` changed from a
  false negative with 17 exit demands to a true positive with one exit demand
  in 153.614 seconds. Request cancellation completed and no containers
  remained.
- Result artifacts are under
  `evaluation/results/talking_rendezvous_merge_guard_realtime_20260731`.

## 2026-07-31 — bounded matrix sampling policy

- Broad RQ1 keeps all 83 recommended traces for its first repetition.
- Focused RQ2/RQ3, escalation, spatial, and continuation matrices now use
  deterministic CE-stratified defaults of 9, 9, 6, 10, and 10 unique traces,
  respectively. An explicit caller-selected cap is supported and the campaign
  policy keeps it at or below 30.
- When more than one repetition is requested, only one deterministically
  seeded trace per authored CE label receives repetitions 2–N. Every other
  selected trace runs once. `--repeat-all-traces` remains available only as an
  explicit opt-in.
- Added `--max-traces` to override the question-level cap and regression tests
  for trace diversity, determinism, the 30-trace bound, and representative
  repetition selection.
# 2026-08-01 — RQ2/RQ3a validity and bounded E4 execution

- Added a fail-closed RQ2 design-validity gate requiring remote and nonzero-transfer
  alternatives and at least one profile-sensitive workload. The 27-cell bounded
  planning replay passes; rendezvous changes realization across network profiles,
  while convoy and robbery are retained as negative controls.
- Added a controlled continuation-trap mechanism validation. Greedy planning drops
  the required `pair_trajectory.v1` continuation; FABLE preserves it and matches the
  bounded exhaustive oracle.
- Made live network-profile changes planner-visible before resource-epoch replanning,
  including generated `dvpg_gq_orin_N` to NetWaggle `s_orinN` mapping. A live W2
  convoy run emitted an epoch-1 plan and remained a true positive; its local
  zero-transfer realization correctly remained unchanged.
- Rebuilt the identity/VLM execution images and ran C0 and C4 live E4 controls. Both
  were true positives. Evidence capture now retains only images referenced by an
  identity prediction. Hosted C1-C3 execution remains gated on a credential visible
  to the launching process; the current offline judge preview requires one paid call.

## Follow-up live validation

- Imported the tmux-global OpenAI credential only into the bounded launching shell
  and ran C1, C2, and C3 on the same convoy trace. All three produced a CE true
  positive; C1 and C2 each emitted one hosted-VLM identity association, while C3
  resolved without a hosted identity association.
- A blinded `gpt-4.1-mini` judge evaluated the two `gpt-4o-mini` association pairs.
  Both were determined to be different identities, giving 0/1 VLM-judged binding
  precision for C1 and C2 on this smoke trace. This is a diagnostic result, not a
  full E4 estimate.
- Ran the RQ2 profile-sensitive rendezvous point live under B2, B4, and FABLE. All
  three were false negatives because the replay mapping supplied alarm/gunshot data
  and no interaction predicates, so the point is invalid as a planner comparison.
- Reran the two previous RQ3b chase failures. Both policies produced an accepted
  chase result before the label deadline, but its event interval began roughly nine
  seconds after the annotated interval, causing the strict overlap matcher to score
  one false positive plus one false negative.
- Added fail-closed E7 runtime controls for R0 no replay and R2 typed replay. The
  selected robbery was a negative control: both policies were true positives without
  a recorded retrospective attempt. R1 remains unavailable until a node-local raw
  media interval adapter exists; it is never silently implemented using typed data.
# 2026-08-01 — exogenous evolving-condition RQ3a design

- Added immutable `fable.condition_trace.v1` traces with monotonic offsets,
  initial network/compute profiles, fixed seeds, typed network/compute/failure/link
  actions, and optional bounded auto-restoration.
- Added independent CE start offsets. The live condition clock begins after stack
  readiness but before replay sync; transitions no longer wait for admission,
  semantic progress, or a FABLE-specific checkpoint.
- Added requested/apply-start/applied/restored timing, helper latency, condition
  epochs, common disturbance records, and active-demand exposure classification.
- Updated RQ3a to compare B2, B3, and FABLE from nominal initial state. Added the
  deterministic 54-cell short-run planner: one pass-follow-clear and one robbery
  trace, three relevant disturbance families per workload, offsets 0/20/45 seconds,
  and one paired run per system/cell.
- Added short WAN, selected-uplink, compute, and provider-failure traces plus the
  planned 480-second mixed trace. Network W1 transitions are executable through the
  existing NetWaggle allowlist. Compute, provider restart, target-specific uplink,
  and link-down actions remain fail-closed until their physical helpers satisfy the
  new calibration and restoration contracts.
- Live-validated the new clock with 2025 robbery A/r012, FABLE, and CE offset 20 s.
  The actual replay offset was 20.0001 s; W1 was requested at 30 s and read-back
  validated at 31.2707 s; N0 recovery was requested at 75 s and validated at
  76.4188 s. The runner remained active through the 120-second condition trace,
  emitted two common disturbance records, produced an epoch-1 alternative change,
  classified the demand as beginning under disturbance, restored N0, and cleaned
  all containers. The CE was a true positive.

## 2026-08-01 — RQ3a executable disturbance completion

- Made selected sensor-uplink profiles target-scoped and directional. NetWaggle
  now applies forward and reverse TC parameters independently, validates that a
  profile changes only its typed target, and supports allowlisted link down/up
  with interface-state readback.
- Added provider-family failure and restoration commands. A failed family blocks
  replacement instance activation, fails every matching active instance, and
  explicitly restarts the same leased managed container on recovery.
- Added a fixed YOLOv8s background-inference E1 workload. It records GPU
  utilization and memory occupancy and fails calibration unless the predefined
  70–85% utilization and 50% memory bounds are met. The orchestrator maps E1 to
  one quarter of nominal logical accelerator capacity with a 2.25x runtime and
  queue-delay profile.
- Wired all eight condition-trace actions through the live runner and added
  per-family fail-safe cleanup. Network restoration can no longer mask a live
  compute, link, or provider disturbance during teardown.
- Expanded adaptation metrics with output latency, plan churn, post-recovery
  stability, and demand exposure, and added a CSV RQ3a run/group report generator.
- Regenerated the immutable 54-cell matrix. Focused RQ3a/NetWaggle/provider tests
  pass 35/35; the wider evaluation suite reached 137 passes before encountering
  an unrelated authored talking/rendezvous expectation (`BOARDS`) that differs
  from the current CE definition.
