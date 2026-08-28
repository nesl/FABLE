# Evaluation implementation status

Updated: 2026-08-01

This file tracks implementation against `implementation_plan.txt`. It is not a
claim that the full experimental matrix has been executed.

## Phase 0 — correctness prerequisites

Implemented:

- canonical event definitions and evaluation aliases;
- occurrence-level ground truth and workload catalog;
- versioned planning cases and epoch-aware decisions;
- structural complete-task demand traversal, including alternatives and
  retrospective templates;
- isolated planning inputs for B3;
- branch, role, plan-isolation and structural-universe tests.

Implemented infrastructure:

- deterministic canonical deployment generation for 5–20 sensing devices;
- one distinct anchor namespace, IP address, switch attachment, and service
  bundle per logical device;
- a checked-in 20-device deployment manifest and matching NetWaggle topology;
- N0, W1, W2, and L1 profiles generated from the same link graph, preventing
  a profile for a smaller topology from being applied to a scaling topology.

Live infrastructure validation completed:

- launched the generated 20-device topology and validated all 22 logical
  anchors (20 sensors, site-local, and cloud): every container was running,
  had its unique expected IP address, and reached its namespace gateway;
- shortened the generated site-local switch name to remain within Linux's
  15-character interface-name limit, and added exact generated-interface
  validation so an invalid topology fails before Mininet is started.

## Phase 1 — baseline policy layer

Implemented:

- canonical B0–B4 and FABLE planning policies;
- offline exhaustive checkpoint oracle;
- static pipeline registry with complete-demand validation;
- live typed admission for B0–B4 and FABLE;
- bounded early-observation matching for whole-event policies;
- spatial and escalation policy families.

Remaining live milestone:

- none for baseline admission/common-record wiring; broader repeated workloads
  remain part of the experiment matrix rather than implementation.

## Phase 2 — provider catalog and tier constraints

Implemented:

- typed provider/data/chain catalogs;
- node-class, runtime, artifact compatibility, movement and live-runtime
  validation;
- fail-closed audited implementation-symbol checks;
- deployment artifact validation.
- a typed hosted-VLM provider and boxed full-context input contract in the
  common catalog;
- hard `hosted_vlm` node capability filtering, with only the cloud deployment
  node eligible;
- explicit LIVE and PROFILED evaluation modes, four-call per-run enforcement,
  secret-name-only LIVE validation, confidence/ambiguity thresholds, and
  deterministic single-use PROFILED replay manifests;
- a `SAME_ENTITY` hosted fallback chain that is structurally distinct from
  calibrated local/cross-sensor ReID.
- a dedicated secret-owning HTTP proxy process for LIVE comparisons; the x86
  identity service uses a secret-free client, while the proxy independently
  rejects unknown fields, non-inline images, duplicate invocation IDs, and
  more than four calls per replay.
- a dedicated `cloud1` node agent, non-reference hosted-provider runtime
  mapping, and typed `IDENTITY_ASSOCIATION` adapter that emits `SAME_ENTITY`
  results only for matching event windows and bindings.
- bounded live container validation of the proxy and cloud agent: both images
  built, the proxy reached Docker `healthy`, the cloud agent emitted a typed
  `AVAILABLE` heartbeat, and an inter-container typed request rejected a
  forbidden filesystem image reference with the expected HTTP 400 response.
- E0 target enumeration directly from the provider catalog and deployment
  graph, covering every structurally feasible provider/node-tier/input tuple;
- immutable warm/cold E0 run manifests and a reducer that writes measured
  success, startup, execution, quality, and ambiguity profiles;
- required multi-input provider bundles represented as one calibration input
  signature, with optional inputs varied one-at-a-time rather than incorrectly
  timing isolated required ports;
- exact E0 readiness classification against deployment and runtime inventory,
  distinguishing real container runtimes, bounded hosted runtimes,
  reference-delay fixtures, and missing tier mappings;
- process-isolated E0 adapter execution with exact provider/tier/input
  registration, persistent warm instances, fresh cold instances, hard
  per-invocation termination deadlines, typed observations, and explicit
  no-adapter/timeout/failure exclusions;
- a versioned JSON worker request/response boundary and fixture manifest that
  invokes only argument arrays with `shell=False`, rejects shell entrypoints
  and duplicate target fixtures, independently bounds worker subprocesses,
  and refuses cold-start observations from warm-only service adapters;
- provider-level worker operations covering 21 exact production target
  signatures across geometry, follows, retained matching, person proximity,
  person/vehicle relations, conversation, localization, audio-visual
  association, diarization, YOLO detection, package detection, and
  multi-object tracking;
- three exact validation-only target signatures for spectral audio, energy
  VAD, and spectral speaker embeddings, which cannot be promoted as measured
  production providers;
- a machine-readable worker capability inventory that prevents the non-final
  backends or unsupported optional-input signatures from being counted as
  measured providers;
- fail-closed desktop-x86 profile promotion: all required targets need the
  configured warm/cold sample counts, success threshold, and exact production
  worker signature before illustrative profiles can be replaced;
- bounded container compatibility validation of the pairwise worker: the
  rebuilt vehicle image completed 10/10 typed requests, while separately
  reporting provider-internal execution and Docker/Python transport overhead.
- persistent JSONL worker mode for warm calibration, validated with 30/30
  pairwise requests in the desktop x86 container; logical `sensor` and
  `server` classes are placement roles on the same physical desktop, not Orin
  versus server hardware tiers.

Remaining:

- live-validate the cloud agent/proxy placement in the NetWaggle namespace and
  a planner-selected `SAME_ENTITY` command against replay evidence;
- execute the eight explicitly unmeasured desktop targets after production
  implementations exist: three validation-only audio fixtures and five missing
  worker contracts (camera projection, interaction analysis, keyword/ASR, and
  two object-transfer signatures);
- add typed replay fixture adapters for the 29 real container-ready targets;
  the generic bounded invocation/result timing capture and worker protocol are
  complete. Twenty-one real target signatures and three validation-only
  signatures are implemented. Five container-ready signatures remain: camera
  projection, interaction analysis, keyword/ASR, and two object-transfer
  signatures. These are catalog/runtime declarations without a matching
  production implementation contract (or, for custody, the catalog inputs do
  not match the existing reasoner API), so they require production
  implementations rather than synthetic timing shims.
  The shared vehicle/multimodal images still need those operations,
  checked-in fixtures, and execution on the actual sensor hardware. The first
  desktop x86 validation is eligible for the logical sensor-placement profile
  in this evaluation configuration. The
  current corrected inventory also has one
  hosted-bounded target, 22 reference-only targets that must not be treated as
  measurements, and 27 missing tier-specific runtime targets.

Desktop E0 calibration completed:

- ran 105 successful observations across 21 exact production target signatures
  (three warm and two cold observations per target);
- promoted all 21 eligible signatures and rejected validation-only/missing
  targets from measured promotion;
- converted the observations into 36 exact logical sensor/server placement
  profiles plus 17 explicitly labelled `UNCALIBRATED_FALLBACK` profiles needed
  to keep mixed-provider live graphs executable on the single desktop host;
- wired the generated calibrated profile into the live orchestrator using a
  read-only Compose mount and `FABLE_PROVIDER_PROFILES`.

## Phase 3 — controller and disturbance schedules

Implemented:

- immutable run manifests and NetWaggle profile binding;
- planner-visible network condition epochs;
- bounded, typed semantic-triggered disturbance schedules;
- disturbance identifiers in adaptation manifests;
- closed allowlists for disturbance target, kind, and condition;
- non-mutating dry-run validation and planner-visible profiled execution;
- dynamic W1/N0 network-profile apply/restore with common condition records;
- dynamic E1/N0 compute-capacity apply/restore with common resource records;
- a hardened host-helper boundary using one root-owned executable, fixed
  argument arrays, `shell=False`, bounded execution, and validated JSON output;
- in-place NetWaggle `TCLink` profile changes over a typed Unix socket without
  topology or namespace restart;
- fail-closed `tc -j qdisc` readback on every changed interface before the
  NetWaggle control socket reports an update as validated;
- bounded ping and iperf3 validation with fixed argument arrays, explicit RTT
  and throughput bounds, and fail-closed output parsing;
- a pinned Alpine NetWaggle anchor image that installs iputils, iproute2, and
  iperf3, exposes a namespace-local iperf3 server, and health-checks probe
  dependencies before evaluation;
- confined cgroup v1/v2 state readback for exact CPU quota/period and
  memory-limit validation;
- resource/failure callbacks that advance active B3/B4/FABLE request epochs
  and re-enter controlled planning while leaving static policies unchanged.

Live host validation completed:

- installed audited root-owned, fixed-policy helpers for NetWaggle profile
  changes and x86 evaluation-container capacity changes;
- applied W1 to the running NetWaggle topology, read back all six affected
  qdiscs, passed bounded ping/iperf probes, and restored N0;
- applied E1 to the allowlisted x86 evaluation workload, read back a six-core
  CPU quota and 48 GiB memory ceiling from the host's cgroup-v1 controllers,
  and restored the original Docker cgroups and unlimited N0 limits;
- wrote the bounded network and capacity validation reports under
  `evaluation/results/validation_pass_20260730/`.

Completed privileged integration:

- recorded successful 22/22 namespace/IP/gateway validation on the live
  20-device topology;
- ran a paired live convoy CE under W1 and E1, emitted the live network/resource
  measurements, restored E1 then W1 to N0, and obtained a true positive.

Remaining privileged integration:

- extend the fixed helper policy only when provider-failure/link-down
  validation is scheduled; those mutation paths remain disabled today.

## Phase 4 — common logging

Implemented:

- versioned schemas and JSONL storage for required record families;
- planning/demand/network emission;
- replay timing, provenance, lifecycle, artifact, coordination and
  retrospective record models;
- request-scoped live admission and successor-frontier demand/plan emission;
- typed live provider-command and lease emission;
- bounded replay-run capture of predicate, lifecycle, artifact, demand, plan,
  command, lease, and resource JSONL records;
- canonical node-heartbeat normalization, live-validated on two nodes with
  sequence/session provenance;
- native many-request attribution on provider lifecycle envelopes, including
  the request whose lease caused a terminal cancellation status;
- native request attribution on request-created reference/retrospective
  artifacts, with ambient rolling artifacts explicitly distinguished.
- fractional shared-capacity attribution based on the request's live lease
  demands intersected with each heartbeat's complete active-demand set;
- automatic artifact `BUFFER_EXPIRED` and `COMPATIBILITY_FAILURE` records
  derived from typed retrospective outcomes;
- a common-record coordination tracker that joins upstream observations,
  selected/activated sensors, provider commands, provider readiness, and
  downstream evidence, including explicit no-evidence expiration episodes.
- campaign year, replay-supported sensors, unavailable mobile sensors, and
  topology deployment IDs embedded in immutable E6 run manifests;
- live MQTT logger construction of the E6 coordination tracker from exactly
  one matching immutable planned run.

Remaining:

- live-validate native artifact attribution on a retrospective/reference case
  (the convoy validation emitted only ambient rolling artifacts);
- live-validate request-joined retrospective failure artifacts on a bounded
  retrospective/reference case.

Concurrent live validation completed:

- two simultaneous requests sharing one replay both produced true positives;
- the acceptance validator found eight genuinely shared provider instances and
  four heartbeat windows whose per-request fractional attribution summed to
  one;
- the selected convoy has no retrospective branch, so this run intentionally
  produced zero retrospective attempts and zero derived failure artifacts.

## Phase 5 — metrics and reports

Implemented:

- event matching, resources, planning, adaptation, spatial coordination,
  escalation, continuation, replay and scaling reducers;
- deterministic JSON summaries plus summary, exclusion and per-run metric CSVs;
- 95% confidence intervals and trace-paired FABLE/baseline comparisons;
- predeclared E8 recall/latency SLOs and maximum-sustainable-load reduction;
- standalone scaling sample CSV and sustainable-load JSON output.

Remaining:

- execute the planned repetitions so statistical outputs contain measured
  rather than fixture/profile-driven inputs.

## Phase 6 — experiment builders

Implemented:

- executable immutable E0–E8 builders;
- bounded one-factor-at-a-time E8 sweeps and combined stress point;
- CLI generation of scaling and catalog-backed run manifests;
- inventory-driven E0 planner and completed-observation profile summarizer;
- immutable E8 relative-recall and p95-control-latency targets;
- an executable `FABLE_NO_SHARING` variant that disables active-provider
  reuse in physical planning, produced-artifact reuse in input realization,
  and compatible provider-token attachment in scheduling;
- B3, FABLE, and FABLE_NO_SHARING at every E8 scaling point;
- a bounded deterministic PROFILE_DRIVEN E8 executor that consumes immutable
  runs and a versioned calibration profile, emits common plan/resource
  records plus typed per-run results, and generates per-baseline
  sustainable-load reports;
- fail-closed rejection of unmeasured profiles unless an explicit
  implementation-validation override is supplied.

RQ4 execution completed:

- derived a content-addressed calibrated desktop scaling profile from measured
  E0, RQ1, RQ2, and RQ3 artifacts;
- validated representative low, standard, and combined-stress points;
- completed the bounded 1,380-run E8/RQ4 matrix and generated CSV plus
  per-policy/per-network sustainable-load reports.

Remaining:

- execute or repair incomplete cells in other planned matrices where their
  coverage gaps remain analytically relevant.

Calibrated pre-matrix execution completed:

- rebuilt and recreated the orchestrator with the generated desktop calibration
  profile;
- ran the same normal-speed live convoy with FABLE and B3; both were admitted,
  selected the live vehicle-pass alternative, produced true positives, and
  completed request cancellation/lease cleanup;
- stopped immediately before the repeated full matrix, as requested.

## Phase 7 — milestones

- Milestone 1: implemented and unit-tested.
- Milestone 2: complete. One common 2026 convoy trace was a true positive for
  B0–B4 and FABLE; every run passed request isolation, emitted all required
  common record families, and had zero normalization errors.
- Milestone 3: complete. Dynamic W1+E1 apply/restore is implemented and validated on the
  live host using audited fixed-policy helpers. W1 passed six-interface qdisc
  readback plus bounded ping/iperf and restored N0. E1 read back six CPU cores
  and 48 GiB on cgroup v1 and restored the affected Docker PID to N0. The
  paired live CE joined these measurements to common records and was a true
  positive.
- Milestone 5: the common hosted contract, bounded LIVE gate, dedicated
  secret-owning proxy, deterministic PROFILED replay, and direct cloud
  node-agent command/result wiring are implemented. Desktop E0 calibration,
  NetWaggle placement validation, and paired calibrated FABLE/B3 runs are
  complete. Live `SAME_ENTITY` replay evidence remains.
- Milestone 4: shared-provider and fractional-capacity acceptance is live
  validated. A retrospective failure-artifact case remains.
- Milestone 6: immutable matrix generation, PROFILE_DRIVEN E8 execution,
  no-sharing isolation, common aggregate records, and sustainable-load reports
  are implemented. A 69-run unmeasured implementation-validation campaign
  completed successfully; it is not a paper result. Calibrated repetitions and
  the remaining experiment families still require execution.
