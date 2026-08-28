# Prepared experiments — do not execute yet

The repository is intentionally frozen with `execution_authorized: false` in
`config/experiment_freeze.yaml`. Setup validation and provider profiling have
been performed; no comparative/headline campaign was launched.

## Physical foundation now available

- Pi and Jetson have isolated FABLE agent environments and publish normal
  `AVAILABLE` heartbeats.
- The node agent supports `MANAGED_PROCESS`, allowing an allowlisted host
  provider to be started and stopped by normal provider leases.
- Jetson heartbeats use `tegrastats` and expose a bounded 16 GiB unified-memory
  reservation pool plus live GPU utilization.
- Pi retention is a single managed video slot with a 2 GiB free-space floor.
- Jetson YOLO has an independent three-sample calibration profile and a separate cold-start probe.
- Compute and network condition tools are dry-run unless `--execute` is given.
- The causal Orin-7 run refuses to start unless
  `FABLE_CONFIRM_EXPERIMENT=YES` is set.

## Closed execution gates

1. Physical network and compute transitions are routed by target through
   `physical_condition_control.py`, and its JSON validation is recorded as
   disturbance evidence.
2. The Pi has the root-owned `/usr/local/sbin/fable-physical-net` helper. It
   accepts one fixed degraded profile and installs a 180-second restore timer.
3. SAME_ENTITY source anchoring now accepts only real deployment-node prefixes;
   synthetic/offline bindings no longer erase all identity alternatives.
4. Independent physical calibration profiles are frozen as planner inputs and
   explicitly excluded from future evaluation observations.
5. Matched RQ3b/RQ3c policy manifests are checksum-frozen in
   `evaluation/manifests/spatial/rq3_matched_campaign_v1.yaml`.

The causal Orin-7 validation cell was replayed on 2026-08-20. Its physical
source check passed: the admitted hypothesis bound to a `dvpg_gq_orin_7` track
produced by Pi replay and Jetson inference. The event-level result was a false
negative because both FOLLOWS observations bound follower and leader to that
same track and were rejected by the distinct-entity constraint. See
`evaluation/results/physical_causal_orin7_validation_20260820/PHYSICAL_SOURCE_VALIDATION.json`.
An alternate, directly comparable three-vehicle recording
(`20241008-route-convoy-2-r013`) then completed as a true positive in 31.427
seconds with distinct physical track IDs `:0` and `:1`. This shows that the
first result was a recording-specific observability failure rather than a
physical execution or semantic binding defect. The guarded validation script
now defaults to the passing r013 recording.
The repository-wide switch remains `execution_authorized: false`.

## Prepared non-physical campaigns

- E1 resumes from `final/rq1_full_20260806_pause/remaining_runs.jsonl`.
- E2 now supports explicit O1 rows, a configurable beam width, planner CPU,
  and planner peak-memory capture.
- E2 beam-width invocations should be made separately for 1, 2, 4, 8, 16,
  and 32 so every output directory is immutable.
- E3's matched spatial/retrospective matrices are frozen by path and checksum.
