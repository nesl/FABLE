## Recommended setup: three terminals

Two terminals are required; a third is useful for monitoring.

* **Terminal 1:** Docker replay/FABLE/evaluation stack.
* **Terminal 2:** Submit the request or predicate, then start replay.
* **Terminal 3:** Follow logs and MQTT traffic.

The evaluation plan calls for a common runner across B0–B4, FABLE, and the
oracle. `evaluation.orchestration.ControlledPlanningCoordinator` now owns the
policy cadence at the planning/orchestration boundary: B0--B2 decide at
admission, B3 may decide again only for a changed resource epoch, and B4/FABLE
may decide again for changed resource or semantic-frontier epochs. The
distributed executor remains policy-neutral because its wire request already
contains complete `PlanCandidate` objects.

`evaluation.live_orchestration.LivePlanningBridge` converts a controlled
decision into the normal `PlanCandidate` representation and submits it through
the existing scheduler/dispatcher. B0 is handled explicitly as a provider-union
baseline: each feasible realization is submitted independently so the
scheduler's normal primary/fallback de-duplication cannot collapse the union.
The other live policies produce one jointly admitted candidate.

There is still no command-line driver that compiles planning cases and executes
an entire B0--B4 run matrix. Setting `FABLE_EVAL_BASELINE` labels the output; by
itself it does not construct a coordinator and live bridge.

NetWaggle/Mininet can carry full-stack evaluation traffic and collect actual
network observations. Its TC profile is not currently translated back into the
planner's `DeploymentGraph`, however. For planning-sensitive network
experiments, the deployment link latency/bandwidth values must match the
selected NetWaggle profile, and a changed profile must advance the submitted
resource epoch. Otherwise B3 will not replan and all policies will optimize
against stale configured link costs even though packets are still shaped by
Mininet.

### Selecting network conditions

Network profiles are a first-class run-matrix dimension. Select one or more
profiles when generating a matrix:

```bash
venv/bin/python scripts/plan_evaluation_runs.py \
  RQ3_OPERATING_ADAPTATION \
  --network-profile good_network \
  --network-profile cloud_degraded \
  --network-profile lossy_edge \
  --output evaluation/manifests/workloads/rq3_network_matrix.jsonl
```

Each row contains `network_profile_id` and `network_profile_path`. Start
NetWaggle with that exact path:

```bash
./evaluation/netwaggle/start_stack.sh \
  --profile netwaggle/configs/profiles/cloud_degraded.json
```

Before compiling the physical alternative graph, load and apply the same
profile with `evaluation.networking.NetworkExperimentState.activate()`. Use its
returned deployment for graph construction, bind the recompiled graphs and
resource epoch with `bind_network_to_planning_case()`, and persist
`network_condition_records()` through
`EvaluationRunner.record_network_conditions()`. Re-activating the same profile
keeps the epoch stable; changing profiles increments it. The driver must then
invoke the coordinator with `PlanningTrigger.RESOURCE_EPOCH`, which lets B3 and
FABLE replan while B0--B2 remain frozen by their defined policy cadence.

The controlled profile registry and FABLE-node-to-NetWaggle-switch mapping are
in `evaluation/manifests/network_profiles/controlled_profiles.yaml`.

### Family-specific planning cases

`evaluation.planning_cases.compile_evaluation_planning_case()` maps every
recommended complex-event variant in the filtered experiment catalog to an
authored semantic family, advances a deterministic true-path semantic trace,
compiles the initial grounded frontier, and separately compiles the complete
remaining task path. It rejects a case when any whole-event demand has no
physical alternative; it never substitutes the convoy/FOLLOWS fixture.

The current variant coverage is:

- route and pass-follow-clear convoy: `FOLLOWS`, followed by `MOVING`;
- cross-sensor robbery: `MOVING` plus retrospective
  `VEHICLE_PRESENT_BEFORE`;
- robbery with alarm: the authored alarm branch followed by
  `DEPARTURE_OR_ESCAPE`;
- talking/rendezvous: `CONVERSATION`;
- two- and three-visit stalking: `EXITS`, followed by `ENTERS`;
- two-vehicle chase: `FOLLOWS`;
- vehicle convergence: `DISTANCE_LT`.

The deterministic traversal chooses the alarm branch for the robbery-with-alarm
catalog variant. Other authored robbery branches require their own catalog
variant/ground-truth branch selection rather than treating every OR child as a
simultaneous demand.

The commands below run the parts that are currently connected end to end.

---

# 1. One-time setup and validation

Run from the FABLE root:

```bash
cd /home/brianw/Documents/FABLE

python -m pip install -e .

python scripts/validate_evaluation.py
python scripts/build_evaluation_manifests.py
python scripts/export_evaluation_schemas.py
```

Inspect the available ground-truth experiments:

```bash
python iobt-minimal-ce-replay/tools/fable_evaluation_catalog.py \
  --recommended-only | less
```

For a 2024 route-convoy spatial experiment:

```bash
python iobt-minimal-ce-replay/tools/fable_evaluation_catalog.py \
  --year 2024 \
  --variant "route convoy" \
  --recommended-only \
  --spatial-only | less
```

For a 2025 robbery experiment:

```bash
python iobt-minimal-ce-replay/tools/fable_evaluation_catalog.py \
  --year 2025 \
  --variant robbery \
  --recommended-only \
  --spatial-only | less
```

Generate the planned run matrices:

```bash
python scripts/plan_evaluation_runs.py RQ1_END_TO_END \
  --output evaluation/manifests/workloads/rq1_end_to_end.jsonl

python scripts/plan_evaluation_runs.py RQ2_PLANNING \
  --output evaluation/manifests/workloads/rq2_planning.jsonl

python scripts/plan_evaluation_runs.py RQ3_OPERATING_ADAPTATION \
  --output evaluation/manifests/workloads/rq3_adaptation.jsonl

python scripts/plan_evaluation_runs.py RQ3_SPATIAL_COORDINATION \
  --output evaluation/manifests/workloads/rq3_spatial.jsonl

python scripts/plan_evaluation_runs.py RQ3_CONTINUATION \
  --output evaluation/manifests/workloads/rq3_continuation.jsonl
```

Inspect one:

```bash
head -n 5 evaluation/manifests/workloads/rq3_spatial.jsonl | jq
```

These commands **plan** runs; they do not execute them.

---

# 2. Select one experiment

Suppose you select:

```text
experiment_id:
20241008-route-convoy-1-r012

duration:
98 seconds
```

There are two different identifiers:

* `FABLE_EVAL_TRACE_ID` is the ground-truth experiment ID.
* `SCENARIO` is the actual recording filename prefix understood by replay, usually something resembling `YYYYMMDD_HHMMSS`.

For example, the likely prefix for the above record would be:

```bash
SCENARIO=20241008_110824
```

Verify that it actually exists:

```bash
find "/media/brianw/Extreme SSD" \
  -type f \
  -name "${SCENARIO}*" \
  | head
```

Do not assume the ground-truth ID itself is a valid replay scenario.

Create a shared environment file so all terminals use identical identifiers:

```bash
cat > /tmp/fable-eval.env <<'EOF'
export FABLE_EVAL_RUN_ID="rq1-fable-convoy-r012-rep01"
export FABLE_EVAL_TRACE_ID="20241008-route-convoy-1-r012"
export FABLE_EVAL_REQUEST_ID="convoy-r012"
export FABLE_EVAL_BASELINE="FABLE"
export FABLE_EVAL_OUTPUT_DIR="$HOME/fable-evaluation-runs"

export SCENARIO="20241008_110824"
export REPLAY_START="0"
export REPLAY_END="98"
EOF
```

Use the exact variable name:

```text
FABLE_EVAL_BASELINE
```

not `FABLE_EVAL_BASELINE_ID`.

Valid values include:

```text
B0_ALWAYS_ON
B1_HANDWRITTEN_STATIC
B2_STATIC_WHOLE_EVENT
B3_TASK_RESOURCE_ADAPTIVE
B4_GREEDY_FRONTIER
FABLE
O1_EXHAUSTIVE_ORACLE
SPATIAL_BROADCAST
SPATIAL_TOPOLOGY_SHORTLIST
SPATIAL_RESOURCE_ONLY
SPATIAL_FABLE
SPATIAL_ORACLE
```

---

# 3. Terminal 1 — start the complete stack

```bash
source /tmp/fable-eval.env

cd /home/brianw/Documents/FABLE/iobt-minimal-ce-replay

./tools/run_fable_evaluation.sh
```

This starts, in the foreground:

* Mosquitto;
* replay containers;
* YOLO and audio providers;
* MongoDB;
* FABLE orchestrator;
* FABLE node agents;
* Phase-7 vehicle provider;
* Phase-8 multimodal provider;
* evaluation logger;
* web UI.

The web UI should be available at:

```text
http://localhost:8080
```

The evaluation output is mounted at:

```text
$HOME/fable-evaluation-runs
```

## Detached alternative

The evaluation script does not currently forward `-d`, so use the Compose command directly:

```bash
source /tmp/fable-eval.env

cd /home/brianw/Documents/FABLE/iobt-minimal-ce-replay

docker compose \
  -f compose.server.yaml \
  -f compose.replay.yaml \
  -f compose.fable.yaml \
  -f compose.fable.phase7.yaml \
  -f compose.fable.phase8.yaml \
  -f compose.fable.evaluation.yaml \
  up -d --build
```

With detached mode, one terminal can technically run everything.

---

# 4. Terminal 2 — compile, submit, and replay

Load the same environment:

```bash
source /tmp/fable-eval.env
```

## 4.1 Check request compilation

```bash
cd /home/brianw/Documents/FABLE

python scripts/compile_request.py "detect a convoy"
```

This verifies that the request maps to the authored convoy family and prints the semantic graph.

**This command only compiles the graph. It does not submit the complete graph to the distributed orchestrator.**

## 4.2 Submit the first vehicle predicate

The currently connected replay submission tool submits an individual predicate demand. For a convoy smoke test, begin with `PASSES`:

```bash
cd /home/brianw/Documents/FABLE/iobt-minimal-ce-replay

python tools/fable_submit_vehicle.py \
  --predicate PASSES \
  --request-id "$FABLE_EVAL_REQUEST_ID" \
  --reference-id camera_a_gate
```

The output should contain one or more admitted plan IDs.

Other available vehicle tests are:

```bash
python tools/fable_submit_vehicle.py \
  --predicate MOVING \
  --request-id "$FABLE_EVAL_REQUEST_ID"
```

```bash
python tools/fable_submit_vehicle.py \
  --predicate STOPPED \
  --request-id "$FABLE_EVAL_REQUEST_ID"
```

`FOLLOWS` requires an observed leader identity:

```bash
python tools/fable_submit_vehicle.py \
  --predicate FOLLOWS \
  --leader-id '<source>:<tracker-session>:<track-id>' \
  --request-id "$FABLE_EVAL_REQUEST_ID"
```

## 4.3 Start replay

For an end-to-end timing run, use real-time playback:

```bash
python tools/replay_control.py \
  --scenario "$SCENARIO" \
  --start "$REPLAY_START" \
  --end "$REPLAY_END" \
  --playback-mode realtime \
  --sync-delay 3
```

For a faster plumbing test:

```bash
python tools/replay_control.py \
  --scenario "$SCENARIO" \
  --start "$REPLAY_START" \
  --end "$REPLAY_END" \
  --playback-mode max \
  --sync-delay 3
```

Use `realtime` for:

* detection-delay measurements;
* provider startup and readiness;
* coordination lead time;
* adaptation timing;
* resource/network experiments.

Use `max` mainly for:

* debugging;
* common-perception preparation;
* functional regression tests;
* high-throughput offline processing.

---

# 5. Terminal 3 — monitor the run

Follow the evaluation logger:

```bash
docker logs -f fable-evaluation-logger
```

Follow the orchestrator:

```bash
docker logs -f fable-orchestrator
```

Watch all MQTT traffic:

```bash
cd /home/brianw/Documents/FABLE/iobt-minimal-ce-replay

python tools/mqtt_tail.py \
  --topic '#' \
  --max-len 1000
```

A narrower FABLE-only view is:

```bash
python tools/mqtt_tail.py --topic 'fable/#'
```

Watch replay and vehicle-provider containers:

```bash
docker logs -f zed-replay-orin11
```

```bash
docker logs -f yolo-detector-orin11
```

```bash
docker logs -f fable-vehicle-orin11
```

---

# 6. Inspect the output

After replay finishes:

```bash
source /tmp/fable-eval.env

RUN_DIR="$FABLE_EVAL_OUTPUT_DIR/$FABLE_EVAL_RUN_ID/$FABLE_EVAL_BASELINE"

find "$RUN_DIR" -maxdepth 1 -type f -printf '%f\n'
```

Count records:

```bash
for file in "$RUN_DIR"/*.jsonl; do
    printf '%-45s ' "$(basename "$file")"
    wc -l < "$file"
done
```

Inspect records:

```bash
jq . "$RUN_DIR/plan_decision.jsonl" | less
```

```bash
jq . "$RUN_DIR/provider_lifecycle_event.jsonl" | less
```

```bash
jq . "$RUN_DIR/resource_sample.jsonl" | less
```

The exact filenames depend on which MQTT records appeared during that run, so start with `find`.

---

# 7. Stop and clean up

If Terminal 1 is running in the foreground, press `Ctrl+C`.

Then:

```bash
cd /home/brianw/Documents/FABLE/iobt-minimal-ce-replay

docker compose \
  -f compose.server.yaml \
  -f compose.replay.yaml \
  -f compose.fable.yaml \
  -f compose.fable.phase7.yaml \
  -f compose.fable.phase8.yaml \
  -f compose.fable.evaluation.yaml \
  down
```

For an independent **cold-start repetition**, also remove persistent volumes:

```bash
docker compose \
  -f compose.server.yaml \
  -f compose.replay.yaml \
  -f compose.fable.yaml \
  -f compose.fable.phase7.yaml \
  -f compose.fable.phase8.yaml \
  -f compose.fable.evaluation.yaml \
  down -v
```

Using `-v` clears MongoDB, MQTT persistence, node-agent state, and orchestrator state. Do not use it when testing restart recovery or warm-state behavior.

---

# 8. Repeating a run

Change at least the run ID:

```bash
export FABLE_EVAL_RUN_ID="rq1-fable-convoy-r012-rep02"
```

For paired baseline runs, keep all of these identical:

```text
trace
replay offsets
playback mode
request
network profile
provider thresholds
retention settings
```

Change only:

```bash
export FABLE_EVAL_BASELINE="B3_TASK_RESOURCE_ADAPTIVE"
export FABLE_EVAL_RUN_ID="rq1-b3-convoy-r012-rep01"
```

However, with the current implementation, this changes the **logged baseline label only**. It does not yet install the B3 policy into the live orchestrator.

---

# 9. Spatial evaluation checks

The spatial matrix is generated with:

```bash
cd /home/brianw/Documents/FABLE

python scripts/plan_evaluation_runs.py RQ3_SPATIAL_COORDINATION \
  --output evaluation/manifests/workloads/rq3_spatial.jsonl
```

Only 2024 and 2025 records are included. Mobile sensors are retained as unavailable candidates but are not activated.

Before attempting a full-stack spatial run, check how many Orin replay services actually exist:

```bash
cd /home/brianw/Documents/FABLE/iobt-minimal-ce-replay

docker compose \
  -f compose.server.yaml \
  -f compose.replay.yaml \
  config --services \
  | grep -E '^(zed|yolo)-orin'
```

In the replay archive you attached, `compose.replay.yaml` currently instantiates only:

```text
zed-orin11
yolo-orin11
respeaker-orin11
audio-detector-orin11
```

Therefore, that specific Compose file cannot measure actual cross-Orin handoff or activation fan-out. It can test the spatial policy logic offline, but a full spatial experiment needs replay services for multiple fixed Orins.

You can regenerate the replay Compose file from the available data roots:

```bash
cd /home/brianw/Documents/FABLE/iobt-minimal-ce-replay

cp compose.replay.yaml compose.replay.orin11-only.yaml

python setup/generate_replay_compose.py \
  --data-dir "/media/brianw/Extreme SSD/West Point Experimentation" \
  --data-dir "/media/brianw/Extreme SSD/GQ Data" \
  --compose-out compose.replay.yaml
```

Or explicitly name fixed Orin folders:

```bash
python setup/generate_replay_compose.py \
  --data-dir "/media/brianw/Extreme SSD/West Point Experimentation" \
  --data-dir "/media/brianw/Extreme SSD/GQ Data" \
  --nodes orin11 orin12 orin14 \
  --compose-out compose.replay.yaml
```

Replace those names with the actual Orin folders present for the selected 2024/2025 recording.

Do not combine a regenerated replay file with the old `orin11`-only FABLE
overlays. Generate a coherent scenario bundle instead:

```bash
cd /home/brianw/Documents/FABLE

venv/bin/python \
  iobt-minimal-ce-replay/setup/generate_evaluation_bundle.py \
  --scenario 20241008_110823 \
  --max-nodes 3
```

This writes replay services, one vehicle stack, multimodal stack, and node agent
per selected fixed Orin, plus the matching deployment graph, provider-runtime
map, aliases, and deployment-artifact manifest under
`iobt-minimal-ce-replay/generated/evaluation_bundles/<scenario>/`.

The default `spectral-rule` audio backend is suitable only for stack smoke
tests. Evaluation-quality audio must be requested explicitly with
`--audio-backend yamnet`; generation then fails unless `models/yamnet` and
`models/yamnet_class_map.csv` exist. Mobile `n*` and `d*` services remain
intentionally deferred.

Cross-sensor person and vehicle identity is also opt-in and fails closed.
Provision separately calibrated Torchreid checkpoints at
`iobt-minimal-ce-replay/models/reid/person.pth` and
`iobt-minimal-ce-replay/models/reid/vehicle.pth`, then generate with:

```bash
venv/bin/python \
  iobt-minimal-ce-replay/setup/provision_reid_models.py

venv/bin/python \
  iobt-minimal-ce-replay/setup/generate_evaluation_bundle.py \
  --scenario 20241008_110823 \
  --max-nodes 3 \
  --enable-reid
```

The provisioner pins and verifies OSNet-AIN/MSMT17 for people and FastReID
SBS-R50-IBN/VeRi for vehicles. Model versions and preprocessing contracts come
from `models/reid/models.json`; do not use the person checkpoint as vehicle
identity evidence. With ReID disabled, the generated runtime map omits
`cross_sensor_identity_association` instead of substituting reference truth.

After a provider or evaluation gateway has observed an authored seed predicate,
submit it through the typed live-request protocol. For example:

```bash
timeout 120s venv/bin/python \
  iobt-minimal-ce-replay/tools/fable_submit_event.py \
  --request-id convoy-20241008-1 \
  --run-id rq1-convoy-1 \
  --trace-id replay-20241008-110823 \
  --family convoy \
  --baseline FABLE \
  --seed-node-key leader_passes \
  --occurrence-id orin1-leader-pass-1 \
  --source orin1_camera \
  --node dvpg_gq_orin_1 \
  --event-start 2024-10-08T11:08:23.000Z \
  --event-end 2024-10-08T11:08:23.500Z \
  --introduced-bindings-json '{"leader":"vehicle-17"}'
```

The client cannot submit an executable graph or provider command. The
orchestrator resolves the family from the authored registry, reconstructs the
seed predicate and graph hash, creates the hypothesis, validates live provider
coverage, and dispatches its first frontier. Its wait is capped at 120 seconds.

---

# What is runnable now versus still missing

### Runnable now

* Catalog and ground-truth inspection.
* RQ run-matrix generation.
* Schema generation and validation.
* Replay stack startup.
* FABLE distributed provider execution.
* Direct vehicle, audio, and multimodal predicate submissions.
* MQTT-to-JSONL evaluation logging.
* Offline unit tests of B0–B4, oracle, spatial policies, and metrics.
* Scenario-selected multi-Orin replay/provider/control-plane bundles.

### Not yet exposed as a complete command

* Submitting `"detect a convoy"` as a full semantic graph to the distributed orchestrator.
* Automatically executing every row of a run matrix.
* Switching the live orchestrator among B0, B1, B2, B3, B4, and FABLE.
* Automatically injecting the requested network profile per run.
* Automatically aggregating JSONL records into final metric tables.
* Evaluation-quality YAMNet execution until its pinned SavedModel and class map
  are provisioned.

The next required executable layer is a driver resembling:

```bash
python scripts/run_evaluation.py \
  --question RQ1_END_TO_END \
  --experiment 20241008-route-convoy-1-r012 \
  --baseline FABLE \
  --mode FULL_STACK \
  --repeat 1
```

That driver needs to submit the compiled CE graph, install the selected baseline policy, control replay, wait for completion, and invoke metric aggregation.
