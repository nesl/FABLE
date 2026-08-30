# FABLE evaluation

This package is the clean replacement for the old evaluation tree. It does not
import old semantic, demand-graph, scheduler, or MQTT contracts.

The initial supported workflow is a deterministic planning smoke evaluation:

```text
manifest cell
  -> compile CE YAML
  -> load deployment RuntimeState
  -> derive discovery frontier
  -> invoke the selected evaluation policy
  -> record plan and timing as JSONL
  -> aggregate a JSON/CSV report
```

Full recording replay will be attached through `replay/` source adapters. The
result schema is intentionally usable by both planning-only and live campaigns.
Live controllers can write durable `fable.completion.v1` JSONL records with
`scripts/run_fable.py --output-jsonl PATH`; `evaluation.artifacts` loads and
scores their binary presence without interpreting provider-private artifacts.

Current policies are implemented independently:

- `FABLE`: normal measured-network, resource-aware frontier planning.
- `B1_STATIC`: an exact whole-event placement loaded once and held fixed.
- `B3_RESOURCE`: compute/resource adaptation with network measurements ablated.
- `B4_GREEDY`: current-frontier planning with width-one joint search.

Directories:

- `manifests/`: immutable experiment inputs.
- `baselines/`: evaluation policies; these must not alter FABLE core behavior.
- `metrics/`: report helpers and metric definitions.
- `results/`: generated and git-ignored.
- `labels/`: immutable experiment metadata; `evaluation.catalog` is its typed reader.

Run the smoke matrix:

```bash
python scripts/run_evaluation.py \
  evaluation/manifests/smoke.yaml \
  --output evaluation/results/smoke/results.jsonl
```

Generate review-only recording candidates from an attached archive:

```bash
python scripts/build_recording_candidates.py \
  evaluation/labels/filtered_complex_event_experiments.csv \
  "/media/brianw/Extreme SSD3" \
  evaluation/results/recording_candidates.ssd3.yaml
```

Generated rows are always `verified: false`; a human must confirm them before
they are promoted into an immutable campaign manifest.
