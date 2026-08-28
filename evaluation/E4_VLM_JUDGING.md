# Bounded E4 VLM reference judging

This workflow measures identity-binding precision without allowing the judge to
affect live execution. Its outputs are VLM-derived reference labels, not human
ground truth.

## Design

- Use 3–6 labeled, identity-heavy traces, stratified across convoy, repeated
  visit, and robbery/rendezvous when evidence is available.
- Run every compared policy on the same traces and replay windows.
- Capture boxed full-frame evidence only for identity associations the policy
  emits. The capture manifest does not contain the API key.
- Deduplicate identical visual pairs across policies using SHA-256.
- Judge pairs after execution, blinded to policy and prediction metadata.
- Cache every model response. `UNDETERMINED` judgments count as judge coverage,
  not as an incorrect binding.
- Combine VLM-judged binding precision with labeled CE precision/recall and
  timely recall. Judging emitted matches alone does not measure identity recall.

## Capture

Pass an evidence root to the bounded suite:

```bash
./.venv/bin/python scripts/run_full_ce_suite.py \
  --output-dir evaluation/results/e4_live/<policy> \
  --baseline FABLE \
  --evaluation-policy-id C3_FABLE_ESCALATION \
  --identity-escalation-policy-id C3_FABLE_ESCALATION \
  --identity-judge-evidence-root evaluation/results/e4_live/evidence/<policy> \
  --experiment-id <bounded-trace-id>
```

Repeat `--experiment-id` for the selected traces. Evidence manifests are written
as `evidence/<policy>/<experiment-id>/identity_predictions.jsonl`.

Run the same bounded trace set separately for C0 through C4. C0 and C4 terminate
after local descriptor matching, C1 invokes only the bounded hosted comparator,
C2 invokes the hosted comparator when local matching fails, and C3 also treats a
low-confidence local association as ambiguous and escalates it. C1–C3 retain the
four-call-per-replay hosted-provider safety limit.

## Cost preview and judging

First omit `--execute`. The command reports the exact number of uncached paid
calls and exits without contacting the API:

```bash
./.venv/bin/python scripts/judge_e4_identity_predictions.py \
  --manifest <identity_predictions.jsonl> \
  --output evaluation/results/e4_live/judged \
  --cache evaluation/results/e4_live/vlm_judge_cache.json \
  --maximum-unique-pairs 30
```

After reviewing that count, export `OPENAI_API_KEY` in the launching shell and
repeat with `--execute`. Multiple `--manifest` arguments are accepted. The hard
pair cap refuses oversized experiments rather than silently sampling or spending
more than intended.

Outputs:

- `identity_judgments.jsonl`: prediction, blinded judgment, confidence, and
  agreement for every policy row;
- `identity_judgment_summary.json`: VLM-judged precision and judge coverage by
  policy;
- `vlm_judge_cache.json`: reusable pair decisions keyed by model and evidence
  hash.

If a policy under evaluation itself invokes a VLM, use a distinct judge model
where possible and report both model IDs. The current C0–C4 profiled campaign
simulates task outcomes; an empirical C0–C4 claim requires those provider-control
policies to execute in the live stack and feed this same capture path.
