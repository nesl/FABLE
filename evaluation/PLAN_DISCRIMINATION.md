# Replay-free plan discrimination

`scripts/run_plan_discrimination.py` checks that the physical planner has a
real cost crossover before a live RQ3 campaign is started. It compiles a real
complex-event frontier and enumerates the real provider chains, but it does not
start containers or replay sensor recordings.

Run it with:

```bash
timeout 120s .venv/bin/python scripts/run_plan_discrimination.py \
  --output evaluation/results/plan_discrimination_$(date +%Y%m%d_%H%M%S)
```

The checked-in manifest is
`evaluation/manifests/adaptation/plan_discrimination.yaml`. It declares:

- the trace used to compile the semantic frontier;
- network profiles and policy IDs;
- temporary node-class timing multipliers;
- whether synthetic raw-stream transfer is permitted; and
- the expected detector placement and transfer bound per condition.

The timing multipliers do not modify E0 results or production provider
profiles. They are explicit assumptions used to verify the mechanism while all
logical tiers still run on one desktop. Replace them with tier-specific E0
measurements when embedded and edge measurements are available.

Outputs are:

- `selected_plans.csv`: independent admission choices under each condition;
- `transition_plans.csv`: one request spanning the condition change, with B2
  frozen and B3/FABLE applying the resource-epoch plan;
- `alternatives.json`: every enumerated placement, cost, transfer, and path;
- `result.json`: fail-closed expected-plan and transition validation.

The test is intentionally separate from CE accuracy. Passing it proves that
provider placement is sensitive to the declared compute/network costs. It does
not prove that a selected runtime container can execute the chain or that the
chain produces correct predicates; those remain live preflight and replay
requirements.
