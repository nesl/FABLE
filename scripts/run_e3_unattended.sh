#!/usr/bin/env bash
set -euo pipefail

if [[ "${FABLE_CONFIRM_E3:-}" != "YES" ]]; then
  echo "Refusing to execute E3: set FABLE_CONFIRM_E3=YES" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest_dir="$repo_root/evaluation/manifests/spatial/e3_prepared_20260827"
result_root="${FABLE_E3_RESULT_ROOT:-/media/brianw/Extreme SSD2/fable_results/e3_campaign_20260827}"

cd "$repo_root"

# Codex-launched shells do not necessarily inherit tmux's global environment.
# Import the already-configured key without printing it; never persist it in a
# manifest or result artifact.
if [[ -z "${OPENAI_API_KEY:-}" ]] && command -v tmux >/dev/null 2>&1; then
  e3_openai_entry="$(tmux show-environment -g OPENAI_API_KEY 2>/dev/null || true)"
  if [[ "$e3_openai_entry" == OPENAI_API_KEY=* ]]; then
    export OPENAI_API_KEY="${e3_openai_entry#OPENAI_API_KEY=}"
  fi
fi
"$repo_root/.venv/bin/python" scripts/run_rq3_campaigns.py \
  --root "$result_root" \
  --manifest-dir "$manifest_dir" \
  --only rq3b \
  --only rq3c \
  --preflight-only

"$repo_root/.venv/bin/python" scripts/run_rq3_campaigns.py \
  --root "$result_root" \
  --manifest-dir "$manifest_dir" \
  --only rq3b \
  --only rq3c \
  --max-seconds 360 \
  --ready-seconds 30
