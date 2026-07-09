#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cd "$REPLAY_DIR"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
path = Path('generated/scenario_catalog.json')
if not path.exists():
    raise SystemExit('generated/scenario_catalog.json does not exist. Generate the scenario catalog first.')
with path.open('r', encoding='utf-8') as f:
    data = json.load(f)
for s in data.get('scenarios', [])[:100]:
    sid = s.get('scenario_id') or s.get('id') or s.get('name')
    nodes = ','.join(s.get('nodes', []) or [])
    modalities = ','.join(s.get('modalities', []) or [])
    valid = s.get('valid', '')
    print(f"{sid}\tnodes={nodes}\tmodalities={modalities}\tvalid={valid}")
PY
