# Porting status from `FABLE_old`

The old repository is reference material only. This table prevents copied
folders from being mistaken for working integrations.

| Area | Status | New boundary |
|---|---|---|
| CE definitions | Ported | `ce_definitions/` and `fable.language` |
| Provider algorithms | Ported core | `fable.providers` |
| Semantic hypotheses/frontiers | Rebuilt | `fable.runtime` |
| Physical planning | Rebuilt | `fable.planning` |
| Node execution | Rebuilt | `fable.execution` |
| Evaluation contracts/metrics | Ported | `evaluation/` |
| Baselines | Clean planning policies ported | `evaluation/baselines/` |
| NetWaggle state bridge | Ported | `netwaggle/bridge.py` |
| Mininet/OVS runner | Ported; host validation remains | `netwaggle/runner.py` |
| Recording replay | Source adapters and discovery ported | `replay/` |
| Docker testbed | Clean controller/agent example ported | `deployment/` |
| Labels/manifests | Typed catalog and v1 manifests ported | `evaluation/labels`, `evaluation/manifests` |
| Historical results/debug | Not ported | external archival storage |

## Items that must not be copied unchanged

- Old Python evaluation modules importing `fable.common`, `fable.semantic`,
  `fable.distributed`, or the old scheduler/planner.
- Old campaign manifests whose policy identifiers or result contracts refer to
  those modules.
- Generated Compose bundles and machine-specific paths.
- Results, caches, recordings, virtual environments, and model checkpoints.

## Deliberately remaining work

- Validate privileged Mininet/OVS execution on the target host.
- Generate explicit recording joins for each campaign; discovery is available,
  but filename timestamps alone are not ground-truth associations.
- Deploy the refactored NodeAgent on the known Pi/Jetson inventory and validate
  its declared TCP control endpoints.

See `deployment/PHYSICAL_READINESS.md` for the checked device/interpreter and
endpoint state. The local implementation is ready; remote installation requires
the user's authenticated device sessions.
