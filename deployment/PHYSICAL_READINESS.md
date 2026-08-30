# Physical deployment readiness

Physical inventories are installation-specific. Keep live addresses, model
paths, recordings, and endpoint status in ignored local configuration files;
commit only the portable `*.example.yaml` templates.

Run the read-only endpoint check from the repository root:

```bash
python scripts/preflight_deployment.py config/deployment.physical.local.yaml
```

Before installation, copy this clean source tree to a new directory on each
device. Install the package into a dedicated virtual environment, provide a
device-specific ignored NodeAgent configuration, and start
`scripts/run_node_agent.py` from that environment. No password, API key,
recording, model, or live inventory belongs in Git.

Live Mininet validation is separate. It ordinarily requires root because it
creates namespaces/veth pairs and controls OVS/TC.
