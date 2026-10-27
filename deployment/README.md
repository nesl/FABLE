# Deployment internals

This directory contains the Docker images, Compose overlays, MQTT settings,
and node configuration used by `./run`. They are implementation details rather
than additional case configuration.

- `compose.example.yaml` is the shared Compose base.
- `compose.direct.yaml` runs agents on ordinary host networking.
- `compose.netwaggle.real.yaml` lets NetWaggle attach the agents.
- `config/` describes the available sensor and edge nodes/providers.

Do not put event names or recording paths here. Define those once in
`replay/cases/<case>.yaml`; select the reusable replay method with
`replay/plans/<plan>.yaml`.

Use the root launcher rather than invoking these Compose fragments directly:

```bash
./run stack
./run --case convoy1 --plan network_nominal
```
