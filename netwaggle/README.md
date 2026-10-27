# NetWaggle integration

NetWaggle supplies reusable network environments around FABLE. It creates the
sensor/edge topology, applies bandwidth, latency, loss, or connectivity
conditions, and publishes the measured link state back to the FABLE planner.

It is an integration layer: `fable/` does not import `netwaggle/`.

## Reusable network plans

User-facing plans live under [`replay/plans`](../replay/plans) and contain no
case or event names:

- `direct.yaml` uses ordinary Docker networking and does not start NetWaggle.
- `network_nominal.yaml` starts NetWaggle without injecting an impairment.
- `network_disconnect.yaml` starts NetWaggle and disconnects the sensor/edge uplink.

Any replay case can use any plan:

```bash
./run --case convoy1 --plan direct
./run --case convoy1 --plan network_nominal
./run --case convoy1 --plan network_disconnect
```

A NetWaggle plan contains a topology and a condition schedule. Conditions are
relative to replay release:

```yaml
version: 1
mode: netwaggle
topology: netwaggle/topologies/sensor_edge.yaml
conditions:
  - starts_at_seconds: 0
    affected_link_name: sensor_edge_uplink
    connection_status: disconnected
```

The runner invokes sudo only for the Mininet/OVS process. Direct mode bypasses
sudo, Mininet, OVS, the NetWaggle control socket, and imports from the
`netwaggle` package. The shared replay-plan parser lives in `replay/`, so the
direct execution path is independent of this integration package.
