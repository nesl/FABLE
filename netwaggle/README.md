# NetWaggle integration

NetWaggle is the network-emulation layer surrounding, not embedded in, FABLE.
Its responsibilities are:

1. create or attach logical network links;
2. apply versioned bandwidth, latency, loss, and availability conditions;
3. measure the applied condition;
4. translate measurements into FABLE `LinkState` updates.

The Mininet/OVS runner is isolated from FABLE core and maintained independently
because it does not depend on the old semantic runtime. Until that runner is
fully validated, this directory exposes only the typed bridge and condition
schema; it does not claim that privileged host mutations are available.

`fable/` must never import this package. The controller or evaluation harness
may consume `NetwaggleLinkObservation` and update `RuntimeState`.
