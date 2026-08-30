# Container deployment

This is a small deployment wrapper around the refactored controller and node
agent. It intentionally contains no Mosquitto, MongoDB, old orchestrator, or
legacy provider-contract services.

The example is structural: copy the configuration and replace source URIs and
controller addresses before running it. Recording mounts are read-only.
