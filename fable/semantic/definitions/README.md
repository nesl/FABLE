# Complex-event definitions

This directory is the canonical home and import surface for executable FABLE
complex-event definitions.

## Adding a complex event

Add `fable/semantic/definitions/<my_event>.py`, author the graph with
`fable.semantic.authoring.ComplexEvent`, and add one entry to `registry.py`.
That registry is the only production family-ID selection table used by request
compilation.

## Files

- `package_exchange.py`, `route_convoy.py`, `robbery_with_alarm.py`,
  `drive_up_shooting.py`, `repeated_visit.py`, `talking_rendezvous.py`,
  `vehicle_convergence.py`, and `two_vehicle_chase.py`: canonical public modules
  organized by complex event.
- `registry.py`: family ID, canonical factory, aliases, and warnings.
- `vehicle.py` and `multimodal.py`: thin compatibility re-exports for historical
  imports. They contain no authored factory implementations.
- `policy.py`: controller-facing scene-clear rearm metadata. Experimental
  return-to-start motion is labelled `TRIAL_RESET` and is not part of semantic
  completion.
- `__init__.py`: compatibility import surface, including the
  pass-follow-clear convoy graph retained from the legacy common graph module
  for graph-hash compatibility.

Import definitions from `fable.semantic.definitions`. Do not add new complex
event graphs to `fable.semantic.examples`, `fable.semantic.phase8_examples`,
evaluation code, or provider implementations.

Evaluation labels under `evaluation/labels` describe ground truth. Providers
under `providers` implement primitive predicates. Neither is an executable
complex-event definition.

## Completion boundaries

- Vehicle convergence completes after group convergence and asynchronous exits
  by every originally bound vehicle; it does not impose an additional dwell.
- Full talking rendezvous completes after interaction and the bound arrival
  vehicle exiting; participant boarding is not required. The visual
  co-presence graph remains a deliberately simpler proxy.
- Package exchange completes after transfer and departure of the bound
  receiving vehicle; a separate source-vehicle departure is not required.
- Repeated visits require a real exit or a finalized pass followed by a
  configurable absence gap before the same vehicle can return.
- Drive-up shooting ends with the bound vehicle exiting. Boarding is required
  by default; `require_boarding=false` selects the documented variant for
  traces where boarding is unobservable.
