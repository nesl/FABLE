# West Point Sensor-Transition Model

This package separates two questions:

1. **Which complex-event variant corresponds to each supplied scenario?**
   See `complex_event_scenario_mapping_2024_2025.json`.
2. **Which sensor is likely to observe an object next?**
   See `site_sensor_transition_model_2024_2025.json`.

## Important interpretation rules

- The letters A, B, C, D, and E are **scenario-local labels**. They do not name
  permanent site locations. The topology therefore uses stable geographic zone
  names such as `northeast_entry`, `central_south_junction`, and `west_entry`.
- Ordered sensor entries are represented as **observation groups**. Sensors in one
  group have overlapping or nearly adjacent coverage and may observe an object at
  roughly the same time.
- The model is qualitative. It was inferred from the drawn routes and camera
  sectors, not from surveyed coordinates or calibrated camera polygons.
- All devices have microphones, but the JSON predicts camera handoffs only.
  Microphone range, directionality, and obstruction must be measured before audio
  can be used for next-sensor prediction.

## Suggested runtime procedure

1. Select the current `mobile_deployment` corresponding to the experiment.
2. Determine the camera currently observing the object.
3. Estimate a coarse heading such as west, southwest, or northeast.
4. Look up the camera and heading in `directional_next_sensor_rules`.
5. If no direct rule matches, select the applicable `corridor` and move to its next
   `fixed_observation_group`.
6. Wake or prioritize every node in the next group when:
   - the fields of view overlap,
   - the branch is unresolved, or
   - the model marks the transition medium/low confidence.
7. Update the model with measured transition frequencies after replaying the data.

## Example

An object seen by `orin_6` moving southwest on the eastern arc should next be
prioritized at `orin_5`. In the 2025 package-exchange or flee-police deployments,
`d3` should also be activated because it overlaps the northeast/east approach.

An object seen near the central junction moving west should generally progress
through:

`orin_1 -> {orin_8, orin_10} -> orin_9`

The braces indicate an overlapping observation group rather than a guaranteed
strict order.

## Known ambiguity

Pages 5 and 6 of the 2024 PDF show different mobile-node placements for the chase
scenario. The JSON preserves them as `2024_temporal_ce2_page5` and
`2024_temporal_ce2_page6`; software should select the layout matching the actual run.

