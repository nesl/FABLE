# FABLE presentation replay cases

This file records reviewed real-data experiments that are suitable candidates
for live presentations. A case is marked **verified** only after the named CLI
command has produced a passing `validation.json` and a dashboard video.

## Audited

### 2025 robbery with alarm — `robbery_alarm1` (partially valid)

- Ground-truth experiment: `20250812-robbery-with-alarm-burglary-c-r024`
- Ground-truth family: Burglary-C, marked usable
- Observable CE used by the demo: vehicle arrives, a real alarm is classified,
  then a vehicle leaves the monitored area
- Cameras displayed: ten fixed Orin cameras plus three mobile cameras
- Inference sources: `orin6_audio`, `orin1_camera`, `orin4_camera`,
  `orin7_camera`
- Nominal command: `./run --plan network_nominal --case robbery_alarm1`
- Disconnect command: `./run --plan network_disconnect --case robbery_alarm1`
- Completed nominal run:
  `.fable/runs/network/robbery_alarm1_normal-20260913T000704Z`
- Dashboard recording:
  `.fable/runs/network/robbery_alarm1_normal-20260913T000704Z/devices-dashboard.mp4`
- Completion: `2025-08-12T19:09:25.733333Z`
- Caveat: arrival and departure use separate track identities. This demo does
  not yet prove cross-camera or long-gap vehicle re-identification.
- Evidence audit: the alarm and departure are supported, but the arrival is
  not. Both vehicles are already parked in the arrival image; initial track
  creation was incorrectly interpreted as `enters`. See
  `.fable/runs/presentation-evidence/robbery_alarm1-r024/`.
- Post-fix run: `.fable/runs/network/robbery_alarm1_normal-20260913T010142Z`.
  It correctly did not complete: the false initial arrival disappeared and the
  sole later entry candidate occurred after the alarm window. The case remains
  unsuitable for presentation; see
  `.fable/runs/presentation-evidence/robbery_alarm1-r024-post-fix/`.
- Detailed timeline review found that the alarm precedes the visible building
  entry by roughly 25 seconds. The stricter CE now requires a person to
  disembark from the arriving vehicle before the alarm; r024 does not satisfy
  that sequence. See `.fable/runs/presentation-evidence/robbery_alarm1-r024-timeline/`.

### Alternative robbery runs reviewed

Replay calibration for the 2025 Burglary-A family:

- Mobile recordings visually lag the Orin recordings by about four seconds.
  Runnable A manifests and the manifest generator therefore assign mobile
  event times four seconds earlier than their filename timestamps.
- Mobile 2, 3, and 4 require orientation correction before frames reach both
  inference and the device dashboard. Manifest rotation is applied upstream,
  so YOLO sees the same corrected image as the dashboard.
- Mobile 5 is the useful doorway view for Burglary-A and is now an inference
  source rather than a display-only source.
- Alarm evidence exists in both the Orin 6 ReSpeaker and the mobile recordings.
  The r008/r009 manifests now include extracted Mobile 4 audio alongside Orin
  6. Offline YAMNet checks confirm the alarm in mobile 4/5/6 for r009 and
  mobile 4/6 for r008 (mobile 5 peaks just below the 0.4 threshold). With the
  four-second mobile correction, those detections align closely with Orin 6.
- A diagnostic r009 replay with both Orin 6 and extracted mobile 4 audio found
  that the source scheduler aligns their event times correctly. The apparent
  timing failure occurs downstream: per-frame `present` results and fragmented
  track IDs create a large CE-candidate backlog, so the controller enables or
  processes the alarm frontier after the synchronized audio window has passed.
  `PresentBasicProvider` now emits once per track, but the long-lived stack must
  be restarted before its existing edge/controller processes use rebuilt code.

- Burglary-A r009: `.fable/runs/network/robbery_a_r009_normal-20260913T013916Z`.
  Two vehicle arrivals were detected, then a person at +47 s, but no valid
  vehicle/person `disembarks` transition. Independent YAMNet analysis found a
  strong siren at +42 s, five seconds before that person observation. It did
  not complete and is not presentation-verified.
- Corrected Burglary-A r009 rerun:
  `.fable/runs/network/robbery_a_r009_normal-20260913T020720Z`.
  Mobile 5 ran YOLO/tracking throughout the relevant doorway footage. It
  produced vehicle-presence candidates but no valid disembark event: the
  vehicle passes the doorway view and the later person observation cannot be
  reliably associated with a person separating from that vehicle. The run did
  not complete.
- Final simplified Burglary-A r009 verification:
  `.fable/runs/network/robbery_a_r009_normal-20260913T033611Z`.
  With embedded Mobile 4 audio, Orin 6 audio, 0.2x synchronized pacing,
  confidence-gated/deduplicated presence, and the observable five-stage CE,
  six instances completed through vehicle present, person present, alarm,
  departing person present, and vehicle exit. `validation.json` passed and the
  completed event interval was 16:11:00.064641--16:11:31.434187 UTC.
- Real-time Burglary-A r009 verification:
  `.fable/runs/network/robbery_a_r009_normal-20260913T035953Z`.
  Continuous audio classification plus a 90-second event-time alarm buffer
  allowed the same five-stage event to complete at 1.0x. `validation.json`
  passed. The directory contains both `devices-dashboard.mp4` and
  `ce-dashboard.mp4` for presentation review.
- Corrected Burglary-A r008 rerun:
  `.fable/runs/network/robbery_a_r008_normal-20260913T021139Z`.
  Mobile 5 again ran detections/tracking. It created 168 vehicle-presence
  candidates, but none advanced through disembark, so this recording also did
  not complete. Its dashboard video is saved in that run directory.
- Burglary-B r015: `.fable/runs/network/robbery_b_r015_normal-20260913T014333Z`.
  Three vehicle arrivals were detected but no person/disembark transition.
  The fixed-camera/audio capture begins 26 seconds after the catalog start and
  Orin-6 contains no alarm-like one-second window above 0.25, so the necessary
  beginning may be absent. It did not complete and is not presentation-verified.
- Mobile-camera contact sheets and raw predicates are saved under
  `.fable/runs/presentation-evidence/robbery-alternatives-review/`.

## 2024 evaluation results

The 2024 campaign contains three CE families. One reviewed recording from each
family was exercised against the current runtime on 2026-09-13.

### Verified after visibility fix: route convoy — `convoy1`

- Ground-truth experiment: `20241008-route-convoy-1-r012`
- Command: `./run --plan network_nominal --case convoy1`
- Runtime result: passed all four leaves (leader enters/exits, follower
  enters/exits).
- Run: `.fable/runs/network/convoy1_normal-20260913T001530Z`
- Dashboard recording:
  `.fable/runs/network/convoy1_normal-20260913T001530Z/devices-dashboard.mp4`
- Caveat: tracker fragmentation creates many equivalent completed candidates;
  the UI's lead candidate should be used for presentation rather than the raw
  candidate count.
- Evidence audit: the completed instances are not visually defensible. The
  earliest candidate treats brief track loss as exit, and the longest candidate
  has no clearly visible follower at its alleged enter/exit times. Do not use
  this case for presentation yet. See
  `.fable/runs/presentation-evidence/convoy1-r012/` and
  `.fable/runs/presentation-evidence/convoy1-r012-reviewed/`.
- Post-fix run: `.fable/runs/network/convoy1_normal-20260913T010544Z`.
  The stricter boundary-crossing logic produced two completions; the clearest
  binds silver leader `track_4` and dark follower `track_18`. All four stages
  are visually consistent in the exact-time images under
  `.fable/runs/presentation-evidence/convoy1-r012-post-fix/`.

### Not yet verified: three-vehicle convergence — `vehicle_convergence1`

- Ground-truth experiment: `20241008-vehicle-convergence-3-r006`
- Command tested: `./run --plan network_nominal --case vehicle_convergence1`
- Run: `.fable/runs/network/vehicle_convergence_normal-20260913T001849Z`
- Result: failed to complete. Six vehicle-arrival candidates were created, but
  none satisfied all three pairwise-near leaves required by the current
  three-vehicle definition.
- Likely next step: choose a fixed camera that contains all three vehicles at
  the dwell point, or implement cross-camera convergence association.

### Not yet verified: two-vehicle chase — `two_vehicle_chase1`

- Ground-truth experiment: `20241009-two-vehicle-chase-5-r008` (usable)
- Command tested: `./run --plan network_nominal --case two_vehicle_chase1`
- Run: `.fable/runs/network/two_vehicle_chase1_normal-20260913T002210Z`
- Result: failed to complete. It created 29 leader-arrival candidates but no
  sustained local `follows` match.
- Cause: the vehicles traverse successive fixed-camera views; the implemented
  `follows_local_geometry` predicate expects both vehicles concurrently in one
  view. This case needs cross-sensor route/transition reasoning.
