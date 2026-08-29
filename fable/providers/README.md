# FABLE provider layer

This folder contains the code that turns raw sensing/model outputs into the
semantic predicates used by the CE runtime.

The design rule is:

```text
provider-internal data                     semantic boundary
----------------------                     -----------------
frames / detections / tracks  ────────►    PredicateMatch
embeddings / audio windows                 predicate + arguments
speech segments                            event time
                                           provider ID
                                           source IDs
                                           confidence
```

Only `PredicateMatch` crosses into the complex-event runtime.

## File organization

```text
providers/
├── predicate_result.py       # PredicateMatch: the one semantic provider result
├── data_models.py            # Small internal records: detections, tracks, audio, embeddings
├── object_detection.py       # YOLO variants + package detector
├── tracking.py               # IoU tracker + optional ByteTrack facade
├── visual_features.py        # projection, crop extraction, ReID/OpenCLIP descriptors
├── identity.py               # cross-camera embedding association + optional hosted VLM fallback
├── audio_classification.py   # YAMNet/deterministic audio classification -> audio_event
├── speech_processing.py      # VAD, speaker embedding, diarization, optional ASR/keywords
├── audio_localization.py     # GCC-PHAT and audio/visual bearing association
├── predicate_implementations/
│   ├── visibility.py         # present / enters / exits
│   ├── motion_relations.py   # moving / near / follows
│   ├── person_vehicle.py     # boards / disembarks
│   ├── transfer.py           # transfer support + custody-change result
│   └── conversation.py       # conversation
├── legacy_compatibility.py   # old zone/pass/route/dwell providers; not CE-v1 primitives
├── provider_inventory.yaml   # complete original 36-provider recovery/history inventory
├── provider_inventory.py     # tiny loader for the inventory
├── provider_capabilities.yaml# active compile-time capabilities + native labels
└── provider_capabilities.py  # loader/query helpers used by compile_event()
```

This replaces the previous ambiguous split between `visual.py`, `audio.py`, and
`predicates.py`. A filename now says *what type of computation is implemented*.

## Predicate provider naming

Public semantic implementations have explicit names:

| CE predicate | Current implementation |
|---|---|
| `present` | `PresentBasicProvider` (`present_basic`) |
| `enters` | `EntersBasicProvider` (`enters_basic`) |
| `exits` | `ExitsBasicProvider` (`exits_basic`) |
| `moving` | `MovingBasicProvider` (`moving_basic`) |
| `near` | `NearGeometryProvider` (`near_geometry`) |
| `follows` | `FollowsLocalGeometryProvider` / `FollowsCrossSensorProvider` |
| `boards` | `BoardsPersonVehicleProvider` (`boards_person_vehicle`) |
| `disembarks` | `DisembarksPersonVehicleProvider` (`disembarks_person_vehicle`) |
| `transfer` | `TransferCustodyProvider` (`transfer_custody`) |
| `conversation` | `ConversationAVProvider` (`conversation_av`) |
| `audio_event` | `AudioEventClassifierProvider` (`audio_event_classifier`) |

There is deliberately no public `LifecycleProvider` anymore. The three
visibility predicates have distinct provider identities even though they share a
small private state helper.

## What happened to `SpeakerTurn`?

The old name was valid terminology in diarization, but it was not obvious from
the data model. It has been renamed:

- `SpeechSegment`: VAD found speech in this time interval.
- `SpeakerEmbedding`: an embedding computed for one speech interval.
- `DiarizedSpeechSegment`: diarization assigned a speech interval to a speaker.
- `DiarizedSpeechWindow`: collection of diarized speech segments from one source.

So `DiarizedSpeechSegment` explicitly says both *what the data is* and *where it
came from in the speech pipeline*.

## Recovered model/perception providers

The rebuild now includes more than YOLO and YAMNet. Implementations or
pluggable adapters are present for:

- three YOLO configurations (`yolo_vehicle_fast_640`,
  `yolo_vehicle_balanced_960`, `yolo_full_context_960`),
- package detection,
- IoU tracking and optional Roboflow/Supervision ByteTrack,
- camera projection,
- track crop extraction,
- model-backed vehicle and person ReID descriptor generation,
- FastReID SBS-R50-IBN/VeRi and Torchreid OSNet-AIN/MSMT17 inference backends,
- OpenCLIP general visual descriptors,
- cross-sensor embedding association,
- bounded hosted-VLM identity comparison through an injectable client,
- YAMNet audio-event classification,
- voice activity detection,
- speaker embeddings,
- speaker diarization,
- optional keyword/ASR processing,
- GCC-PHAT audio localization,
- audio/visual association,
- person/vehicle interaction reasoning,
- package interaction/custody reasoning.

Heavy model libraries/checkpoints are loaded lazily. The pinned ReID model metadata lives under `providers/reid/`; provision the actual checkpoints with `python scripts/provision_reid_models.py`. Install the ReID Python stack with `pip install -e ".[reid]"`.

## Complete old provider inventory

`provider_inventory.yaml` contains **all 36 provider IDs from the previous FABLE
physical-provider catalog**. Each entry says where it lives in the rebuild and
whether it is:

- directly implemented,
- implemented with an optional model backend,
- replaced by a simpler/current predicate provider, or
- retained only for compatibility because its old predicate is not in the new
  CE language.

Examples that are retained but are *not* current CE-v1 primitives include zone
membership/transition, `passes`, route matching, dwell, and retrospective
`vehicle_present_before`. Keeping these in `legacy_compatibility.py` lets us
recover useful code without letting old semantics leak back into the new
language.

## Why model providers and predicate implementations are separate

A model provider answers questions such as:

```text
What objects are visible?
What track ID belongs to each object?
What audio labels are likely?
What embedding describes this crop?
```

A predicate implementation answers a semantic question such as:

```text
Did OBJECT enter?
Are OBJECT_A and OBJECT_B near?
Is LEADER following FOLLOWER?
Did PERSON board VEHICLE?
```

For example:

```text
YOLO -> tracker -> NearGeometryProvider -> PredicateMatch("near", ...)
```

The CE runtime never sees the YOLO boxes or tracker state.

## Provider capability and label catalog

`provider_capabilities.yaml` is the compile-time description of what the
currently configured provider set can understand.  It is deliberately separate
from `provider_inventory.yaml`:

- `provider_inventory.yaml` is the recovery/history inventory of old and new
  provider implementations.
- `provider_capabilities.yaml` is the small executable contract used by the CE
  compiler.

The capability catalog records three kinds of information:

1. **Visual detector label support.**  The default full-context YOLO provider
   advertises the standard COCO-80 native labels.  Native labels are exposed as
   snake-case semantic classes (`dog` -> `dog`, `traffic light` ->
   `traffic_light`), and explicit aliases can group several native labels under
   one semantic class.  For example:

   ```text
   semantic class vehicle
       -> car | motorcycle | bus | truck
   ```

   The package detector similarly advertises:

   ```text
   semantic class package
       -> backpack | handbag | suitcase
   ```

2. **Predicate implementations.**  Each public CE predicate has at least one
   enabled implementation provider.  Specialized implementations can constrain
   the semantic classes accepted by an argument.  For example,
   `boards_person_vehicle` requires `person=person` and a vehicle-like class for
   `vehicle`, while `near_geometry` accepts any observable visual class.

3. **Semantic literal support.**  `audio_event_classifier` currently guarantees
   the CE-level classes `gunshot` and `alarm`, each mapped to one or more native
   YAMNet labels.  YAMNet itself may expose a much larger class map at runtime,
   but a class is not considered usable in CE YAML until the predicate provider
   explicitly maps it into the semantic catalog.

The compiler queries this catalog; it does not inspect model tensors or choose
an execution plan.  If a deployment changes model weights or label spaces, its
capability catalog should be changed with it.

## Typed provider interfaces for planning

Each active provider in `provider_capabilities.yaml` now also declares small `inputs` and `outputs` lists, for example `video_frame -> detections`, `detections -> tracks`, and `tracks -> predicate_match:near`. These are ordinary catalog fields, not a stored provider graph. `fable.planning.provider_search` indexes the declarations at planning time and searches backward from the current frontier target. Adding a compatible provider therefore adds a planning alternative without editing a separate chain file.

## Identity after predicate results

Cross-sensor ReID providers in this folder produce identity associations, but
they do not directly modify CE instances.  The execution layer applies those
associations through `fable.execution.IdentityResolver`.  Predicate
implementations first emit ordinary `PredicateMatch` records with their local
track IDs; the identity resolver canonicalizes identity-bearing arguments before
the CE instance manager sees the match.

This intentionally separates three questions:

```text
ReID provider          -> are these two track IDs the same physical object?
IdentityResolver       -> what canonical object ID should CE reasoning use?
CEInstanceManager      -> are these candidate complex-event occurrences duplicates?
```

The third question is not answered merely because ReID says two object IDs are
the same.
