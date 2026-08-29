# FABLE Complex-Event Language

This folder contains the declarative language for describing **what a complex event means**.

A CE definition contains only:

1. the CE name,
2. its logical object roles, and
3. a recursive pattern built from structure operators and semantic predicates.

Everything about **how**, **where**, and **when** to obtain the needed evidence belongs to later FABLE runtime layers.

---

# Quick start

A minimal event looks like this:

```yaml
version: 1
event: two_vehicle_chase

roles:
  LEADER:
    class: vehicle
  FOLLOWER:
    class: vehicle

pattern:
  seq:
    - enters:
        object: LEADER

    - for:
        duration: 3s
        pattern:
          follows:
            leader: LEADER
            follower: FOLLOWER
            max_gap_m: 30
```

Load it with:

```python
from fable.language import load_event

event = load_event("ce_definitions/two_vehicle_chase.yaml")

print(event.name)          # two_vehicle_chase
print(event.roles)         # {'LEADER': 'vehicle', 'FOLLOWER': 'vehicle'}
print(event.pattern.op)    # seq
```

The authored role form is `ROLE: {class: ...}`. The parser normalizes this to a compact `role -> class` mapping internally.

JSON is also accepted because JSON is valid YAML, although YAML is the recommended authoring format.

---

# Language grammar

At a high level:

```text
Event :=
    version
    event
    description?
    roles
    pattern

Role :=
    ROLE_NAME:
      class: object_class

Pattern :=
      Predicate
    | Seq
    | All
    | Any
    | KOfN
    | Within
    | For

Seq :=
    seq: [Pattern, Pattern, ...]

All :=
    all: [Pattern, Pattern, ...]

Any :=
    any: [Pattern, Pattern, ...]

KOfN :=
    k_of_n:
      k: Integer
      patterns: [Pattern, Pattern, ...]

Within :=
    within:
      min: Duration?       # optional
      max: Duration        # required
      pattern: Pattern

For :=
    for:
      duration: Duration
      pattern: Pattern
```

Durations must include units:

```text
500ms
3s
5m
1h
```

Bare numbers are rejected so there is no ambiguity about whether `30` means milliseconds or seconds.

---

# Roles, classes, and literals

## Logical roles are UPPERCASE

Roles are identity-bearing variables that can be reused across predicates:

```yaml
roles:
  VEHICLE_A:
    class: vehicle
  PERSON_A:
    class: person
  PACKAGE:
    class: package
```

The same role name means the same logical object. Different role names mean different logical objects.

```yaml
pattern:
  seq:
    - enters:
        object: VEHICLE_A
    - exits:
        object: VEHICLE_A
```

The vehicle that exits must therefore be the same logical vehicle that entered.

## `class` is an object-class constraint, not a FABLE data type

All declared roles are object roles. Their `class` describes what kind of object the role may bind to.

Examples include:

```text
person
vehicle
car
truck
dog
bicycle
backpack
package
```

The vocabulary is intentionally open. FABLE is not limited to `person` and `vehicle`.

The CE language also does **not** hard-code a detector ontology such as COCO. A later provider layer maps a semantic class to the labels understood by a concrete provider. For example:

```text
DOG_A has class dog
        ↓
provider selected by FABLE: COCO object detector
        ↓
provider mapping: semantic class dog -> COCO label dog
```

A broader semantic class can map to several provider labels:

```text
semantic class vehicle -> {car, truck, bus, motorcycle}
```

That mapping is deliberately outside the language parser. The CE only states the semantic class it needs.

## Predicates, event names, classes, and literals are lowercase

```yaml
event: drive_up_shooting
```

```yaml
audio_event:
  class: gunshot
```

Audio classes are literals rather than object roles because a sound event such as `gunshot` normally does not have persistent object identity.

---

# Structure operators

## `seq`

```yaml
seq:
  - A
  - B
  - C
```

means:

```text
A, then B, then C
```

The list order is semantically meaningful. `seq` requires at least two children.

Example:

```yaml
pattern:
  seq:
    - disembarks:
        person: PERSON
        vehicle: VEHICLE
    - audio_event:
        class: gunshot
    - boards:
        person: PERSON
        vehicle: VEHICLE
```

## `all`

```yaml
all:
  - A
  - B
```

means both patterns must occur, but their order is not specified.

`all` does **not** mean simultaneous, and it does not wait forever. The first direct child that becomes satisfied starts a **5-minute join window**. All remaining children must become satisfied within that same window or that attempt at the `all` fails.

Example:

```yaml
all:
  - enters:
      object: VEHICLE_A
  - enters:
      object: VEHICLE_B
```

If `VEHICLE_A` enters first, `VEHICLE_B` must enter within the next five minutes, and vice versa.

The 5-minute default is defined once in `pattern_parser.py` as `DEFAULT_JOIN_WINDOW_MS`. The parsed AST stores the resolved `window_ms`, so later runtime code does not need its own hidden copy of this assumption.

## `any`

```yaml
any:
  - A
  - B
```

means any one branch is sufficient.

Example:

```yaml
any:
  - audio_event:
      class: gunshot
  - audio_event:
      class: alarm
```

## `k_of_n`

```yaml
k_of_n:
  k: 2
  patterns:
    - A
    - B
    - C
```

means any **2 of the 3** child patterns are sufficient.

As with `all`, the first satisfied direct child starts the default **5-minute join window**. The operator must collect `k` distinct satisfied children before that window expires.

`k` must be at least 1 and cannot exceed the number of listed patterns.

## `within`

`within` bounds how long an activated child has to complete:

```yaml
within:
  max: 60s
  pattern:
    enters:
      object: FOLLOWER
```

A minimum can also be supplied:

```yaml
within:
  min: 30s
  max: 5m
  pattern:
    enters:
      object: VEHICLE
```

Within a `seq`, the wrapper becomes active when the previous sequence stage completes.

## `for`

`for` expresses sustained evidence:

```yaml
for:
  duration: 3s
  pattern:
    follows:
      leader: LEADER
      follower: FOLLOWER
```

The language parser records the duration in the AST. A later runtime is responsible for determining whether the child predicate held for that duration.

---

# Semantic predicates

The author-facing predicate vocabulary is defined in `predicates.yaml`.

All identity-bearing visual predicate arguments have language type `visual_object`; they therefore refer to an UPPERCASE role declared in `roles`. `visual_object` is a language-level reference type, not an object class. The role's `class` field supplies the semantic class (`dog`, `vehicle`, `person`, etc.). Argument names still describe the role the visual object plays in that relation.

| Predicate | Arguments | Meaning |
|---|---|---|
| `present` | `object` | Object is currently visible in the relevant camera view |
| `enters` | `object` | Object transitions from not visible to visible |
| `exits` | `object` | Object transitions from visible to not visible |
| `moving` | `object` | Object exhibits meaningful motion over an observation interval |
| `near` | `object_a`, `object_b` | Two objects are spatially proximate |
| `follows` | `leader`, `follower` | One object follows another; current FABLE examples normally use vehicles |
| `boards` | `person`, `vehicle` | A person boards a vehicle |
| `disembarks` | `person`, `vehicle` | A person leaves a vehicle |
| `transfer` | `item`, `giver`, `receiver` | Custody of an item changes between two object roles |
| `conversation` | `participant_a`, `participant_b` | Two participants converse |
| `audio_event` | `class` | Classified sound such as `gunshot` or `alarm` |

Optional numeric/string arguments such as `max_gap_m` and `minimum_confidence` are listed in `predicates.yaml` as part of the same argument mapping.

## `present`, `enters`, `exits`, and `moving` are different

These predicates should not be treated as aliases:

- `present(OBJECT)` means the object is visible now.
- `enters(OBJECT)` means FABLE observed a transition from not visible to visible.
- `exits(OBJECT)` means FABLE observed a transition from visible to not visible.
- `moving(OBJECT)` means the object shows meaningful motion over an observation interval.

A vehicle that is already parked in view when a video begins can therefore satisfy `present` while not satisfying `enters` or `moving`.

There is intentionally no generic `passes` predicate in the v1 vocabulary. A bare `passes(OBJECT)` does not specify what the object is crossing. If a CE means that an object enters and later leaves a view, write those transitions explicitly with `seq`. If a future CE genuinely needs crossing relative to a calibrated line or region, that should be introduced as a separate predicate with an explicit reference rather than hidden inside `passes`.

## Predicate arguments are semantic names, not types

For example:

```yaml
near:
  object_a: PERSON_A
  object_b: VEHICLE_A
```

Both arguments have language type `visual_object`; `object_a` and `object_b` simply name their positions in the relation.

Similarly:

```yaml
transfer:
  item: PACKAGE
  giver: PERSON_A
  receiver: PERSON_B
```

uses `giver` and `receiver` instead of ambiguous names such as `source` and `destination`.

The predicate catalog contains only semantic signatures and literal validation. It does not contain provider families, artifact types, placement, or binding modes.

---

# Example: package exchange

```yaml
version: 1
event: package_exchange

description: Two vehicles arrive, a package is transferred, and the receiver leaves in the second vehicle.

roles:
  VEHICLE_A:
    class: vehicle
  VEHICLE_B:
    class: vehicle
  PERSON_A:
    class: person
  PERSON_B:
    class: person
  PACKAGE:
    class: package

pattern:
  seq:
    - all:
        - enters:
            object: VEHICLE_A
        - enters:
            object: VEHICLE_B

    - transfer:
        item: PACKAGE
        giver: PERSON_A
        receiver: PERSON_B

    - boards:
        person: PERSON_B
        vehicle: VEHICLE_B

    - exits:
        object: VEHICLE_B
```

---

# Example: repeated visit

```yaml
version: 1
event: repeated_visit

roles:
  VEHICLE:
    class: vehicle

pattern:
  seq:
    - enters:
        object: VEHICLE

    - exits:
        object: VEHICLE

    - within:
        min: 30s
        max: 5m
        pattern:
          enters:
            object: VEHICLE

    - exits:
        object: VEHICLE
```

The repeated use of `VEHICLE` means the same logical object must satisfy both visits. The `within` applies to the second `enters`, so the re-entry must happen 30 seconds to 5 minutes after the first exit. How FABLE performs re-identification is outside the event definition.

---

# How to create a new complex event

## Step 1: Write the event in plain language

For example:

> A person gets out of a vehicle, a gunshot occurs, that same person gets back into that same vehicle, and the vehicle leaves.

## Step 2: Declare the logical object roles

```yaml
roles:
  PERSON:
    class: person
  VEHICLE:
    class: vehicle
```

Do not add cameras, source IDs, providers, or local tracker IDs.

## Step 3: Choose semantic predicates

The plain-language steps map to:

```text
disembarks(PERSON, VEHICLE)
audio_event(gunshot)
boards(PERSON, VEHICLE)
exits(VEHICLE)
```

## Step 4: Compose them with structure operators

```yaml
pattern:
  seq:
    - disembarks:
        person: PERSON
        vehicle: VEHICLE
    - audio_event:
        class: gunshot
    - boards:
        person: PERSON
        vehicle: VEHICLE
    - exits:
        object: VEHICLE
```

## Step 5: Save the definition

Place it in:

```text
ce_definitions/<event_name>.yaml
```

## Step 6: Validate it

```python
from fable.language import load_event

load_event("ce_definitions/drive_up_shooting.yaml")
```

The parser reports path-specific errors for unknown roles, unknown predicate arguments, malformed role declarations, missing required arguments, and malformed temporal syntax.

---

# Adding a new primitive predicate

Add its semantic signature to `predicates.yaml` only when there is, or will immediately be, a runtime implementation for it.

Simple required visual-object arguments can use shorthand:

```yaml
predicates:
  stopped:
    description: The object is stationary.
    arguments:
      object: visual_object
```

Arguments with validation options use the expanded form:

```yaml
predicates:
  audio_event:
    description: A classified sound is observed.
    arguments:
      class: audio_class
      minimum_confidence:
        type: number
        required: false
        minimum: 0
        maximum: 1
```

Do not put provider information in this file. Provider-to-predicate mapping belongs to a later physical/runtime layer.

---

# Where the language implementation lives

```text
fable/language/
├── event_parser.py       # parses the whole CE document and role declarations
├── pattern_parser.py     # parses seq/all/any/k_of_n/within/for and predicate calls into the AST
├── predicates.py         # loads and validates semantic predicate signatures
├── predicates.yaml       # author-facing primitive predicate catalog
└── README.md             # this document
```

`pattern_parser.py` implements the **grammar and AST representation** of structure operators. It does not implement runtime matching/frontier state. Runtime semantics such as advancing a `seq`, collecting unfinished `all` branches, or expiring a join window belong in the later event-runtime layer.

# Parse-time vs. compile-time validation

The language now has two intentionally separate checks.

`load_event()` / `parse_event()` validate the CE language itself.  They answer:

> Is this a well-formed FABLE complex event?

They intentionally allow an open semantic class vocabulary.  Thus this is
syntactically valid:

```yaml
roles:
  CREATURE:
    class: dragon

pattern:
  moving:
    object: CREATURE
```

After parsing, call `compile_event()` (or use `load_and_compile_event()`).
Compilation checks the definition against
`fable/providers/provider_capabilities.yaml` and answers:

> Can the currently declared provider set execute this complex event?

```python
from fable.language import load_and_compile_event

event = load_and_compile_event("ce_definitions/two_vehicle_chase.yaml")
```

A `dragon` role is rejected at this stage because no enabled visual detector
advertises that semantic class.  The standard full-context YOLO capability does
advertise COCO classes such as `person`, `dog`, `bicycle`, `car`, etc., plus
FABLE semantic aliases such as `vehicle`.  Similarly,
`audio_event.class: dragon_roar` is rejected because the current audio predicate
provider advertises only configured semantic classes such as `gunshot` and
`alarm`.

Compilation also checks class restrictions imposed by specialized predicate
implementations.  For example, a dog can participate in `near`, but cannot fill
the `person` argument of the current `boards_person_vehicle` implementation.

`compile_event()` returns the same compact `Event` AST.  It does **not** pin a
provider.  Provider selection remains a later planning decision.
