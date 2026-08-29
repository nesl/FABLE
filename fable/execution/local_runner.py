"""Local closed-loop execution for the simplified FABLE rebuild.

This runner closes the local loop using the same dynamic provider search and
physical planner used by distributed deployments.  The default RuntimeState is
a single local node, so tests can exercise planning without a cluster:

    frontier -> active providers -> PredicateMatch -> identity resolution
             -> CEInstanceManager -> new frontier

The runner coalesces shared local provider prefixes and activates/deactivates
continuation-only work as CE instances appear or expire.  It is not yet a
network/compute-aware planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

from fable.language.event_parser import Event
from fable.providers.audio_classification import AudioEventClassifierProvider
from fable.providers.data_models import AudioWindow, DiarizedSpeechWindow, TrackFrame, ImageCrop
from fable.providers.object_detection import (
    YoloFullContext960Provider,
    YoloVehicleBalanced960Provider,
    YoloVehicleFast640Provider,
)
from fable.providers.predicate_implementations import (
    BoardsPersonVehicleProvider,
    ConversationAVProvider,
    DisembarksPersonVehicleProvider,
    EntersBasicProvider,
    ExitsBasicProvider,
    FollowsLocalGeometryProvider,
    MovingBasicProvider,
    NearGeometryProvider,
    PresentBasicProvider,
    TransferCustodyProvider,
)
from fable.providers.predicate_result import PredicateMatch
from fable.providers.tracking import MultiObjectTrackerProvider
from fable.runtime.ce_instance import CEInstance
from fable.runtime.frontier import ActiveFrontier, FrontierItem
from fable.runtime.instance_manager import CEInstanceManager

from .identity_resolver import IdentityResolver
from .reid import ReIDPipeline
from fable.planning import (
    NodeState, SourceState, RunningProvider, RuntimeState,
    PhysicalPlanner, load_provider_profiles,
)


def active_frontier_items(frontier: ActiveFrontier) -> tuple[FrontierItem, ...]:
    return tuple(frontier.discovery) + tuple(
        item
        for instance_id in sorted(frontier.continuation)
        for item in frontier.continuation[instance_id]
    )


@dataclass(frozen=True, slots=True)
class ActivationEvent:
    event_time: datetime
    action: str  # activate | deactivate
    provider_id: str


@dataclass(frozen=True, slots=True)
class RunnerUpdate:
    """Observable result of one local-runner input."""

    matches: tuple[PredicateMatch, ...] = ()
    produced_instances: tuple[CEInstance, ...] = ()
    completed_instances: tuple[CEInstance, ...] = ()
    activation_events: tuple[ActivationEvent, ...] = ()


class LocalRunner:
    """Execute one compiled CE locally using the dynamic physical planner."""

    def __init__(
        self,
        event: Event,
        *,
        instance_manager: CEInstanceManager | None = None,
        identity_resolver: IdentityResolver | None = None,
        audio_event_provider: AudioEventClassifierProvider | None = None,
        detector_factories: Mapping[str, Callable[[], object]] | None = None,
        tracker_factory: Callable[[], MultiObjectTrackerProvider] | None = None,
        physical_planner: PhysicalPlanner | None = None,
        runtime_state: RuntimeState | None = None,
        reid_pipeline: ReIDPipeline | None = None,
    ) -> None:
        self.event = event
        self.manager = instance_manager or CEInstanceManager(event)
        self.identity_resolver = identity_resolver or IdentityResolver()
        self.audio_event_provider = audio_event_provider

        self._detector_factories: dict[str, Callable[[], object]] = {
            "yolo_vehicle_fast_640": YoloVehicleFast640Provider,
            "yolo_vehicle_balanced_960": YoloVehicleBalanced960Provider,
            "yolo_full_context_960": YoloFullContext960Provider,
        }
        if detector_factories:
            self._detector_factories.update(detector_factories)
        self._tracker_factory = tracker_factory or (
            lambda: MultiObjectTrackerProvider(algorithm="iou")
        )
        self.physical_planner = physical_planner or PhysicalPlanner()
        self.reid_pipeline = reid_pipeline or ReIDPipeline()
        self.runtime_state = runtime_state or RuntimeState(
            nodes={"local": NodeState("local", "local")},
            sources={
                "local_video": SourceState("local_video", "local", "video_frame", "local", sample_bytes=250_000),
                "local_audio": SourceState("local_audio", "local", "audio_window", "local", sample_bytes=64_000),
                "local_multichannel_audio": SourceState("local_multichannel_audio", "local", "multichannel_audio", "local", sample_bytes=256_000),
            },
            profiles=load_provider_profiles(),
        )
        self._current_plan = None

        # Heavy/model providers are instantiated lazily.  Stateful tracking and
        # predicate evaluators are isolated per sensor source.
        self._detectors: dict[str, object] = {}
        self._trackers: dict[str, MultiObjectTrackerProvider] = {}
        self._predicate_providers: dict[tuple[str, str], object] = {}

        self._running_provider_ids: set[str] = set()
        self._activation_log: list[ActivationEvent] = []

    # ------------------------------------------------------------------
    # Frontier / activation
    # ------------------------------------------------------------------
    def current_frontier(self, now: datetime) -> ActiveFrontier:
        return self.manager.current_frontier(now)

    def sync(self, now: datetime) -> tuple[ActivationEvent, ...]:
        """Reconcile logically running providers with the current frontier."""

        frontier = self.manager.current_frontier(now)
        # Reuse information from the previous selected plan when ranking the
        # next one. Planning itself remains stateless.
        if self._current_plan is not None:
            self.runtime_state.running = tuple(
                RunningProvider(step.provider_id, step.node_id, step.source_ids)
                for step in self._current_plan.steps
            )
        plan = self.physical_planner.plan(frontier, self.runtime_state, now=now)
        self._current_plan = plan
        required = set(plan.provider_ids)

        events: list[ActivationEvent] = []
        for provider_id in sorted(required - self._running_provider_ids):
            event = ActivationEvent(now, "activate", provider_id)
            events.append(event)
            self._activation_log.append(event)
        for provider_id in sorted(self._running_provider_ids - required):
            event = ActivationEvent(now, "deactivate", provider_id)
            events.append(event)
            self._activation_log.append(event)
        self._running_provider_ids = required
        return tuple(events)

    @property
    def running_provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._running_provider_ids))

    @property
    def activation_log(self) -> tuple[ActivationEvent, ...]:
        return tuple(self._activation_log)

    @property
    def execution_plan(self):
        return self._current_plan

    # ------------------------------------------------------------------
    # Provider -> PredicateMatch -> identity -> CE runtime
    # ------------------------------------------------------------------
    def process_predicate_match(self, match: PredicateMatch) -> RunnerUpdate:
        """Canonicalize one provider result and feed it to the CE manager."""

        before_completed = len(self.manager.completed_instances())
        canonical = self.identity_resolver.canonicalize_match(match)
        if len(canonical.source_ids) == 1:
            for arg_name in canonical.classes:
                object_id = canonical.arguments.get(arg_name)
                if isinstance(object_id, str) and object_id:
                    self.runtime_state.object_sources[object_id] = canonical.source_ids[0]
        produced = self.manager.handle_match(canonical)
        activations = self.sync(canonical.event_time)
        completed = self.manager.completed_instances()[before_completed:]
        return RunnerUpdate((canonical,), produced, completed, activations)

    def process_predicate_matches(
        self,
        matches: Iterable[PredicateMatch],
    ) -> RunnerUpdate:
        all_matches: list[PredicateMatch] = []
        produced: list[CEInstance] = []
        completed: list[CEInstance] = []
        activations: list[ActivationEvent] = []
        for match in sorted(matches, key=lambda value: value.event_time):
            update = self.process_predicate_match(match)
            all_matches.extend(update.matches)
            produced.extend(update.produced_instances)
            completed.extend(update.completed_instances)
            activations.extend(update.activation_events)
        return RunnerUpdate(
            tuple(all_matches), tuple(produced), tuple(completed), tuple(activations)
        )

    # ------------------------------------------------------------------
    # Local visual pipeline
    # ------------------------------------------------------------------
    def process_image(
        self,
        image: Any,
        *,
        source_id: str,
        event_time: datetime,
        frame_id: str = "",
    ) -> RunnerUpdate:
        """Run the currently required local detector/tracker/predicate path."""

        initial_activations = self.sync(event_time)
        detector_id = self._selected_visual_detector()
        if detector_id is None:
            return RunnerUpdate(activation_events=initial_activations)

        detector = self._detector(detector_id)
        detection_frame = detector.detect(
            image, source_id=source_id, event_time=event_time, frame_id=frame_id
        )
        tracker = self._trackers.setdefault(source_id, self._tracker_factory())
        track_frame = tracker.update(detection_frame)
        update = self.process_track_frame(track_frame, _already_synced=True)
        return RunnerUpdate(
            update.matches,
            update.produced_instances,
            update.completed_instances,
            initial_activations + update.activation_events,
        )

    def process_track_frame(
        self,
        frame: TrackFrame,
        *,
        _already_synced: bool = False,
    ) -> RunnerUpdate:
        """Evaluate active track-based predicate implementations once per frame.

        This entry point is useful for replay/tests and for deployments where an
        upstream detector/tracker is already available.
        """

        initial_activations = () if _already_synced else self.sync(frame.event_time)
        frontier = self.manager.current_frontier(frame.event_time)
        items = active_frontier_items(frontier)
        needed = {item.predicate for item in items}

        matches: list[PredicateMatch] = []
        source = frame.source_id

        # Stateful providers that emit every matching object/relation are run
        # once per frame, regardless of how many CE instances need them.
        update_once = {
            "present": PresentBasicProvider,
            "enters": EntersBasicProvider,
            "exits": ExitsBasicProvider,
            "moving": MovingBasicProvider,
            "boards": BoardsPersonVehicleProvider,
            "disembarks": DisembarksPersonVehicleProvider,
            "transfer": TransferCustodyProvider,
        }
        for predicate, provider_cls in update_once.items():
            if predicate not in needed:
                continue
            provider = self._predicate_provider(source, predicate, provider_cls)
            matches.extend(provider.update(frame))

        # Parameterized pairwise predicates are evaluated for each unique
        # semantic requirement.  Duplicate frontier items with the same resolved
        # request do not trigger duplicate provider work.
        if "near" in needed:
            provider = self._predicate_provider(source, "near", NearGeometryProvider)
            for item in _unique_items(items, "near"):
                matches.extend(
                    provider.evaluate(
                        frame,
                        object_a_id=self._local_identity(item.arguments.get("object_a"), source),
                        object_b_id=self._local_identity(item.arguments.get("object_b"), source),
                        max_distance_m=_optional_float(item.parameters.get("max_distance_m")),
                        max_normalized_gap=_optional_float(item.parameters.get("max_normalized_gap")),
                    )
                )

        if "follows" in needed:
            provider = self._predicate_provider(
                source, "follows", FollowsLocalGeometryProvider
            )
            for item in _unique_items(items, "follows"):
                matches.extend(
                    provider.evaluate(
                        frame,
                        leader_id=self._local_identity(item.arguments.get("leader"), source),
                        follower_id=self._local_identity(item.arguments.get("follower"), source),
                        max_gap_m=float(item.parameters.get("max_gap_m", 15.0)),
                    )
                )

        update = self.process_predicate_matches(_unique_matches(matches))
        return RunnerUpdate(
            update.matches,
            update.produced_instances,
            update.completed_instances,
            initial_activations + update.activation_events,
        )

    # ------------------------------------------------------------------
    # Local audio / AV paths
    # ------------------------------------------------------------------
    def process_audio_window(self, window: AudioWindow) -> RunnerUpdate:
        initial_activations = self.sync(window.event_time)
        frontier = self.manager.current_frontier(window.event_time)
        items = [
            item for item in active_frontier_items(frontier)
            if item.predicate == "audio_event"
        ]
        if not items:
            return RunnerUpdate(activation_events=initial_activations)
        if self.audio_event_provider is None:
            raise RuntimeError(
                "audio_event is active but no AudioEventClassifierProvider was "
                "supplied to LocalRunner"
            )

        # Run at the least restrictive requested confidence; the CE matcher
        # still enforces each individual frontier item's minimum.
        minimums: dict[str, float] = {}
        for item in items:
            semantic_class = item.parameters.get("class")
            if not isinstance(semantic_class, str):
                continue
            value = float(item.parameters.get("minimum_confidence", 0.0))
            minimums[semantic_class] = min(minimums.get(semantic_class, value), value)
        matches = self.audio_event_provider.classify(
            window, minimum_confidence=minimums
        )
        update = self.process_predicate_matches(matches)
        return RunnerUpdate(
            update.matches,
            update.produced_instances,
            update.completed_instances,
            initial_activations + update.activation_events,
        )

    def process_conversation(
        self,
        tracks: TrackFrame,
        speech: DiarizedSpeechWindow,
    ) -> RunnerUpdate:
        initial_activations = self.sync(tracks.event_time)
        frontier = self.manager.current_frontier(tracks.event_time)
        items = [
            item for item in active_frontier_items(frontier)
            if item.predicate == "conversation"
        ]
        if not items:
            return RunnerUpdate(activation_events=initial_activations)
        provider = self._predicate_provider(
            tracks.source_id, "conversation", ConversationAVProvider
        )
        matches: list[PredicateMatch] = []
        for item in _unique_items(items, "conversation"):
            required_terms_raw = item.parameters.get("required_terms", "")
            required_terms = (
                (required_terms_raw,)
                if isinstance(required_terms_raw, str) and required_terms_raw
                else ()
            )
            matches.extend(
                provider.evaluate(
                    tracks,
                    speech,
                    participant_a_id=self._local_identity(
                        item.arguments.get("participant_a"), tracks.source_id
                    ),
                    participant_b_id=self._local_identity(
                        item.arguments.get("participant_b"), tracks.source_id
                    ),
                    max_distance_m=float(item.parameters.get("max_distance_m", 2.5)),
                    required_terms=required_terms,
                )
            )
        update = self.process_predicate_matches(_unique_matches(matches))
        return RunnerUpdate(
            update.matches,
            update.produced_instances,
            update.completed_instances,
            initial_activations + update.activation_events,
        )

    # ------------------------------------------------------------------
    # ReID / identity association
    # ------------------------------------------------------------------
    def merge_identity(self, left: str, right: str) -> str:
        """Apply one ReID association and recanonicalize stored CE bindings.

        This does *not* deduplicate CE instances.  A repeated occurrence by the
        same physical object remains a distinct candidate.
        """

        canonical = self.identity_resolver.merge(left, right)
        self.manager.recanonicalize_identities(self.identity_resolver.canonical)
        return canonical

    def apply_identity_associations(self, associations: Mapping[str, str]) -> None:
        self.identity_resolver.apply_associations(associations)
        self.manager.recanonicalize_identities(self.identity_resolver.canonical)

    def process_reid_crops(
        self,
        left_crops: Sequence[ImageCrop],
        right_crops: Sequence[ImageCrop],
        *,
        entity_kind: str,
    ):
        """Run model-backed ReID and apply resulting object-identity merges.

        ReID remains separate from CE-instance deduplication: this method only
        canonicalizes physical object identities and then rewrites stored CE
        bindings through the IdentityResolver.
        """

        records = self.reid_pipeline.associate(
            left_crops, right_crops, entity_kind=entity_kind
        )
        self.apply_identity_associations(
            {row.left_object_id: row.right_object_id for row in records}
        )
        return records

    def deduplicate_ce_instances(self) -> tuple[str, ...]:
        """Run the separate, conservative CE-instance deduplication step."""

        return self.manager.deduplicate_instances()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _selected_visual_detector(self) -> str | None:
        for provider_id in (
            "yolo_full_context_960",
            "yolo_vehicle_balanced_960",
            "yolo_vehicle_fast_640",
        ):
            if provider_id in self._running_provider_ids:
                return provider_id
        return None

    def _detector(self, provider_id: str) -> object:
        if provider_id not in self._detectors:
            try:
                factory = self._detector_factories[provider_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"no detector factory registered for {provider_id!r}"
                ) from exc
            self._detectors[provider_id] = factory()
        return self._detectors[provider_id]

    def _predicate_provider(
        self,
        source_id: str,
        predicate: str,
        provider_cls: type,
    ) -> object:
        key = (source_id, predicate)
        if key not in self._predicate_providers:
            self._predicate_providers[key] = provider_cls()
        return self._predicate_providers[key]

    def _local_identity(self, canonical: str | None, source_id: str) -> str | None:
        if canonical is None:
            return None
        aliases = self.identity_resolver.aliases(canonical)
        source_prefix = f"{source_id}:"
        for alias in aliases:
            if alias.startswith(source_prefix):
                return alias
        return canonical


def _unique_items(items: Sequence[FrontierItem], predicate: str) -> tuple[FrontierItem, ...]:
    seen: set[tuple[object, ...]] = set()
    result: list[FrontierItem] = []
    for item in items:
        if item.predicate != predicate:
            continue
        key = (
            item.predicate,
            tuple(sorted(item.arguments.items())),
            tuple(sorted(item.classes.items())),
            tuple(sorted((key, repr(value)) for key, value in item.parameters.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return tuple(result)


def _unique_matches(matches: Iterable[PredicateMatch]) -> tuple[PredicateMatch, ...]:
    seen: set[tuple[object, ...]] = set()
    result: list[PredicateMatch] = []
    for match in matches:
        key = (
            match.predicate,
            match.event_time,
            tuple(match.source_ids),
            tuple(sorted(match.arguments.items())),
            match.provider_id,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(match)
    return tuple(result)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
