"""Phase-7 vehicle perception and predicate providers.

Optional model dependencies are loaded lazily; importing this package does not
require CUDA, Ultralytics, Roboflow Trackers, PyTorch, or OpenCLIP.
"""

from .association import CrossSensorIdentityAssociator
from .descriptors import (
    DeterministicDescriptorProvider,
    FastReidEntityDescriptor,
    OpenClipVisualDescriptor,
    TorchScriptVehicleReIDDescriptor,
)
from .detector import DEFAULT_YOLO_VARIANTS, LegacyReplayYoloAdapter, UltralyticsYoloDetector
from .follows import FollowsLocalGeometryEvaluator, summarize_tracks
from .geometry import (
    DwellEvaluator,
    MotionStateEvaluator,
    PairwiseDistanceEvaluator,
    PassReferenceEvaluator,
    TrackLifecycleExitEvaluator,
    ReferenceLine,
    RelativeOrderEvaluator,
    RouteMapMatcher,
    RoutePolyline,
    ZoneMembershipEvaluator,
    ZoneTransitionEvaluator,
)
from .models import (
    BoundingBox,
    Detection,
    DetectionFrame,
    DescriptorRecord,
    DescriptorSet,
    EntityAssociation,
    EntityAssociationSet,
    Point2D,
    PredicateObservation,
    TrackObservation,
    TrackSet,
    VehicleZone,
)
from .replay import JsonlDetectionStore, RetrospectiveVehicleExecutor
from .tracker import DetectionReplayTracker, RoboflowTrackerAdapter

__all__ = [
    "BoundingBox",
    "CrossSensorIdentityAssociator",
    "DEFAULT_YOLO_VARIANTS",
    "Detection",
    "DetectionFrame",
    "DetectionReplayTracker",
    "DescriptorRecord",
    "DescriptorSet",
    "DeterministicDescriptorProvider",
    "DwellEvaluator",
    "EntityAssociation",
    "EntityAssociationSet",
    "FollowsLocalGeometryEvaluator",
    "FastReidEntityDescriptor",
    "JsonlDetectionStore",
    "LegacyReplayYoloAdapter",
    "MotionStateEvaluator",
    "OpenClipVisualDescriptor",
    "PairwiseDistanceEvaluator",
    "PassReferenceEvaluator",
    "TrackLifecycleExitEvaluator",
    "Point2D",
    "PredicateObservation",
    "ReferenceLine",
    "RelativeOrderEvaluator",
    "RetrospectiveVehicleExecutor",
    "RoboflowTrackerAdapter",
    "RouteMapMatcher",
    "RoutePolyline",
    "TorchScriptVehicleReIDDescriptor",
    "TrackObservation",
    "TrackSet",
    "UltralyticsYoloDetector",
    "VehicleZone",
    "ZoneMembershipEvaluator",
    "ZoneTransitionEvaluator",
    "summarize_tracks",
]
