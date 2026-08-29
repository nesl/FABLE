"""Provider layer for the simplified FABLE rebuild."""
from .predicate_result import PredicateMatch
from .data_models import (
    BoundingBox,VideoFrame,Detection,DetectionFrame,Track,TrackFrame,AudioWindow,MultichannelAudioWindow,
    SpeechSegment,SpeakerEmbedding,DiarizedSpeechSegment,DiarizedSpeechWindow,TranscriptSegment,
    ImageCrop,EmbeddingVector,AudioLocalization,VisualBearing,InteractionEvidence,
)
from .object_detection import (
    YoloConfig,UltralyticsObjectDetectorProvider,YoloVehicleFast640Provider,YoloVehicleBalanced960Provider,
    YoloFullContext960Provider,PackageDetectorProvider,OptionalProviderDependency,
)
from .tracking import IoUTrackerProvider,MultiObjectTrackerProvider
from .visual_features import (
    CameraProjectionProvider,TrackCropExtractorProvider,VehicleReIDDescriptorProvider,PersonReIDDescriptorProvider,
    OpenClipVisualDescriptorProvider,DeterministicDescriptorBackend,FastReIDDescriptorBackend,
    TorchreidDescriptorBackend,OpenClipDescriptorBackend,
)
from .identity import IdentityAssociation,CrossSensorIdentityAssociationProvider,HostedVLMIdentityComparatorProvider
from .audio_classification import AudioEventClassifierProvider,AudioThreshold,DeterministicAudioBackend,YamNetBackend
from .speech_processing import VoiceActivityDetectorProvider,SpeakerEmbeddingProvider,SpeakerDiarizationProvider,KeywordOrASRProvider
from .audio_localization import GccPhatAudioLocalizerProvider,AudioVisualAssociationProvider
from .provider_inventory import ProviderInfo,load_provider_inventory
from .provider_capabilities import (
    ProviderCapabilityCatalog,
    canonical_semantic_class,
    load_provider_capabilities,
    native_labels_for_visual_class,
    predicate_providers,
    semantic_literal_values,
    supported_visual_classes,
    visual_providers_for_class,
)
from .predicate_implementations import *

CURRENT_PUBLIC_PREDICATES=frozenset({"present","enters","exits","moving","near","follows","boards","disembarks","transfer","conversation","audio_event"})

__all__=[name for name in globals() if not name.startswith("_")]
