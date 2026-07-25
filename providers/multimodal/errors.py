"""Phase-8 provider exceptions."""


class MultimodalProviderError(RuntimeError):
    """Base error for multimodal provider execution."""


class InvalidAudioInput(MultimodalProviderError, ValueError):
    """Raised when audio input does not satisfy a provider contract."""


class OptionalDependencyError(MultimodalProviderError, ImportError):
    """Raised when an optional model/runtime dependency is unavailable."""


class InteractionStateError(MultimodalProviderError, ValueError):
    """Raised for invalid or out-of-order interaction state."""
