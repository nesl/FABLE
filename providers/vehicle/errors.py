"""Provider-specific exceptions."""


class VehicleProviderError(RuntimeError):
    """Base error for Phase-7 vehicle providers."""


class OptionalDependencyError(VehicleProviderError):
    """Raised when a requested real provider dependency is not installed."""


class ArtifactCompatibilityError(VehicleProviderError):
    """Raised when model/feature-space or coordinate-frame metadata conflicts."""


class InvalidProviderInput(VehicleProviderError):
    """Raised when a provider receives malformed or semantically invalid input."""
