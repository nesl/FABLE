from fable.common.enums import BindingCapability
from fable.planning.search.feasibility import _binding_mode_supported


def test_introduce_or_validate_is_disjunctive_for_provider_capabilities() -> None:
    required = BindingCapability.INTRODUCE_OR_VALIDATE
    assert _binding_mode_supported(required, {BindingCapability.INTRODUCE})
    assert _binding_mode_supported(required, {BindingCapability.VALIDATE})
    assert not _binding_mode_supported(required, {BindingCapability.CONSUME})
