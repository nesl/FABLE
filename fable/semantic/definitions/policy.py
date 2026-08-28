"""Shared post-completion policy metadata for authored trial definitions."""

from __future__ import annotations


def trial_rearm_annotations(*, clear_interval_ms: int = 7_500) -> dict[str, object]:
    """Return controller-facing reset policy kept outside CE semantics.

    The semantic graph completes before experimental return-to-start motion.
    Controllers may use this metadata to suppress a second occurrence until
    the scene is clear, while loggers can classify reset traffic separately.
    """

    if clear_interval_ms <= 0:
        raise ValueError("clear_interval_ms must be positive")
    return {
        "post_completion_policy": {
            "mode": "scene_clear_rearm",
            "clear_interval_ms": clear_interval_ms,
            "reset_interval_label": "TRIAL_RESET",
            "exclude_reset_from_positive_annotations": True,
        }
    }
