"""Network-emulation integration boundary for FABLE.

The Mininet host uses Ubuntu's Python 3.8 while FABLE requires Python 3.11.
Keep package initialization dependency-free so ``python3 -m netwaggle.runner``
does not import the optional FABLE bridge in the privileged interpreter.
"""

__all__ = ["NetwaggleLinkObservation", "apply_link_observations"]


def __getattr__(name):
    if name in __all__:
        from .bridge import NetwaggleLinkObservation, apply_link_observations

        return {
            "NetwaggleLinkObservation": NetwaggleLinkObservation,
            "apply_link_observations": apply_link_observations,
        }[name]
    raise AttributeError(name)
