r"""Models for the latent dynamical system."""

__all__ = [
    # Constants
    "FLOWS",
    # ABCs & Protocols
    "ContinuousFlow",
    "DiscreteFlow",
    "FlowBase",
    "Flow",
    # Classes
    "LinearFlow",
]

from .base import Flow, FlowBase
from .continuous import ContinuousFlow
from .discrete import DiscreteFlow
from .linear import LinearFlow

FLOWS: dict[str, type[Flow]] = {
    "LinearFlow" : LinearFlow,
}  # fmt: skip
r"""Dictionary of all available system components."""
