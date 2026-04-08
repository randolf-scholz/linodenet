r"""Implementations of activation functions.

Notes:
    - See `linodenet.activations.functional` for functional implementations.
    - See `linodenet.activations.modular` for module-based  implementations.
"""

__all__ = [
    # Sub-Modules
    "base",
    # Constants
    "ACTIVATIONS",
    "ACTIVATION_FNS",
    "ACTIVATION_FNS_WITH_ARGS",
    # ABCs & Protocols
    "Activation",
    "ActivationBase",
    "GenericActivation",
    # utils
    "get_activation",
]


from . import base
from .base import (
    ACTIVATION_FNS,
    ACTIVATION_FNS_WITH_ARGS,
    ACTIVATIONS,
    Activation,
    ActivationBase,
    GenericActivation,
    get_activation,
)
