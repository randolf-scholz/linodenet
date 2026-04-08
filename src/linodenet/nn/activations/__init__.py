r"""Implementations of activation functions.

Notes:
    - See `linodenet.activations.functional` for functional implementations.
    - See `linodenet.activations.modular` for module-based  implementations.
"""

__all__ = [
    # Constants
    "ACTIVATIONS",
    "ACTIVATION_FNS",
    "ACTIVATION_FNS_WITH_ARGS",
    # ABCs & Protocols
    "Activation",
    "Activations",
    "ActivationBase",
    "GenericActivation",
]


from .base import (
    ACTIVATION_FNS,
    ACTIVATION_FNS_WITH_ARGS,
    ACTIVATIONS,
    Activation,
    ActivationBase,
    Activations,
    GenericActivation,
)
