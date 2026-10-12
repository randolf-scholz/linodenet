r"""Activation functions, modules, and registries.

Notes:
    - `imtskit.nn.activations.base` provides the activation protocol, enum, and
        registries of functions and module classes.
    - `imtskit.nn.activations.crelu` and `imtskit.nn.activations.misc` provide
        local activation implementations.
    - `imtskit.nn.activations.imported` provides wrappers for special functions.
"""

__all__ = [
    # Constants
    "ACTIVATIONS",
    "ACTIVATION_FNS",
    "ACTIVATION_FNS_WITH_ARGS",
    # ABCs & Protocols
    "Activation",
    "Activations",
    "GenericActivation",
]


from .base import (
    ACTIVATION_FNS,
    ACTIVATION_FNS_WITH_ARGS,
    ACTIVATIONS,
    Activation,
    Activations,
    GenericActivation,
)
