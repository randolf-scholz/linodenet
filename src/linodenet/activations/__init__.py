r"""Implementations of activation functions.

Notes:
    - See `linodenet.activations.functional` for functional implementations.
    - See `linodenet.activations.modular` for module-based  implementations.
"""

__all__ = [
    # Sub-Modules
    "base",
    "functional",
    "modules",
    # Constants
    "ACTIVATIONS",
    "FUNCTIONAL_ACTIVATIONS",
    "MODULAR_ACTIVATIONS",
    "TORCH_ACTIVATIONS",
    "TORCH_FUNCTIONAL_ACTIVATIONS",
    "TORCH_MODULAR_ACTIVATIONS",
    # ABCs & Protocols
    "Activation",
    "ActivationBase",
    # Classes
    "HardBend",
    "GeGLU",
    "ReGLU",
    # Functions
    "geglu",
    "hard_bend",
    "reglu",
    # utils
    "get_activation",
]

from linodenet.activations import base, functional, modules
from linodenet.activations._torch_imports import (
    TORCH_ACTIVATIONS,
    TORCH_FUNCTIONAL_ACTIVATIONS,
    TORCH_MODULAR_ACTIVATIONS,
)
from linodenet.activations.base import Activation, ActivationBase
from linodenet.activations.functional import geglu, hard_bend, reglu
from linodenet.activations.modules import GeGLU, HardBend, ReGLU

FUNCTIONAL_ACTIVATIONS: dict[str, Activation] = {
    **TORCH_FUNCTIONAL_ACTIVATIONS,
    "reglu": reglu,
    "geglu": geglu,
    "hard_bend": hard_bend,
}
r"""Dictionary containing all available functional activations."""

MODULAR_ACTIVATIONS: dict[str, type[Activation]] = {
    **TORCH_MODULAR_ACTIVATIONS,
    "HardBend": HardBend,
    "GeGLU": GeGLU,
    "ReGLU": ReGLU,
}
r"""Dictionary containing all available activations."""

ACTIVATIONS: dict[str, Activation | type[Activation]] = {
    **TORCH_MODULAR_ACTIVATIONS,
    **MODULAR_ACTIVATIONS,
    **TORCH_FUNCTIONAL_ACTIVATIONS,
    **FUNCTIONAL_ACTIVATIONS,
}
r"""Dictionary containing all available activations."""


def get_activation(activation: object, /) -> Activation:
    r"""Get an activation function by name."""
    match activation:
        case type() as cls:
            return cls()
        case str(name):
            return get_activation(ACTIVATIONS[name])
        case Activation() as fn:
            return fn
        case _:
            raise TypeError(f"Invalid activation: {activation!r}")
