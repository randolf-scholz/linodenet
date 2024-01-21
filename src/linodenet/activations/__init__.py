r"""Implementations of activation functions.

Notes:
    Contains activations in both functional and modular form.
    - See `linodenet.activations.functional` for functional implementations.
    - See `linodenet.activations.modular` for modular implementations.
"""

__all__ = [
    # Sub-Modules
    "functional",
    # Constants
    "ACTIVATIONS",
    "FUNCTIONAL_ACTIVATIONS",
    "MODULAR_ACTIVATIONS",
    "TORCH_ACTIVATIONS",
    "TORCH_FUNCTIONAL_ACTIVATIONS",
    "TORCH_MODULAR_ACTIVATIONS",
    # ABCs & Protocols
    "Activation",
    "ActivationABC",
    # Classes
    "HardBend",
    # Functions
    "geglu",
    "reglu",
    "hard_bend",
]

from linodenet.activations import functional
from linodenet.activations._activations import Activation, ActivationABC, HardBend
from linodenet.activations._torch_imports import (
    TORCH_ACTIVATIONS,
    TORCH_FUNCTIONAL_ACTIVATIONS,
    TORCH_MODULAR_ACTIVATIONS,
)
from linodenet.activations.functional import geglu, hard_bend, reglu

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
}
r"""Dictionary containing all available activations."""

ACTIVATIONS: dict[str, Activation | type[Activation]] = {
    **TORCH_MODULAR_ACTIVATIONS,
    **MODULAR_ACTIVATIONS,
    **TORCH_FUNCTIONAL_ACTIVATIONS,
    **FUNCTIONAL_ACTIVATIONS,
}
r"""Dictionary containing all available activations."""
