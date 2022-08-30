r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if $x∼𝓝(0,1)$, then $Ax∼𝓝(0,1)$ as well.

Lemma
~~~~~

In this case: $e^{A}$



Notes
-----
Contains initializations in both modular and functional form.
  - See `~linodenet.initializations.functional` for functional implementations.
  - See `~linodenet.initializations.modular` for modular implementations.
"""

__all__ = [
    # Types
    "Initialization",
    "FunctionalInitialization",
    "ModularInitialization",
    # Constants
    "FunctionalInitializations",
    "ModularInitializations",
    "Initializations",
    # Sub-Modules
    "functional",
    "modular",
    # Functions
    "canonical_skew_symmetric",
    "diagonally_dominant",
    "gaussian",
    "low_rank",
    "orthogonal",
    "skew_symmetric",
    "special_orthogonal",
    "symmetric",
    # Classes
]

from typing import Callable, Final, Union

from torch import Tensor, nn

from linodenet.initializations import functional, modular
from linodenet.initializations.functional import (
    canonical_skew_symmetric,
    diagonally_dominant,
    gaussian,
    low_rank,
    orthogonal,
    skew_symmetric,
    special_orthogonal,
    symmetric,
)

ModularInitialization = nn.Module
r"""Type hint for modular regularizations."""

ModularInitializations: Final[dict[str, type[ModularInitialization]]] = {}
r"""Dictionary of all available modular metrics."""

FunctionalInitialization = Callable[
    [int | tuple[int, ...]], Tensor
]  # SizeLike to matrix
r"""Type hint for Initializations."""

FunctionalInitializations: Final[dict[str, FunctionalInitialization]] = {
    "canonical_skew_symmetric": canonical_skew_symmetric,
    "diagonally_dominant": diagonally_dominant,
    "gaussian": gaussian,
    "low_rank": low_rank,
    "orthogonal": orthogonal,
    "skew-symmetric": skew_symmetric,
    "special-orthogonal": special_orthogonal,
    "symmetric": symmetric,
}
r"""Dictionary containing all available initializations."""

Initialization = Union[FunctionalInitialization, ModularInitialization]
r"""Type hint for initializations."""

Initializations: Final[
    dict[str, Union[FunctionalInitialization, type[ModularInitialization]]]
] = {
    **FunctionalInitializations,
    **ModularInitializations,
}
r"""Dictionary of all available initializations."""
