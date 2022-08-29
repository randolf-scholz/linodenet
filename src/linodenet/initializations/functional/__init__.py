r"""Initializations for the Linear ODE Networks.

All initializations are normalized such that if $x∼𝓝(0,1)$, then $Ax∼𝓝(0,1)$ as well.

Notes
-----
Contains initializations in functional form.
  - See `~linodenet.initializations.modular` for modular implementations.
"""

__all__ = [
    # Types
    "SizeLike",
    "FunctionalInitialization",
    # Constants
    "FunctionalInitializations",
    # Functions
    "canonical_skew_symmetric",
    "diagonally_dominant",
    "gaussian",
    "low_rank",
    "orthogonal",
    "skew_symmetric",
    "special_orthogonal",
    "symmetric",
]

from typing import Callable, Final

from torch import Tensor

from linodenet.initializations.functional._functional import (
    SizeLike,
    canonical_skew_symmetric,
    diagonally_dominant,
    gaussian,
    low_rank,
    orthogonal,
    skew_symmetric,
    special_orthogonal,
    symmetric,
)

FunctionalInitialization = Callable[[SizeLike], Tensor]  # SizeLike to matrix
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
